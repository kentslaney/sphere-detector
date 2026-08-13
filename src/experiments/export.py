import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax import export

import coremltools as ct
from coremltools.converters.mil import Builder as mb

from stablehlo_coreml.ops_register import register_stablehlo_op
from stablehlo_coreml.converter import convert
from stablehlo_coreml.translation_context import TranslationContext
from stablehlo_coreml import DEFAULT_HLO_PIPELINE, register_optimizations

import jax.numpy as jnp
from .stateful import implicit, restores
from ..sphere_detector.cml import MilInjector

from jaxlib.mlir.dialects.stablehlo import CustomCallOp

@restores(momentum1=jnp.ones(()))
def block1(x):
    global momentum1
    momentum1 += x
    return x + momentum1

@restores(momentum2=jnp.zeros(()))
def block2(x):
    global momentum2
    momentum2 += x
    return x + momentum2

@implicit
def exported(x):
    return block1(x) + block2(x)

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros(()),)
jax_exported = export.export(
    jax.jit(exported, donate_argnames=["state"]), disabled_checks=[
        export.DisabledSafetyCheck.custom_call("ffi_read")
    ]
)(exported(..., *input_shapes)[0], *input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

print()
print(hlo_module)

class StatefulIO(MilInjector):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.internal_io = {}

    @register_stablehlo_op
    def op_custom_call(self, context: TranslationContext, op: CustomCallOp):
        call_target = op.call_target_name.value
        if call_target == "ffi_read":
            placeholder = op.inputs[0]
            hlo_func = placeholder.owner.owner
            with hlo_func.context:
                attrs = hlo_func.arg_attrs[placeholder.arg_number]
            mapping = int(attrs['tf.aliasing_output'])
            assert placeholder.arg_number == mapping
            key = op.attributes["backend_config"].value
            state = context[placeholder.get_name()]
            state.set_name(key)
            self.internal_io[mapping] = state
            context.add_result(op.result, mb.read_state(input=state))
        else:
            return super().op_custom_call(context, op)

    def patch(self, *a):
        for k, v in self.internal_io.items():
            mb.coreml_update_state(state=v, value=a[k])
        return [x for i, x in enumerate(a) if i not in self.internal_io]

mil_program = StatefulIO(opset_version=ct.target.iOS18).convert(hlo_module)

print(mil_program)
exit(0)

register_optimizations()
cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    pass_pipeline=DEFAULT_HLO_PIPELINE,
)
