import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax import export

import coremltools as ct
from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

import jax.numpy as jnp
from .stateful import implicit, restores

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
    jax.jit(exported), disabled_checks=[
        export.DisabledSafetyCheck.custom_call("ffi_read")
    ]
)(..., *input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

print()
print(hlo_module)

mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    pass_pipeline=DEFAULT_HLO_PIPELINE,
)
