import pathlib, sys
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from static import Depth, M2
sys.path.pop(0)

target = M2.target

import jax
import jax.numpy as jnp

@jax.jit
def jax_density(x):
    x = x.reshape(target)
    # return M2(jnp.array([]), x).depth.binned().nominate()
    return M2(jnp.array([]), x).depth.binned().bounds

from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros((1, 1) + target, dtype=jnp.float16),)
jax_exported = export(jax_density)(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

# print(jax_density.lower(*input_shapes).as_text())
# exit(0)

# TODO: patch via forked submodule to https://github.com/apple/coremltools/pull/2626
import coremltools as ct
from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)

pipeline = DEFAULT_HLO_PIPELINE
pipeline.remove_passes(['common::add_int16_cast'])

cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
    pass_pipeline=pipeline,
    inputs=[ct.ImageType(
        "_arg0", shape=target, color_layout=ct.colorlayout.GRAYSCALE_FLOAT16,
        channel_first=True)],
)

dist = local / "dist"
dist.mkdir(parents=True, exist_ok=True)

cml_model.save(str(dist / "centers.mlpackage"))
