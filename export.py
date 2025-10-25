import pathlib, sys
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from static import Depth
sys.path.pop(0)

def jax_density(x):
    return Depth(x).density()

import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

import jax.numpy as jnp

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros((392, 518), dtype=jnp.float16),)
jax_exported = export(jax.jit(jax_density))(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

import coremltools as ct
from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    pass_pipeline=DEFAULT_HLO_PIPELINE,
)

