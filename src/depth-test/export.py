import jax
import jax.numpy as jnp
import coremltools as ct

from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

from stablehlo_coreml import DEFAULT_HLO_PIPELINE

from .utils import dist
from .cml import CmlConfig, convert, jax_center_size_width_first

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros(CmlConfig.resolution, dtype=jnp.float32),)
jax_exported = export(jax_center_size_width_first)(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

# with open(dist / "jaxpr.mlir", "w") as fp:
#     fp.write(jax_center_size_width_first.lower(*input_shapes).as_text())

mil_program = convert(hlo_module, patch_tags=True, patch_output=True)

mil_args = mil_program.functions[
        mil_program.default_function_name].inputs.keys()
mil_arg0 = next(iter(mil_args))

pipeline = DEFAULT_HLO_PIPELINE
pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e2"})

import logging
from coremltools import _logger as logger
logger_level = logger.level
logger.setLevel(logging.ERROR)

cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=CmlConfig.opset_version,
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
    compute_precision=ct.precision.FLOAT32,
    pass_pipeline=pipeline,
    inputs=[ct.TensorType(
        mil_arg0, shape=CmlConfig.resolution,
        dtype=ct.converters.mil.mil.types.fp32)],
)

logger.setLevel(logger_level)

cml_model.save(str(dist / "partial.mlpackage"))

# separate slow graph conversion from interface changes
from .detailing import *
