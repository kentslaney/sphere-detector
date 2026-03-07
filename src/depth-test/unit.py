import jax, json
import jax.numpy as jnp
import coremltools as ct
import numpy as np

from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

from stablehlo_coreml import DEFAULT_HLO_PIPELINE

from .detect import Raster
from .utils import dist
from .cml import config, convert
from .integration import im4_cml

target = config.resolution

@jax.jit
def jax_density(x):
    return Raster(None, x, resolution=config.resolution).opt().predict()
    return jax.tree_util.tree_flatten(
            Raster(None, x, resolution=config.resolution).opt().points)[0]

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros(config.input_shape, dtype=config.input_dtype),)
jax_exported = export(jax_density)(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

mil_program = convert(hlo_module)

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
    minimum_deployment_target=ct.target.iOS18,
    # compute_units=ct.ComputeUnit.ALL,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    compute_precision=ct.precision.FLOAT32,
    pass_pipeline=pipeline,
    inputs=[ct.TensorType(
        mil_arg0, shape=config.resolution, dtype=config.input_cml_dtype)],
)

logger.setLevel(logger_level)

cml_out = cml_model.predict({"_arg0": np.array(im4_cml.depth.depth)})
jax_out = jax_density(im4_cml.depth.depth)
fmt_kw = {"sep": "\n", "end": "\n\n"}
print("CoreML", *[cml_out[k] for k in cml_model.output_description], **fmt_kw)
print("Jax", *jax_out, **fmt_kw)
