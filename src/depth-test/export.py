import jax, json
import jax.numpy as jnp
# TODO: consolidate ct.target.iOS18
# TODO: float16 internals
# jax.config.update("jax_enable_x64", True)
import coremltools as ct

from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

from stablehlo_coreml import DEFAULT_HLO_PIPELINE

from .detect import Config, Raster
from .utils import dist
from .mil import convert

target = Config.resolution

@jax.jit
def jax_density(x):
    x = x.reshape(target)
    confidence, coordinates = Raster(None, x, resolution=target).opt().predict()
    # confidence = jnp.astype(confidence, jnp.float16)
    coordinates /= jnp.tile(jnp.array(target), [1, 2])
    ll, hh = coordinates[:, 1::-1], coordinates[:, 3:1:-1]
    coordinates = jnp.hstack(((ll + hh) / 2, hh - ll + 1))
    # coordinates = jnp.astype(coordinates, jnp.float16)
    # return confidence.reshape((-1, 1)), coordinates.reshape((-1, 4))
    return confidence.reshape((1, 1, -1)), coordinates.T.reshape((1, 4, -1))

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros((1, 1) + target, dtype=jnp.float16),)
jax_exported = export(jax_density)(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

with open(dist / "jaxpr.mlir", "w") as fp:
    fp.write(jax_density.lower(*input_shapes).as_text())

mil_program = convert(hlo_module)

mil_args = mil_program.functions[
        mil_program.default_function_name].inputs.keys()
mil_arg0 = next(iter(mil_args))

pipeline = DEFAULT_HLO_PIPELINE
pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e2"})
# pipeline.remove_passes(['common::add_int16_cast'])

import logging
from coremltools import _logger as logger
# logger_level = logger.level
# logger.setLevel(logging.ERROR)
logging.getLogger("coremltools").disabled = True

cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
    # compute_units=ct.ComputeUnit.CPU_ONLY,
    pass_pipeline=pipeline,
    inputs=[ct.ImageType(
        mil_arg0, shape=target, color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
)

# logger.setLevel(logger_level)
logging.getLogger("coremltools").disabled = False

cml_model.save(str(dist / "partial.mlpackage"))

# separate slow graph conversion from interface changes
from .detailing import *
