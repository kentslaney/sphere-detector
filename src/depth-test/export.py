# TODO: float16 internals
import jax, json
import jax.numpy as jnp
import coremltools as ct

from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

from stablehlo_coreml import DEFAULT_HLO_PIPELINE

from .detect import Config, Raster
from .utils import local
from .uniq import convert

target = Config.resolution

@jax.jit
def jax_density(x):
    x = x.reshape(target)
    confidence, coordinates = Raster(None, x, resolution=target).opt().predict()
    confidence = jnp.astype(confidence, jnp.float16)
    coordinates /= jnp.tile(jnp.array(target), [1, 2])
    ll, hh = coordinates[:, 1::-1], coordinates[:, 3:1:-1]
    coordinates = jnp.hstack(((ll + hh) / 2, hh - ll + 1))
    coordinates = jnp.astype(coordinates, jnp.float16)
    return confidence.reshape((1, 1, -1)), coordinates.T.reshape((1, 4, -1))

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros((1, 1) + target, dtype=jnp.float16),)
jax_exported = export(jax_density)(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

dist = local / "dist"
dist.mkdir(parents=True, exist_ok=True)

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
logger_level = logger.level
logger.setLevel(logging.ERROR)

cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
    # compute_units=ct.ComputeUnit.CPU_ONLY,
    pass_pipeline=pipeline,
    inputs=[ct.ImageType(
        mil_arg0, shape=target, color_layout=ct.colorlayout.GRAYSCALE_FLOAT16,
        channel_first=True)],
)

logger.setLevel(logger_level)

spec = cml_model.get_spec()
# ct.utils.rename_feature(spec, '_arg0', 'depth')
ct.utils.rename_feature(spec, next(iter(cml_model.input_description)), 'depth')
it = iter(cml_model.output_description)
ct.utils.rename_feature(spec, next(it), 'confidence')
ct.utils.rename_feature(spec, next(it), 'coordinates')
model = ct.models.MLModel(spec, weights_dir=cml_model.weights_dir)

model.input_description["depth"] = (
    "Estimated, unitless, 518x392 depth map, as a grayscale image. "
    "Proportional to LiDaR raycast distance to projection plane "
    "as formatted by the KITTI dataset."
)
model.output_description["coordinates"] = (
    "Boxes × CENTER_SIZE_WIDTH_FIRST as proportions"
)
model.output_description["confidence"] = (
    "Boxes × 1: Class confidences over [0, 1]. "
    "Emphasis on time/scene-stable meanings. "
    "Reference points as of 0.2.0 (single digit number of test sessions) "
    "with a resolution of 392×518: "
    "0.25 is almost certain, 0.1 is likely, 0.01 is marginal."
)

model.author = "Kent Slaney"
model.license = "CC0"
model.version = "0.2.0"
model.user_defined_metadata["com.apple.coreml.model.preview.type"] = \
        "objectDetector" # https://github.com/apple/coremltools/issues/2265

model.save(str(dist / "spheres.mlpackage"))
