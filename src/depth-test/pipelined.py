import coremltools as ct
from .utils import dist
from .cml import config

depth_model = ct.models.MLModel(str(dist / f"{config.da2_name}.mlpackage"))
spheres_model = ct.models.MLModel(str(dist / "spheres.mlpackage"))

model = ct.utils.make_pipeline(depth_model, spheres_model)
# TODO: Move NMS from MIL to CoreML pipeline
# model.user_defined_metadata["com.apple.coreml.model.preview.type"] = \
#         "objectDetector" # https://github.com/apple/coremltools/issues/2265
model.save(dist / "pipelined.mlpackage")
