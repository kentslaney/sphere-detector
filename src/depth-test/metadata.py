import coremltools as ct
from .utils import dist
from .nms import createNmsModelSpec

cml_model = ct.models.MLModel(str(dist / "partial.mlpackage"))

spec = cml_model.get_spec()
ct.utils.rename_feature(spec, next(iter(cml_model.input_description)), 'depth')
it = iter(cml_model.output_description)
ct.utils.rename_feature(spec, next(it), 'confidence')
ct.utils.rename_feature(spec, next(it), 'coordinates')

cml_input = spec.description.input[0].type.imageType
pipeline = ct.models.pipeline.Pipeline(
        input_features=[("depth", ct.models.datatypes.Array(
            1, cml_input.height, cml_input.width))],
        output_features=["confidence", "coordinates"])
pipeline.spec.specificationVersion = ct.target.iOS18

nmsSpec = createNmsModelSpec(spec)
pipeline.add_model(spec)
pipeline.add_model(nmsSpec)

pipeline.spec.description.input[0].ParseFromString(
    spec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(
    nmsSpec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(
    nmsSpec.description.output[1].SerializeToString())

model = ct.models.MLModel(pipeline.spec, weights_dir=cml_model.weights_dir)

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
    "with data resolution of 392×518 from Depth Anything V2 small unquantized: "
    "0.25 is almost certain, 0.1 is likely, 0.01 is marginal."
)

model.author = "Kent Slaney"
model.license = "CC0"
model.version = "0.2.1"
model.short_description = (
    "Looks for 3d curves, "
    "fits a circle to surrounding depth drop-offs, then "
    "evaluates spherical depth fit."
)
model.user_defined_metadata["com.apple.coreml.model.preview.type"] = \
        "objectDetector" # https://github.com/apple/coremltools/issues/2265

model.save(str(dist / "spheres.mlpackage"))
