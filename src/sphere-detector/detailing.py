import coremltools as ct
from .utils import dist

cml_model = ct.models.MLModel(str(dist / "partial.mlpackage"))

spec = cml_model.get_spec()
ct.utils.rename_feature(spec, next(iter(cml_model.input_description)), 'depth')
it = iter(cml_model.output_description)
ct.utils.rename_feature(spec, next(it), 'confidence')
ct.utils.rename_feature(spec, next(it), 'coordinates')
model = ct.models.MLModel(spec, weights_dir=cml_model.weights_dir)

model.input_description["depth"] = (
    "Estimated, unitless depth map, as a grayscale image. "
    "Proportional to LiDaR raycast distance to projection plane "
    "as formatted by the KITTI dataset."
)
model.output_description["coordinates"] = (
    "1 \xd7 CENTER_SIZE_WIDTH_FIRST \xd7 Boxes as proportions"
)
model.output_description["confidence"] = (
    "1 \xd7 1 \xd7 Boxes: Class confidences over [0, 1]. "
    "Emphasis on time/scene-stable meanings. "
    "Reference points as of 0.2.0 (single digit number of test sessions) "
    "with data resolution of 392\xd7518 "
    "from Depth Anything V2 small unquantized: "
    "0.25 is almost certain, 0.1 is likely, 0.01 is marginal."
)

model.author = "Kent Slaney"
model.license = "CC0"
model.version = "0.3.1"
model.short_description = (
    "Looks for 3d curves, "
    "fits a circle to surrounding depth drop-offs, then "
    "evaluates spherical depth fit."
)

model.save(str(dist / "spheres.mlpackage"))
