# https://github.com/huggingface/coreml-examples/blob/main/tutorials/depth-anything-coreml-guide.ipynb

import torch, torchvision
import coremltools as ct
import numpy as np

from importlib.metadata import distribution
from functools import cached_property

from transformers import AutoModelForDepthEstimation
from transformers import AutoImageProcessor

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op

from .utils import Image, examples, dist
from .cml import config

import logging
logger = logging.getLogger(__name__)

height, width = target = config.resolution

class Da2:
    size_mapping = { 'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large' }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, encoder):
        self.encoder = encoder

    @property
    def size(self):
        return self.size_mapping[self.encoder]

    @property
    def model_repo(self):
        return f'depth-anything/Depth-Anything-V2-{self.size}-hf'

    @cached_property
    def model(self):
        model = AutoModelForDepthEstimation.from_pretrained(self.model_repo)
        return model.eval()

model = Da2(config.depth_checkpoint)

def load_image(path):
    image = Image.open(path)

    target_aspect = width / height
    current_aspect = image.width / image.height

    if current_aspect > target_aspect:
        new_width = int(target_aspect * image.height)
        offset = (image.width - new_width) / 2
        image = image.crop((offset, 0, offset + new_width, image.height))
    else:
        new_height = int(image.width / target_aspect)
        offset = (image.height - new_height) / 2
        image = image.crop((0, offset, image.width, offset + new_height))

    return image.resize((width, height), resample=Image.BICUBIC)

scaled_image = load_image(examples / "IMG_0004.HEIC")
example_inputs = torchvision.transforms.functional.pil_to_tensor(scaled_image)

# These will be our Core ML inputs (unscaled and unnormalized)
example_inputs_coreml = example_inputs.unsqueeze(0).float()

# We further normalize to compare with the PyTorch pre-processing pipeline
example_inputs = example_inputs / 255.0
example_inputs = torchvision.transforms.functional.normalize(
        example_inputs, mean=model.mean, std=model.std)
example_inputs = example_inputs.unsqueeze(0)

with torch.inference_mode():
    outputs = model.model(example_inputs)
    baseline = outputs.predicted_depth

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.mean = [255 * x for x in model.mean]
        self.std = [255 * x for x in model.std]
        self.flattened = config.da2_precision == ct.precision.FLOAT32

    @torch.no_grad()
    def forward(self, pixel_values):
        """pixel_values are floats in the range `[0, 255]`"""
        # Apply ImageNet normalization
        pixel_values = torchvision.transforms.functional.normalize(
                pixel_values, mean=self.mean, std=self.std)

        outputs = self.model(pixel_values, return_dict=False)
        # Normalize output to `[0, 1]` and add batch size dimension
        normalized = outputs[0] / outputs[0].max()
        if self.flattened:
            return normalized.squeeze(0)
        else:
            return normalized.unsqueeze(0)

to_trace = Wrapper(model)
traced_model = torch.jit.trace(to_trace, example_inputs_coreml)
traced_model.eval()
with torch.no_grad():
    out = traced_model(example_inputs_coreml)

logger.info("preprocessing error", (out - baseline/baseline.max()).abs().max())

input_types = [ct.ImageType(name="image", shape=example_inputs_coreml.shape)]
patched_transformers = distribution("transformers").read_text("direct_url.json")
ane_patch = patched_transformers and patched_transformers.endswith(
        "github.com/pcuenca/transformers@dino-ane-patch")

if patched_transformers and not ane_patch:
    logger.warn("exporting for ANE with unknown transformers distribution")

if to_trace.flattened:
    if patched_transformers and ane_patch:
        logger.error("exporting for GPU with transformers patched for ANE")
    compute_units = ct.ComputeUnit.CPU_AND_GPU
    output_types = [ct.TensorType(
        "depth", dtype=ct.converters.mil.mil.types.fp32)]
else:
    if not patched_transformers:
        logger.error(
                "exporting for ANE with unpatched transformers distribution")
    compute_units = ct.ComputeUnit.ALL
    output_types = [ct.ImageType(
        "depth", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)]

@register_torch_op
def upsample_bicubic2d(context, node):
    a = context[node.inputs[0]]
    align_corners = context[node.inputs[2]].val
    scale = context[node.inputs[3]]
    if scale is None:
        output_size = context[node.inputs[1]].val
        input_height = a.shape[-2]
        input_width = a.shape[-1]
        scale_h = output_size[0] / input_height
        scale_w = output_size[1] / input_width
    else:
        scale_h = scale.val[0]
        scale_w = scale.val[1]

    context.add(mb.upsample_bilinear(
        x=a,
        scale_factor_height=scale_h,
        scale_factor_width=scale_w,
        align_corners=align_corners,
        name=node.name
    ))

coreml_model = ct.convert(
    traced_model,
    minimum_deployment_target = config.opset_version,
    inputs = input_types,
    outputs = output_types,
    compute_units = compute_units,
    compute_precision = config.da2_precision,
)

coreml_inputs = {"image": scaled_image}
coreml_outputs = coreml_model.predict(coreml_inputs)

output_image = coreml_outputs["depth"]
output_array = np.array(output_image)
assert output_array.shape == target
baseline_np = (baseline / baseline.max()).numpy()[0]
logger.info("conversion error", np.abs(output_array - baseline_np).max())

model_precision = config.da2_precision.name.replace("FLOAT", "F")
model_name = f"DepthAnythingV2{model.size}{model_precision}"
coreml_model.name = model_name
coreml_model.version = "2.0"
coreml_model.short_description = (
    "Depth Anything V2 is a state-of-the-art deep learning model for "
    "depth estimation."
)
coreml_model.author = "Original Paper: Lihe Yang et al. (Depth Anything V2)"
coreml_model.license = "Apache 2"
coreml_model.input_description["image"] = \
        "Input image whose depth will be estimated."
coreml_model.output_description["depth"] = \
        "Estimated depth map, as a grayscale output image."

rdns = "com.apple.developer.machine-learning.models"
coreml_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = \
        "depthEstimation"
coreml_model.user_defined_metadata[f"{rdns}.category"] = "image"
coreml_model.user_defined_metadata[f"{rdns}.name"] = f"{model_name}.mlpackage"
coreml_model.user_defined_metadata[f"{rdns}.version"] = "2.0"
coreml_model.user_defined_metadata[f"{rdns}.release-date"] = "2024-06"

coreml_model.save(dist / f"{model_name}.mlpackage")
