import jax
import jax.numpy as jnp
import numpy as np
import coremltools as ct
from dataclasses import dataclass
from functools import partial

from stablehlo_coreml.converter import (
    StableHloConverter, register_optimizations
)
from stablehlo_coreml.utils import get_numpy_type
from coremltools.converters.mil.mil import Builder as mb
from jax._src.lib.mlir.dialects import hlo

from .detect import Config, Raster
from .utils import patch_label, patch_sep, Image
from .depth import Da2

class DepthInputMode:
    F16Image = partial(
            ct.ImageType, color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)
    # F16Tensor = partial(ct.TensorType, dtype=ct.converters.mil.mil.types.fp16)
    F32Tensor = partial(ct.TensorType, dtype=ct.converters.mil.mil.types.fp32)

@dataclass
class CmlConfig(Config):
    resolution: any = (518, 294)
    iou_threshold: any = 0.6
    opset_version: any = ct.target.iOS18
    da2_precision: any = ct.precision.FLOAT16
    spheres_input: any = DepthInputMode.F16Image

    @property
    def input_shape(self):
        nc = ((1, 1) if self.spheres_input is DepthInputMode.F16Image else ())
        return nc + self.resolution

    @property
    def input_dtype(self):
        return jnp.float32 if self.spheres_input is DepthInputMode.F32Tensor \
                else jnp.float16

    @property
    def input_cml_type(self):
        return partial(self.spheres_input, shape=self.resolution)

    @property
    def input_cml_dtype(self):
        return ct.converters.mil.mil.types.fp32 \
                if self.spheres_input is DepthInputMode.F32Tensor else \
                ct.converters.mil.mil.types.fp16

    def input_cast(self, x):
        x = np.array(x)
        if self.spheres_input is DepthInputMode.F16Image:
            x = Image.fromarray(x)
        return x

    @property
    def da2_name(self):
        model_precision = config.da2_precision.name.replace("FLOAT", "F")
        model_size = Da2.size_mapping[self.depth_checkpoint]
        return f"DepthAnythingV2{model_size}{model_precision}"

config = CmlConfig()
config_kw = {
        k: getattr(config, k) for k in config.__dataclass_fields__
        if k in Config.__dataclass_fields__}

@jax.jit
def jax_center_size_width_first(x):
    if len(x.shape) == 4:
        assert x.shape[:2] == (1, 1)
        x = x.reshape(x.shape[2:])
    confidence, coordinates = Raster(None, x, **config_kw).opt().predict()
    ll, hh = coordinates[:, 1::-1], coordinates[:, 3:1:-1]
    coordinates = jnp.hstack(((ll + hh) / 2, hh - ll + 1))
    coordinates /= jnp.tile(jnp.array(config.resolution[::-1]), [1, 2])
    return confidence.reshape((1, 1, -1)), coordinates.T.reshape((1, 4, -1))

def convert(module, patch_tags=True, patch_output=False):
    register_optimizations()
    converter = UniqPatch if patch_output else TagPatcher
    if not patch_tags:
        converter.tag_map = None
    return converter(opset_version=config.opset_version).convert(module)

DefaultConverter = StableHloConverter

class RegisteredConverter(DefaultConverter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stablehlo_ops_registry = DefaultConverter._stablehlo_ops_registry

    def default_dispatch(self, context, op):
        return DefaultConverter._dispatch_op(self, context, op)

class LabelRegistry(RegisteredConverter):
    def mps_gather_shape(self, context, op):
        if op.name != "stablehlo.gather":
            return self.default_dispatch(context, op)

        dim_numbers = hlo.GatherDimensionNumbers(op.dimension_numbers)
        expected_dim_numbers = {
            "offset_dims": (2,),
            "operand_batching_dims": (0, 1),
            "start_indices_batching_dims": (0, 1),
            "start_index_map": (2,),
            "index_vector_dim": 2
        }
        assert all([
            type(v)(getattr(dim_numbers, k)) == v
            for k, v in expected_dim_numbers.items()])
        assert tuple(op.operand.type.shape) == (
                config.candidates, config.rays, config.distance)
        assert tuple(op.start_indices.type.shape) == (
                config.candidates, config.rays, 1)
        assert tuple(op.slice_sizes) == (1, 1, 1)

        start_indices = context[op.start_indices.get_name()]
        operand = context[op.operand.get_name()]

        res = mb.gather_along_axis(x=operand, indices=start_indices, axis=2)

        context.add_result(op.result, res)
        return

class TagPatcher(RegisteredConverter):
    tag_map = LabelRegistry()

    def _dispatch_op(self, context, op):
        stack = str(op.location)
        if patch_label in stack:
            index = stack.index(patch_label)
            prefix, postfix = stack[:index], stack[index + len(patch_label):]
            assert postfix.startswith(patch_sep) and prefix.endswith('""'[0])
            tag = postfix[len(patch_sep):postfix.index('""'[0])]
            # one callsite for the decorator and one for the decorated
            depth = prefix.count("callsite()"[:-1]) - 2  # >= 0
            if depth == 0 and hasattr(self.tag_map, tag):
                return getattr(self.tag_map, tag)(context, op)
        return self.default_dispatch(context, op)

class MilInjector(TagPatcher):
    def process_block(self, context, block):
        self.process_block = super().process_block
        return list(self.patch(*super().process_block(context, block)))

    def patch(self, *outputs):
        return outputs

class UniqPatch(MilInjector):
    def patch(self, confidence, coordinates):
        iou_threshold = get_numpy_type(coordinates)(config.iou_threshold)
        coordinates, confidence, _ = mb.non_maximum_suppression(
            boxes=coordinates,
            scores=confidence,
            iou_threshold=mb.const(val=iou_threshold),
            max_boxes=confidence.shape[-1]
        )
        return confidence, coordinates
