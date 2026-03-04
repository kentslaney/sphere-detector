import jax
import jax.numpy as jnp
import numpy as np
import coremltools as ct
from dataclasses import dataclass

from stablehlo_coreml.converter import (
    StableHloConverter, register_optimizations
)
from stablehlo_coreml.utils import get_numpy_type
from coremltools.converters.mil.mil import Builder as mb

from .detect import Config, Raster

@dataclass
class CmlConfig(Config):
    resolution: any = (518, 294)
    iou_threshold: any = 0.6
    opset_version: any = ct.target.iOS18
    da2_precision: any = ct.precision.FLOAT16

config_kw = {
        k: getattr(CmlConfig, k) for k in CmlConfig.__dataclass_fields__
        if k in Config.__dataclass_fields__}

@jax.jit
def jax_center_size_width_first(x):
    confidence, coordinates = Raster(None, x, **config_kw).opt().predict()
    ll, hh = coordinates[:, 1::-1], coordinates[:, 3:1:-1]
    coordinates = jnp.hstack(((ll + hh) / 2, hh - ll + 1))
    coordinates /= jnp.tile(jnp.array(CmlConfig.resolution[::-1]), [1, 2])
    return confidence.reshape((1, 1, -1)), coordinates.T.reshape((1, 4, -1))

def convert(module, patch=False):
    register_optimizations()
    converter = UniqPatch if patch else StableHloConverter
    return converter(opset_version=CmlConfig.opset_version).convert(module)

class MilInjector(StableHloConverter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stablehlo_ops_registry = \
                __class__.__bases__[0]._stablehlo_ops_registry

    def process_block(self, context, block):
        self.process_block = super().process_block
        return list(self.patch(*super().process_block(context, block)))

    def patch(self, *outputs):
        return outputs

class UniqPatch(MilInjector):
    def patch(self, confidence, coordinates):
        iou_threshold = get_numpy_type(coordinates)(CmlConfig.iou_threshold)
        coordinates, confidence, _ = mb.non_maximum_suppression(
            boxes=coordinates,
            scores=confidence,
            iou_threshold=mb.const(val=iou_threshold),
            max_boxes=confidence.shape[-1]
        )
        return confidence, coordinates
