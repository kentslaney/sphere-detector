import numpy as np
import coremltools as ct
from dataclasses import dataclass

from stablehlo_coreml.converter import (
    StableHloConverter, register_optimizations
)
from stablehlo_coreml.utils import get_numpy_type
from coremltools.converters.mil.mil import Builder as mb

from .detect import Config

@dataclass
class CmlConfig(Config):
    iou_threshold: any = 0.6
    resolution: any = (518, 294)

def convert(module, patch=False, opset_version=ct.target.iOS18):
    register_optimizations()
    converter = UniqPatch if patch else StableHloConverter
    return converter(opset_version=opset_version).convert(module)

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
