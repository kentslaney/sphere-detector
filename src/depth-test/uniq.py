import numpy as np
import coremltools as ct
from stablehlo_coreml.converter import (
    StableHloConverter, register_optimizations
)
from coremltools.converters.mil.mil import Builder as mb

class MilNmsConfig:
    iou_threshold = 0.6

def convert(module):
    register_optimizations()
    return UniqPatch(opset_version=ct.target.iOS18).convert(module)

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
        n = confidence.shape[-1]
        coordinates, confidence, _ = mb.non_maximum_suppression(
            boxes=coordinates,
            scores=confidence,
            iou_threshold=mb.const(val=np.float16(MilNmsConfig.iou_threshold)),
            max_boxes=n
        )
        return confidence, coordinates
