import numpy as np
import coremltools as ct
from stablehlo_coreml.converter import (
    StableHloConverter, register_optimizations
)
from stablehlo_coreml.utils import get_numpy_type
from coremltools.converters.mil.mil import Builder as mb

def convert(module, patch=False):
    register_optimizations()
    converter = MilInjector if patch else StableHloConverter
    return MilInjector(opset_version=ct.target.iOS18).convert(module)

class MilInjector(StableHloConverter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stablehlo_ops_registry = \
                __class__.__bases__[0]._stablehlo_ops_registry

    def process_block(self, context, block):
        self.process_block = super().process_block
        return list(self.patch(*super().process_block(context, block)))

    def patch(self, *outputs):
        # add any MIL specific ops here
        return outputs
