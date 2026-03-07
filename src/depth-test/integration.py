import coremltools as ct
import numpy as np
from PIL import Image

from .examples import cache, Example
from .utils import dist, examples
from .cml import jax_center_size_width_first, config, config_kw

im4_cml = Example.file(
        examples / "IMG_0004.HEIC", cache / "da2_4_cml.npy", "im4", **config_kw)

if __name__ == "__main__":
    cml_model = ct.models.MLModel(str(dist / "e2e.mlpackage"))
    cml_out = cml_model.predict({"image": Image.open(
        examples / "IMG_0004.HEIC").crop((0, 0, 294, 518))})
    # cml_out = cml_model.predict({
    #         "depth": config.input_cast(im4_cml.depth.depth)})
    fmt_kw = {"sep": "\n", "end": "\n\n"}
    print("CoreML", cml_out["confidence"], cml_out["coordinates"], **fmt_kw)
    print("Jax", *jax_center_size_width_first(im4_cml.depth.depth), **fmt_kw)
