import coremltools as ct
import numpy as np
from PIL import Image

from .examples import im4
from .utils import local, dist
from .export import jax_center_size_width_first

cml_model = ct.models.MLModel(str(dist / "spheres.mlpackage"))
cml_out = cml_model.predict({"depth": np.array(im4.depth.depth)})
fmt_kw = {"sep": "\n", "end": "\n\n"}
print("CoreML", cml_out["confidence"], cml_out["coordinates"], **fmt_kw)
print("Jax", *jax_center_size_width_first(im4.depth.depth), **fmt_kw)
