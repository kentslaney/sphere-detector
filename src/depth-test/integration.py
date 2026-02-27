import coremltools as ct
import numpy as np
from PIL import Image

from .examples import im4
from .utils import local, dist

cml_model = ct.models.MLModel(str(dist / "spheres.mlpackage"))
pil_depth = Image.fromarray(np.array(im4.depth.depth))
cml_out = cml_model.predict({"depth": pil_depth})
fmt_kw = {"sep": "\n", "end": "\n\n"}
print("CoreML", cml_out["confidence"], cml_out["coordinates"], **fmt_kw)
print("Jax", *im4.opt().predict(), **fmt_kw)
