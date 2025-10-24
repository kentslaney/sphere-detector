import pathlib, sys
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from static import Depth
sys.path.pop(0)

def jax_density(x):
    return Depth(x).density()

from jax.experimental import jax2tf
import tensorflow as tf

tf_density = tf.function(
        jax2tf.convert(jax_density), autograph=False,
        input_signature=[tf.TensorSpec([392, 518], tf.float16)])
conc_density = tf_density.get_concrete_function()

import coremltools as ct

mlmodel = ct.convert([conc_density], convert_to="mlprogram")
dist = local / "dist"
dist.mkdir(parents=True, exist_ok=True)
mlmodel.save(dist / "density.mlpackage")
