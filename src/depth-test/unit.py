import jax, json
import jax.numpy as jnp
import coremltools as ct
import numpy as np

from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

from stablehlo_coreml import DEFAULT_HLO_PIPELINE

from .detect import Config, Raster
from .utils import dist
from .mil import convert
from .examples import im4

target = Config.resolution

@jax.jit
def jax_density(x):
    return Raster(None, x, resolution=target).depth.binned().counts

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros(target, dtype=jnp.float32),)
jax_exported = export(jax_density)(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

mil_program = convert(hlo_module)

mil_args = mil_program.functions[
        mil_program.default_function_name].inputs.keys()
mil_arg0 = next(iter(mil_args))

pipeline = DEFAULT_HLO_PIPELINE
pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e2"})
# pipeline.remove_passes(['common::add_int16_cast'])

import logging
from coremltools import _logger as logger
# logger_level = logger.level
# logger.setLevel(logging.ERROR)
logging.getLogger("coremltools").disabled = True

cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
    # compute_units=ct.ComputeUnit.CPU_ONLY,
    compute_precision=ct.precision.FLOAT32,
    pass_pipeline=pipeline,
    inputs=[ct.TensorType(mil_arg0, shape=target, dtype=np.float32)],
)

# logger.setLevel(logger_level)
logging.getLogger("coremltools").disabled = False

cml_out = cml_model.predict({"_arg0": np.array(im4.depth.depth)})
jax_out = jax_density(im4.depth.depth)
fmt_kw = {"sep": "\n", "end": "\n\n"}
print("CoreML", *cml_out.values(), **fmt_kw)
print("Jax", jax_out, **fmt_kw)

cml_im = next(iter(cml_out.values()))

import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(cml_im)
ax1.imshow(jax_out)
plt.show()
