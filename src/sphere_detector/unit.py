import unittest
import pickle
import numpy as np
import jax.numpy as jnp
import coremltools as ct
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

from .detect import Raster
from .cml import config, config_kw, convert, jax_center_size_width_first
from .integration import im4_cml
from .utils import assets


class TestNmsPipeline(unittest.TestCase):

    def setUp(self):
        self.golden_path = assets / "nms_golden.pkl"
        if not self.golden_path.exists():
            self.skipTest(f"Golden dataset {self.golden_path} not found")
        with open(self.golden_path, "rb") as f:
            self.golden = pickle.load(f)
        self.depth = self.golden["depth"]
        self.raster = Raster(None, self.depth, **config_kw)
        self.seives = self.raster.seives

    def test_nms_golden_levels(self):
        """Verify NMS outputs bit-for-bit against golden reference."""
        for level in range(len(self.seives.stack) - 1, 0, -1):
            with self.subTest(level=level):
                nms_out = np.array(self.seives.nms(level))
                expected = self.golden["nms_outputs"][level]
                np.testing.assert_array_equal(nms_out, expected)

    def test_nominate_golden(self):
        """Verify candidate nomination top-K against golden reference."""
        vals, idxs = self.seives.nominate(config.candidates)
        np.testing.assert_array_equal(
            np.array(idxs), self.golden["nominate_indices"])
        np.testing.assert_allclose(
            np.array(vals), self.golden["nominate_values"], rtol=1e-5, atol=1e-6)

    def test_export_pipeline_conversion(self):
        """Verify JAX export and CoreML MIL conversion and prediction."""
        context = jax_mlir.make_ir_context()
        input_shapes = (jnp.zeros(config.input_shape, dtype=config.input_dtype),)
        jax_exported = export(jax_center_size_width_first)(*input_shapes)
        hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

        mil_program = convert(hlo_module, patch_tags=True, patch_output=True)
        mil_args = mil_program.functions[
            mil_program.default_function_name].inputs.keys()
        mil_arg0 = next(iter(mil_args))

        pipeline = DEFAULT_HLO_PIPELINE
        pipeline.set_options(
            "common::const_elimination", {"skip_const_by_size": "1e2"})

        cml_model = ct.convert(
            mil_program,
            source="milinternal",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT32,
            pass_pipeline=pipeline,
            inputs=[ct.TensorType(
                mil_arg0, shape=config.input_shape, dtype=config.input_cml_dtype)],
        )

        input_names = [f.name for f in cml_model.get_spec().description.input]
        depth_in = np.array(im4_cml.depth.depth).reshape(config.input_shape)
        cml_out = cml_model.predict({input_names[0]: depth_in})
        jax_out = jax_center_size_width_first(depth_in)

        self.assertIsNotNone(cml_out)
        self.assertEqual(len(cml_out), 2)


if __name__ == "__main__":
    unittest.main()

