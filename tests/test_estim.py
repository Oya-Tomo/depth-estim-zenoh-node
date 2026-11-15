import importlib.util
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from transformers import (
    DepthAnythingConfig,
    DepthAnythingForDepthEstimation,
    DPTImageProcessor,
)
from transformers.utils import logging as transformers_logging

from config import DepthAnything3EstimatorConfig, TransformersEstimatorConfig
from estim import DepthAnything3Estimator, TransformersEstimator, load_estimator

transformers_logging.disable_progress_bar()

TRANSFORMERS_CONFIG = TransformersEstimatorConfig(
    backend="transformers", model_id="org/model", device="cpu"
)
DEPTH_ANYTHING_3_CONFIG = DepthAnything3EstimatorConfig(
    backend="depth_anything_3",
    model_id="org/da3",
    revision="v1",
    device="cpu",
    process_res=28,
)


class TransformersEstimatorTests(unittest.TestCase):
    def test_depth_has_the_size_of_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            DepthAnythingForDepthEstimation(
                DepthAnythingConfig(
                    backbone_config={
                        "model_type": "dinov2",
                        "hidden_size": 16,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "out_features": ["stage1", "stage2"],
                        "reshape_hidden_states": False,
                    },
                    reassemble_hidden_size=16,
                    neck_hidden_sizes=[8, 8],
                    reassemble_factors=[1, 0.5],
                    fusion_hidden_size=8,
                    head_hidden_size=8,
                )
            ).save_pretrained(directory)
            DPTImageProcessor(size={"height": 28, "width": 28}).save_pretrained(
                directory
            )
            estimator = TransformersEstimator(
                TRANSFORMERS_CONFIG.model_copy(update={"model_id": directory})
            )
        pixels = np.random.default_rng(0).integers(0, 256, (30, 40, 3), dtype=np.uint8)

        depth = estimator.estimate(Image.fromarray(pixels))

        self.assertEqual(depth.shape, (30, 40))
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.isfinite(depth).all())


class DepthAnything3EstimatorTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("depth_anything_3"),
        "requires the depth-anything-3 dependency group",
    )
    def test_depth_is_resized_and_rescaled_to_the_source_frame(self) -> None:
        image = Image.new("RGB", (80, 60))
        # The processed image is 28x21, so the focal length shrinks by 0.35.
        cases = {0: 0.7, 1: 2.0}
        for is_metric, expected in cases.items():
            model = MagicMock()
            model.to.return_value = model
            model.inference.return_value = SimpleNamespace(
                depth=np.full((1, 21, 28), 2.0, dtype=np.float32), is_metric=is_metric
            )
            with (
                self.subTest(is_metric=is_metric),
                patch(
                    "depth_anything_3.api.DepthAnything3.from_pretrained",
                    return_value=model,
                ) as from_pretrained,
            ):
                depth = DepthAnything3Estimator(DEPTH_ANYTHING_3_CONFIG).estimate(image)

                from_pretrained.assert_called_once_with("org/da3", revision="v1")
                model.to.assert_called_once_with(torch.device("cpu"))
                model.inference.assert_called_once_with([image], process_res=28)
                self.assertEqual(depth.shape, (60, 80))
                self.assertEqual(depth.dtype, np.float32)
                np.testing.assert_allclose(depth, expected, rtol=1e-6)

    def test_missing_dependency_group_is_reported(self) -> None:
        missing = ModuleNotFoundError(name="depth_anything_3")
        with (
            self.assertRaisesRegex(RuntimeError, "uv sync --group depth-anything-3"),
            patch("builtins.__import__", side_effect=missing),
        ):
            DepthAnything3Estimator(DEPTH_ANYTHING_3_CONFIG)

    def test_other_import_failures_are_not_masked(self) -> None:
        missing = ModuleNotFoundError(name="addict")
        with (
            self.assertRaises(ModuleNotFoundError),
            patch("builtins.__import__", side_effect=missing),
        ):
            DepthAnything3Estimator(DEPTH_ANYTHING_3_CONFIG)


class LoadEstimatorTests(unittest.TestCase):
    def test_backend_selects_the_estimator_class(self) -> None:
        with (
            patch("estim.TransformersEstimator") as transformers_estimator,
            patch("estim.DepthAnything3Estimator") as depth_anything_3_estimator,
        ):
            self.assertIs(
                load_estimator(TRANSFORMERS_CONFIG), transformers_estimator.return_value
            )
            self.assertIs(
                load_estimator(DEPTH_ANYTHING_3_CONFIG),
                depth_anything_3_estimator.return_value,
            )

        transformers_estimator.assert_called_once_with(TRANSFORMERS_CONFIG)
        depth_anything_3_estimator.assert_called_once_with(DEPTH_ANYTHING_3_CONFIG)


if __name__ == "__main__":
    unittest.main()
