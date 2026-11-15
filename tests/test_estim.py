import importlib.util
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import (
    DepthAnythingConfig,
    DepthAnythingForDepthEstimation,
    DPTImageProcessor,
)
from transformers.utils import logging as transformers_logging

from config import DepthAnything3EstimatorConfig, TransformersEstimatorConfig
from estim import (
    DEPTH_ANYTHING_3_PATCH_SIZE,
    DepthAnything3Estimator,
    TransformersEstimator,
    load_estimator,
)

transformers_logging.disable_progress_bar()

TRANSFORMERS_CONFIG = TransformersEstimatorConfig(
    backend="transformers", model_id="org/model", device="cpu"
)
DEPTH_ANYTHING_3_CONFIG = DepthAnything3EstimatorConfig(
    backend="depth_anything_3",
    model_id="org/da3",
    revision="v1",
    device="cpu",
    process_res=56,
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
        image = torch.randint(
            0,
            256,
            (3, 30, 40),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(0),
        )

        depth = estimator.estimate(image)

        self.assertEqual(estimator.device, torch.device("cpu"))
        self.assertEqual(depth.shape, (30, 40))
        self.assertEqual(depth.dtype, torch.float32)
        self.assertEqual(depth.device, torch.device("cpu"))
        self.assertTrue(depth.isfinite().all())


@unittest.skipUnless(
    importlib.util.find_spec("depth_anything_3"),
    "requires the depth-anything-3 dependency group",
)
class DepthAnything3EstimatorTests(unittest.TestCase):
    def estimate(
        self, image: torch.Tensor, **output: int
    ) -> tuple[torch.Tensor, MagicMock]:
        """Estimate with a model that returns a depth of 2 at the processed size."""
        from addict import Dict

        model = MagicMock()
        model.to.return_value = model
        model.side_effect = lambda batch, **_: Dict(
            depth=torch.full((1, 1, *batch.shape[-2:]), 2.0), **output
        )
        with patch(
            "depth_anything_3.api.DepthAnything3.from_pretrained", return_value=model
        ) as from_pretrained:
            estimator = DepthAnything3Estimator(DEPTH_ANYTHING_3_CONFIG)
        from_pretrained.assert_called_once_with("org/da3", revision="v1")
        model.to.assert_called_once_with(torch.device("cpu"))
        self.assertEqual(estimator.device, torch.device("cpu"))
        return estimator.estimate(image), model

    def test_model_gets_one_normalized_view_at_the_processed_size(self) -> None:
        sizes = {
            (60, 80): (42, 56),
            (80, 60): (56, 42),
            # The shorter side is never less than one patch.
            (10, 400): (14, 56),
        }
        for (height, width), processed_size in sizes.items():
            with self.subTest(height=height, width=width):
                image = torch.full((3, height, width), 255, dtype=torch.uint8)

                depth, model = self.estimate(image)

                (batch,) = model.call_args.args
                self.assertEqual(model.call_args.kwargs, {"export_feat_layers": []})
                self.assertEqual(batch.shape, (1, 1, 3, *processed_size))
                self.assertEqual(max(processed_size), 56)
                self.assertEqual(
                    [side % DEPTH_ANYTHING_3_PATCH_SIZE for side in processed_size],
                    [0, 0],
                )
                # White under the ImageNet normalization.
                torch.testing.assert_close(
                    batch.mean(dim=(0, 1, 3, 4)),
                    torch.tensor([2.2489, 2.4286, 2.6400]),
                    atol=1e-4,
                    rtol=0,
                )
                self.assertEqual(depth.shape, (height, width))
                self.assertEqual(depth.dtype, torch.float32)

    def test_depth_is_rescaled_to_the_source_frame_unless_metric(self) -> None:
        image = torch.zeros((3, 60, 80), dtype=torch.uint8)
        # The processed image is 56x42, so the focal length shrinks by 0.7.
        cases: list[tuple[dict[str, int], float]] = [
            ({}, 1.4),
            ({"is_metric": 0}, 1.4),
            ({"is_metric": 1}, 2.0),
        ]
        for output, expected in cases:
            with self.subTest(output=output):
                depth, _ = self.estimate(image, **output)

                torch.testing.assert_close(depth, torch.full((60, 80), expected))


class DepthAnything3DependencyTests(unittest.TestCase):
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
