import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import zenoh
from pydantic import ValidationError

from config import (
    DepthAnything3EstimatorConfig,
    ImageConfig,
    NodeConfig,
    TransformersEstimatorConfig,
    load_node_config,
)

VALID_CONFIG: dict[str, Any] = {
    "zenoh_key_prefix": "depth",
    "estimator": {"backend": "transformers", "model_id": "org/model", "device": "cpu"},
    "streams": {"front": "camera/front", "view_main": "spherical_camera/view/main"},
    "image": {"colormap": "turbo", "jpeg_quality": 90},
    "publisher": {"congestion_control": "drop", "reliability": "best_effort"},
}


def config_with(path: tuple[str, ...], value: object) -> dict[str, Any]:
    document = copy.deepcopy(VALID_CONFIG)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return document


def config_without(*path: str) -> dict[str, Any]:
    document = copy.deepcopy(VALID_CONFIG)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    return document


class ExampleConfigTests(unittest.TestCase):
    def test_example_node_config_is_valid(self) -> None:
        config = load_node_config(Path("config/node-config.example.json5"))

        self.assertIsInstance(config.estimator, TransformersEstimatorConfig)
        self.assertEqual(
            dict(config.streams),
            {"front": "camera/front", "view_main": "spherical_camera/view/main"},
        )
        self.assertEqual(
            config.image,
            ImageConfig(colormap="turbo", value_range=None, jpeg_quality=90),
        )

    def test_example_zenoh_configs_are_valid(self) -> None:
        modes = [
            json.loads(zenoh.Config.from_file(path).get_json("mode"))
            for path in (
                "config/zenoh-config.example.json5",
                "examples/viewer-zenoh-config.json5",
            )
        ]

        self.assertEqual(modes, ["peer", "client"])


class NodeConfigTests(unittest.TestCase):
    def test_transformers_estimator_defaults(self) -> None:
        estimator = NodeConfig.model_validate(VALID_CONFIG).estimator

        assert isinstance(estimator, TransformersEstimatorConfig)
        self.assertEqual((estimator.revision, estimator.dtype), ("main", "float32"))

    def test_depth_anything_3_estimator_defaults(self) -> None:
        estimator = NodeConfig.model_validate(
            config_with(
                ("estimator",),
                {
                    "backend": "depth_anything_3",
                    "model_id": "org/da3",
                    "device": "cuda",
                },
            )
        ).estimator

        assert isinstance(estimator, DepthAnything3EstimatorConfig)
        self.assertEqual((estimator.revision, estimator.process_res), ("main", 504))

    def test_value_range_defaults_to_the_range_of_each_frame(self) -> None:
        self.assertIsNone(NodeConfig.model_validate(VALID_CONFIG).image.value_range)

    def test_value_range_accepts_integers_and_floats(self) -> None:
        for value_range in ([0, 5], [0.0, 5.0], [-1.5, 2]):
            with self.subTest(value_range=value_range):
                image = NodeConfig.model_validate(
                    config_with(("image", "value_range"), value_range)
                ).image

                self.assertEqual(image.value_range, tuple(value_range))

    def test_colormap_accepts_reversed_and_mixed_case_names(self) -> None:
        for colormap in ("viridis", "magma_r", "Spectral"):
            with self.subTest(colormap=colormap):
                image = NodeConfig.model_validate(
                    config_with(("image", "colormap"), colormap)
                ).image

                self.assertEqual(image.colormap, colormap)

    def test_jpeg_quality_accepts_its_bounds(self) -> None:
        for jpeg_quality in (1, 100):
            with self.subTest(jpeg_quality=jpeg_quality):
                NodeConfig.model_validate(
                    config_with(("image", "jpeg_quality"), jpeg_quality)
                )

    def test_invalid_configs_are_rejected(self) -> None:
        invalid_configs = {
            "unknown field": config_with(("unknown",), 1),
            "unknown estimator field": config_with(("estimator", "process_res"), 504),
            "missing backend": config_without("estimator", "backend"),
            "unknown backend": config_with(("estimator", "backend"), "onnx"),
            "blank model id": config_with(("estimator", "model_id"), " "),
            "missing device": config_without("estimator", "device"),
            "unknown dtype": config_with(("estimator", "dtype"), "float64"),
            "wildcard prefix": config_with(("zenoh_key_prefix",), "depth/*"),
            "blank prefix": config_with(("zenoh_key_prefix",), " "),
            "multi-segment stream name": config_with(
                ("streams",), {"front/left": "camera/front"}
            ),
            "wildcard stream name": config_with(("streams",), {"*": "camera/front"}),
            "blank stream name": config_with(("streams",), {" ": "camera/front"}),
            "wildcard source key": config_with(("streams",), {"front": "camera/*"}),
            "blank source key": config_with(("streams",), {"front": " "}),
            "duplicate source keys": config_with(
                ("streams",), {"front": "camera/front", "again": "camera/front"}
            ),
            "image key equal to a source key": config_with(
                ("streams",), {"front": "camera/front", "loop": "depth/front"}
            ),
            "raw key equal to a source key": config_with(
                ("streams",), {"front": "camera/front", "loop": "depth/front/raw"}
            ),
            "empty streams": config_with(("streams",), {}),
            "missing image": config_without("image"),
            "unknown image field": config_with(("image", "gamma"), 1.0),
            "unknown colormap": config_with(("image", "colormap"), "rainbow_road"),
            "missing colormap": config_without("image", "colormap"),
            "string value range": config_with(("image", "value_range"), "0-5"),
            "string range bound": config_with(("image", "value_range"), ["0", "5"]),
            "boolean range bound": config_with(("image", "value_range"), [False, True]),
            "NaN range bound": config_with(
                ("image", "value_range"), [0.0, float("nan")]
            ),
            "infinite range bound": config_with(
                ("image", "value_range"), [0.0, float("inf")]
            ),
            "one range bound": config_with(("image", "value_range"), [5.0]),
            "three range bounds": config_with(("image", "value_range"), [0, 1, 2]),
            "empty range": config_with(("image", "value_range"), [5.0, 5.0]),
            "reversed range": config_with(("image", "value_range"), [5.0, 0.0]),
            "jpeg quality below 1": config_with(("image", "jpeg_quality"), 0),
            "jpeg quality above 100": config_with(("image", "jpeg_quality"), 101),
            "fractional jpeg quality": config_with(("image", "jpeg_quality"), 90.5),
            "missing jpeg quality": config_without("image", "jpeg_quality"),
            "unknown reliability": config_with(("publisher", "reliability"), "maybe"),
            "unknown congestion control": config_with(
                ("publisher", "congestion_control"), "queue"
            ),
        }
        for case, document in invalid_configs.items():
            with self.subTest(case=case), self.assertRaises(ValidationError):
                NodeConfig.model_validate(document)

    def test_unknown_colormap_error_names_matplotlib(self) -> None:
        with self.assertRaisesRegex(ValidationError, "matplotlib colormap name"):
            NodeConfig.model_validate(
                config_with(("image", "colormap"), "rainbow_road")
            )

    def test_loaded_config_is_immutable(self) -> None:
        config = NodeConfig.model_validate(VALID_CONFIG)
        streams: Any = config.streams

        with self.assertRaises(ValidationError):
            config.zenoh_key_prefix = "other"
        with self.assertRaises(ValidationError):
            config.estimator.device = "cuda"
        with self.assertRaises(ValidationError):
            config.image.colormap = "viridis"
        with self.assertRaises(TypeError):
            streams["rear"] = "camera/rear"


class LoadNodeConfigTests(unittest.TestCase):
    def load_text(self, text: str) -> NodeConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "node.json5")
            path.write_text(text, encoding="utf-8")
            return load_node_config(path)

    def test_duplicate_json5_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid JSON5"):
            self.load_text('{zenoh_key_prefix: "a", zenoh_key_prefix: "b"}')

    def test_json5_special_numbers_are_rejected_as_range_bounds(self) -> None:
        document = json.dumps(VALID_CONFIG).replace(
            '"jpeg_quality"', '"value_range": [0, Infinity], "jpeg_quality"'
        )

        with self.assertRaisesRegex(ValueError, "invalid node configuration"):
            self.load_text(document)

    def test_validation_errors_name_the_file(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"node\.json5: invalid node configuration"
        ):
            self.load_text("{}")


if __name__ == "__main__":
    unittest.main()
