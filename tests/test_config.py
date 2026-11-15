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
    NodeConfig,
    TransformersEstimatorConfig,
    load_node_config,
)

VALID_CONFIG: dict[str, Any] = {
    "zenoh_key_prefix": "depth",
    "estimator": {"backend": "transformers", "model_id": "org/model", "device": "cpu"},
    "streams": {"front": "camera/front", "view_main": "spherical_camera/view/main"},
    "publisher": {"congestion_control": "drop", "reliability": "best_effort"},
}


def config_with(path: tuple[str, ...], value: object) -> dict[str, Any]:
    document = copy.deepcopy(VALID_CONFIG)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return document


class ExampleConfigTests(unittest.TestCase):
    def test_example_node_config_is_valid(self) -> None:
        config = load_node_config(Path("config/node-config.example.json5"))

        self.assertIsInstance(config.estimator, TransformersEstimatorConfig)
        self.assertEqual(
            dict(config.streams),
            {"front": "camera/front", "view_main": "spherical_camera/view/main"},
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

    def test_invalid_configs_are_rejected(self) -> None:
        invalid_configs = {
            "unknown field": config_with(("unknown",), 1),
            "unknown estimator field": config_with(("estimator", "process_res"), 504),
            "missing backend": config_with(
                ("estimator",), {"model_id": "org/model", "device": "cpu"}
            ),
            "unknown backend": config_with(("estimator", "backend"), "onnx"),
            "blank model id": config_with(("estimator", "model_id"), " "),
            "unknown dtype": config_with(("estimator", "dtype"), "float64"),
            "wildcard prefix": config_with(("zenoh_key_prefix",), "depth/*"),
            "blank prefix": config_with(("zenoh_key_prefix",), " "),
            "multi-segment stream name": config_with(
                ("streams",), {"front/left": "camera/front"}
            ),
            "wildcard stream name": config_with(("streams",), {"*": "camera/front"}),
            "blank stream name": config_with(("streams",), {" ": "camera/front"}),
            "wildcard source key": config_with(("streams",), {"front": "camera/*"}),
            "duplicate source keys": config_with(
                ("streams",), {"front": "camera/front", "again": "camera/front"}
            ),
            "depth key equal to a source key": config_with(
                ("streams",), {"front": "camera/front", "loop": "depth/front"}
            ),
            "empty streams": config_with(("streams",), {}),
            "unknown reliability": config_with(("publisher", "reliability"), "maybe"),
        }
        for case, document in invalid_configs.items():
            with self.subTest(case=case), self.assertRaises(ValidationError):
                NodeConfig.model_validate(document)

    def test_loaded_config_is_immutable(self) -> None:
        config = NodeConfig.model_validate(VALID_CONFIG)
        streams: Any = config.streams

        with self.assertRaises(ValidationError):
            config.zenoh_key_prefix = "other"
        with self.assertRaises(ValidationError):
            config.estimator.device = "cuda"
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

    def test_validation_errors_name_the_file(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"node\.json5: invalid node configuration"
        ):
            self.load_text("{}")


if __name__ == "__main__":
    unittest.main()
