"""Depth estimation backends behind one interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
from torch.nn.functional import interpolate
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config import (
    BaseEstimatorConfig,
    DepthAnything3EstimatorConfig,
    EstimatorConfig,
    TransformersEstimatorConfig,
)

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DEPTH_ANYTHING_3_PATCH_SIZE = 14


class DepthEstimator(ABC):
    def __init__(self, config: BaseEstimatorConfig) -> None:
        self.device = torch.device(config.device)

    @abstractmethod
    def estimate(self, image: torch.Tensor) -> torch.Tensor:
        """Return the depth of an RGB image, both on `self.device`.

        `image` is uint8 of shape (3, height, width); the depth is float32 of shape
        (height, width).
        """


class TransformersEstimator(DepthEstimator):
    def __init__(self, config: TransformersEstimatorConfig) -> None:
        super().__init__(config)
        self._dtype = TORCH_DTYPES[config.dtype]
        self._processor = AutoImageProcessor.from_pretrained(
            config.model_id, revision=config.revision
        )
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(
                config.model_id, revision=config.revision, dtype=self._dtype
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def estimate(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        # The torchvision-backend processors keep a tensor input on its device.
        inputs = self._processor(images=image, return_tensors="pt").to(
            dtype=self._dtype
        )
        outputs = self._model(**inputs)
        # Positional: the parameter is `source_sizes` for ZoeDepth, else `target_sizes`.
        return self._processor.post_process_depth_estimation(
            outputs, [(height, width)]
        )[0]["predicted_depth"].float()


class DepthAnything3Estimator(DepthEstimator):
    def __init__(self, config: DepthAnything3EstimatorConfig) -> None:
        super().__init__(config)
        # The package's logger reads its level at import time and otherwise prints
        # INFO lines to stdout.
        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
        try:
            from depth_anything_3.api import DepthAnything3
        except ModuleNotFoundError as error:
            if error.name != "depth_anything_3":
                raise
            raise RuntimeError(
                "the depth_anything_3 backend requires the depth-anything-3 "
                "dependency group: uv sync --group depth-anything-3"
            ) from error
        self._model = DepthAnything3.from_pretrained(
            config.model_id, revision=config.revision
        ).to(self.device)
        self._process_res = config.process_res
        # The ImageNet statistics the model normalizes its input with.
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    @torch.inference_mode()
    def estimate(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        scale = self._process_res / max(height, width)
        processed_height, processed_width = (
            max(1, round(side * scale / DEPTH_ANYTHING_3_PATCH_SIZE))
            * DEPTH_ANYTHING_3_PATCH_SIZE
            for side in (height, width)
        )
        # Resized here on the device: the package's `inference()` resizes on the CPU
        # with PIL and OpenCV.
        batch = interpolate(
            image[None] / 255,
            size=(processed_height, processed_width),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
        batch = (batch - self._mean) / self._std
        # `forward` applies autocast itself; its default `export_feat_layers=None`
        # crashes.
        output = self._model(batch[None], export_feat_layers=[])
        depth = output["depth"][0, 0]
        # `output` is an addict.Dict, where a missing attribute is an empty Dict.
        if not output.get("is_metric", 0):
            # Non-metric depth is either up to scale (unaffected) or the canonical
            # depth of DA3METRIC: meters = depth * focal_px / 300 with the focal
            # length of the processed image. The factor makes that hold with the
            # focal length of the source frame. Metric depth (DA3NESTED) is meters.
            depth = depth * ((processed_width / width + processed_height / height) / 2)
        depth = interpolate(depth[None, None], size=(height, width), mode="bilinear")
        return depth[0, 0].float()


def load_estimator(config: EstimatorConfig) -> DepthEstimator:
    match config:
        case TransformersEstimatorConfig():
            return TransformersEstimator(config)
        case DepthAnything3EstimatorConfig():
            return DepthAnything3Estimator(config)
