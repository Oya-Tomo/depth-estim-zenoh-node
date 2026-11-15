"""Depth estimation backends behind one interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config import (
    DepthAnything3EstimatorConfig,
    EstimatorConfig,
    TransformersEstimatorConfig,
)

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """Return the depth of an RGB image as an array of shape (height, width)."""


class TransformersEstimator(DepthEstimator):
    def __init__(self, config: TransformersEstimatorConfig) -> None:
        self._device = torch.device(config.device)
        self._dtype = TORCH_DTYPES[config.dtype]
        self._processor = AutoImageProcessor.from_pretrained(
            config.model_id, revision=config.revision
        )
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(
                config.model_id, revision=config.revision, dtype=self._dtype
            )
            .to(self._device)
            .eval()
        )

    @torch.inference_mode()
    def estimate(self, image: Image.Image) -> npt.NDArray[np.float32]:
        inputs = self._processor(
            images=image, return_tensors="pt", device=self._device
        ).to(dtype=self._dtype)
        outputs = self._model(**inputs)
        # The size is positional because the parameter is `source_sizes` for ZoeDepth
        # and `target_sizes` for the other models; both yield depth at the image size.
        depth = self._processor.post_process_depth_estimation(
            outputs, [(image.height, image.width)]
        )[0]["predicted_depth"]
        return depth.float().cpu().numpy()


class DepthAnything3Estimator(DepthEstimator):
    def __init__(self, config: DepthAnything3EstimatorConfig) -> None:
        # The logger of depth_anything_3 reads its level at import time and
        # otherwise prints INFO lines to stdout on every inference.
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
        self._process_res = config.process_res
        self._model = DepthAnything3.from_pretrained(
            config.model_id, revision=config.revision
        ).to(torch.device(config.device))

    def estimate(self, image: Image.Image) -> npt.NDArray[np.float32]:
        prediction = self._model.inference([image], process_res=self._process_res)
        depth = prediction.depth[0]
        if not prediction.is_metric:
            # Non-metric depth is either up to scale or canonical (DA3METRIC):
            # meters = depth * focal_px / 300 with the focal length of the processed
            # image. Rescaling makes that hold with the focal length of the source
            # frame. Metric depth is already in meters.
            processed_height, processed_width = depth.shape
            depth = depth * (
                (processed_width / image.width + processed_height / image.height) / 2
            )
        return np.asarray(
            Image.fromarray(depth).resize(image.size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )


def load_estimator(config: EstimatorConfig) -> DepthEstimator:
    match config:
        case TransformersEstimatorConfig():
            return TransformersEstimator(config)
        case DepthAnything3EstimatorConfig():
            return DepthAnything3Estimator(config)
