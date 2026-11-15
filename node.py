"""Estimate depth from Zenoh image streams and publish it over Zenoh."""

from __future__ import annotations

import argparse
import io
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import NoReturn, cast

import matplotlib
import numpy as np
import torch
import zenoh

# TorchCodec re-exports its public API without `__all__`, which pyright reports.
from torchcodec.decoders import decode_jpeg  # pyright: ignore[reportPrivateImportUsage]
from torchcodec.encoders import JpegEncoder  # pyright: ignore[reportPrivateImportUsage]

from config import ImageConfig, NodeConfig, load_node_config
from estim import DepthEstimator, load_estimator

JPEG_ENCODING = "image/jpeg"
NPY_ENCODING = "application/x-npy"
DEFAULT_ZENOH_CONFIG_PATH = Path("config/zenoh-config.json5")
DEFAULT_NODE_CONFIG_PATH = Path("config/node-config.json5")
IDLE_POLL_INTERVAL_SECONDS = 0.005
CONGESTION_CONTROLS = {
    "drop": zenoh.CongestionControl.DROP,
    "block": zenoh.CongestionControl.BLOCK,
    "block_first": zenoh.CongestionControl.BLOCK_FIRST,
}
RELIABILITIES = {
    "best_effort": zenoh.Reliability.BEST_EFFORT,
    "reliable": zenoh.Reliability.RELIABLE,
}


class DepthImageEncoder:
    """Turn a depth map into the published image."""

    def __init__(self, config: ImageConfig, device: torch.device) -> None:
        colormap = matplotlib.colormaps[config.colormap]
        self._color_table = torch.tensor(
            colormap(np.linspace(0, 1, 256), bytes=True)[:, :3], device=device
        )
        self._value_range = config.value_range
        self._jpeg_quality = config.jpeg_quality

    def encode(self, depth: torch.Tensor) -> bytes:
        if self._value_range is None:
            finite = depth[depth.isfinite()]
            low, high = (finite.min(), finite.max()) if finite.numel() else (0, 0)
        else:
            low, high = self._value_range
        # 256 equal bins from low to high; a frame without a range gets scale 0,
        # which maps every value, even infinite ones, to the low end.
        scale = 256 / (high - low) if high > low else 0
        # The index must be int64: a uint8 index would be taken as a mask.
        index = (
            ((depth - low) * scale)
            .nan_to_num(nan=0, posinf=255, neginf=0)
            .clamp(0, 255)
            .long()
        )
        image = self._color_table[index].permute(2, 0, 1).contiguous()
        jpeg = JpegEncoder(image).to_tensor(quality=self._jpeg_quality)
        return jpeg.cpu().numpy().tobytes()


def publish_depth(
    streams: Sequence[
        tuple[
            zenoh.Subscriber[zenoh.handlers.Handler[zenoh.Sample]],
            zenoh.Publisher,
            zenoh.Publisher,
        ]
    ],
    estimator: DepthEstimator,
    image_encoder: DepthImageEncoder,
) -> NoReturn:
    while True:
        idle = True
        for subscriber, image_publisher, raw_publisher in streams:
            sample = subscriber.try_recv()
            if sample is None:
                continue
            idle = False
            try:
                # decode_jpeg returns a list only for a list of sources.
                image = cast(
                    torch.Tensor,
                    decode_jpeg(sample.payload.to_bytes(), device=estimator.device),
                )
            except (RuntimeError, ValueError) as error:
                # The decoder reports running out of memory as a plain RuntimeError
                # too, but that is fatal rather than a fault of the payload.
                if "out of memory" in str(error):
                    raise
                print(
                    f"[WARNING] Skipped an undecodable image on '{sample.key_expr}': "
                    f"{error}",
                    file=sys.stderr,
                )
                continue
            depth = estimator.estimate(image)
            image_publisher.put(
                image_encoder.encode(depth),
                attachment=sample.attachment,
                timestamp=sample.timestamp,
            )
            raw = io.BytesIO()
            np.save(raw, depth.cpu().numpy(), allow_pickle=False)
            raw_publisher.put(
                raw.getvalue(),
                attachment=sample.attachment,
                timestamp=sample.timestamp,
            )
        if idle:
            time.sleep(IDLE_POLL_INTERVAL_SECONDS)


def run(zenoh_config: zenoh.Config, node_config: NodeConfig) -> NoReturn:
    zenoh.init_log_from_env_or("error")
    estimator_config = node_config.estimator
    congestion_control = CONGESTION_CONTROLS[node_config.publisher.congestion_control]
    reliability = RELIABILITIES[node_config.publisher.reliability]

    # The model is loaded before the session opens: the first start may download it
    # for minutes.
    print(
        f"[INFO] Load depth model '{estimator_config.model_id}' "
        f"({estimator_config.backend}) on {estimator_config.device}..."
    )
    estimator = load_estimator(estimator_config)
    image_encoder = DepthImageEncoder(node_config.image, estimator.device)

    print("[INFO] Open Zenoh session...")
    with zenoh.open(zenoh_config) as session, ExitStack() as resources:
        streams = []
        for name, source_key in node_config.streams.items():
            image_key = f"{node_config.zenoh_key_prefix}/{name}"
            raw_key = f"{image_key}/raw"
            # A ring of one keeps only the newest frame of each stream.
            subscriber = resources.enter_context(
                session.declare_subscriber(source_key, zenoh.handlers.RingChannel(1))
            )
            image_publisher = resources.enter_context(
                session.declare_publisher(
                    image_key,
                    encoding=JPEG_ENCODING,
                    congestion_control=congestion_control,
                    reliability=reliability,
                )
            )
            raw_publisher = resources.enter_context(
                session.declare_publisher(
                    raw_key,
                    encoding=NPY_ENCODING,
                    congestion_control=congestion_control,
                    reliability=reliability,
                )
            )
            streams.append((subscriber, image_publisher, raw_publisher))
            print(
                f"[INFO] Publishing depth of '{source_key}' on '{image_key}' (image) "
                f"and '{raw_key}' (raw)."
            )

        print("[INFO] Press CTRL-C to quit.")
        publish_depth(streams, estimator, image_encoder)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="depth-estim-zenoh-node",
        description="Estimate depth from Zenoh image streams and publish it over Zenoh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--zenoh-config",
        type=Path,
        default=DEFAULT_ZENOH_CONFIG_PATH,
        metavar="FILE",
        help="Zenoh JSON5 configuration file.",
    )
    parser.add_argument(
        "--node-config",
        type=Path,
        default=DEFAULT_NODE_CONFIG_PATH,
        metavar="FILE",
        help="Node JSON5 configuration file.",
    )
    args = parser.parse_args(argv)

    try:
        node_config = load_node_config(args.node_config)
        run(zenoh.Config.from_file(args.zenoh_config), node_config)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    except (RuntimeError, ValueError, OSError, zenoh.ZError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
