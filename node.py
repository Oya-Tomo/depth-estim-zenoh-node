"""Estimate depth from Zenoh image streams and publish it over Zenoh."""

from __future__ import annotations

import argparse
import io
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import NoReturn

import numpy as np
import zenoh
from PIL import Image

from config import NodeConfig, load_node_config
from estim import DepthEstimator, load_estimator

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


def publish_depth(
    streams: Sequence[
        tuple[zenoh.Subscriber[zenoh.handlers.Handler[zenoh.Sample]], zenoh.Publisher]
    ],
    estimator: DepthEstimator,
) -> NoReturn:
    while True:
        idle = True
        for subscriber, publisher in streams:
            sample = subscriber.try_recv()
            if sample is None:
                continue
            idle = False
            try:
                image = Image.open(io.BytesIO(sample.payload.to_bytes())).convert("RGB")
            except OSError as error:
                print(
                    f"[WARNING] Skipped an undecodable image on '{sample.key_expr}': "
                    f"{error}",
                    file=sys.stderr,
                )
                continue
            buffer = io.BytesIO()
            np.save(buffer, estimator.estimate(image), allow_pickle=False)
            publisher.put(
                buffer.getvalue(),
                attachment=sample.attachment,
                timestamp=sample.timestamp,
            )
        if idle:
            time.sleep(IDLE_POLL_INTERVAL_SECONDS)


def run(zenoh_config: zenoh.Config, node_config: NodeConfig) -> NoReturn:
    zenoh.init_log_from_env_or("error")
    estimator_config = node_config.estimator
    publisher_config = node_config.publisher

    # The model is loaded before the session opens: the first start may download it
    # for minutes.
    print(
        f"[INFO] Load depth model '{estimator_config.model_id}' "
        f"({estimator_config.backend}) on {estimator_config.device}..."
    )
    estimator = load_estimator(estimator_config)

    print("[INFO] Open Zenoh session...")
    with zenoh.open(zenoh_config) as session, ExitStack() as resources:
        streams = []
        for name, source_key in node_config.streams.items():
            depth_key = f"{node_config.zenoh_key_prefix}/{name}"
            # A ring of one keeps only the newest frame of each stream.
            subscriber = resources.enter_context(
                session.declare_subscriber(source_key, zenoh.handlers.RingChannel(1))
            )
            publisher = resources.enter_context(
                session.declare_publisher(
                    depth_key,
                    encoding=NPY_ENCODING,
                    congestion_control=CONGESTION_CONTROLS[
                        publisher_config.congestion_control
                    ],
                    reliability=RELIABILITIES[publisher_config.reliability],
                )
            )
            streams.append((subscriber, publisher))
            print(f"[INFO] Publishing depth of '{source_key}' on '{depth_key}'.")

        print("[INFO] Press CTRL-C to quit.")
        publish_depth(streams, estimator)


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
