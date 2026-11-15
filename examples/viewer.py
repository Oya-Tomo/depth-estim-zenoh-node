"""Display a JPEG image stream received over Zenoh with pygame."""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path

import pygame
import zenoh
from PIL import Image

DEFAULT_ZENOH_CONFIG_PATH = Path(__file__).with_name("viewer-zenoh-config.json5")
INITIAL_WINDOW_SIZE = (640, 480)
EVENT_POLL_INTERVAL_MS = 10
QUIT_KEYS = {pygame.K_q, pygame.K_ESCAPE}


def parse_concrete_key(value: str) -> str:
    try:
        zenoh.KeyExpr(value)
    except zenoh.ZError as error:
        raise argparse.ArgumentTypeError(f"invalid Zenoh key: {error}") from error
    if not value.strip() or "*" in value:
        raise argparse.ArgumentTypeError("image key must be a key without wildcards")
    return value


def run(zenoh_config: zenoh.Config, key_expression: str) -> None:
    zenoh.init_log_from_env_or("error")
    with (
        zenoh.open(zenoh_config) as session,
        session.declare_subscriber(
            key_expression, zenoh.handlers.RingChannel(1)
        ) as subscriber,
    ):
        print(f"[INFO] Showing '{key_expression}'.")
        print("[INFO] Press q, Esc, or close the window to quit.")

        pygame.display.init()
        try:
            pygame.display.set_caption(key_expression)
            screen = pygame.display.set_mode(INITIAL_WINDOW_SIZE)
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in QUIT_KEYS
                    ):
                        return

                sample = subscriber.try_recv()
                if sample is None:
                    pygame.time.wait(EVENT_POLL_INTERVAL_MS)
                    continue
                try:
                    image = Image.open(io.BytesIO(sample.payload.to_bytes())).convert(
                        "RGB"
                    )
                except OSError as error:
                    print(
                        f"[WARNING] Skipped an undecodable image: {error}",
                        file=sys.stderr,
                    )
                    continue
                if screen.get_size() != image.size:
                    screen = pygame.display.set_mode(image.size)
                screen.blit(
                    pygame.image.frombytes(image.tobytes(), image.size, "RGB"), (0, 0)
                )
                pygame.display.flip()
        finally:
            pygame.display.quit()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Display a JPEG image stream published over Zenoh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "key_expression",
        type=parse_concrete_key,
        help="Zenoh key of a JPEG image stream, such as depth/front.",
    )
    parser.add_argument(
        "--zenoh-config",
        type=Path,
        default=DEFAULT_ZENOH_CONFIG_PATH,
        metavar="FILE",
        help="Zenoh JSON5 configuration file.",
    )
    args = parser.parse_args(argv)

    try:
        run(zenoh.Config.from_file(args.zenoh_config), args.key_expression)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    except (OSError, pygame.error, zenoh.ZError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
