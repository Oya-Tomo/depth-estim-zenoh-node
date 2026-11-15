import io
import unittest
import warnings
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import matplotlib
import numpy as np
import numpy.typing as npt
import torch
import zenoh
from PIL import Image

from config import BaseEstimatorConfig, ImageConfig, load_node_config
from estim import DepthEstimator
from node import (
    IDLE_POLL_INTERVAL_SECONDS,
    JPEG_ENCODING,
    NPY_ENCODING,
    DepthImageEncoder,
    main,
    publish_depth,
    run,
)

EXAMPLE_CONFIG = load_node_config(Path("config/node-config.example.json5"))
IMAGE_CONFIG = ImageConfig(colormap="turbo", value_range=(0, 5), jpeg_quality=95)
LOW_COLOR, HIGH_COLOR = (
    matplotlib.colormaps["turbo"](end, bytes=True)[:3] for end in (0.0, 1.0)
)
# Largest error of a flat color after JPEG coding.
JPEG_TOLERANCE = 4
NAN, INF = float("nan"), float("inf")


class StopLoop(Exception):
    pass


class CountingEstimator(DepthEstimator):
    """Fill the depth of the n-th estimated image with n."""

    def __init__(self) -> None:
        super().__init__(BaseEstimatorConfig(model_id="counting", device="cpu"))
        self.estimated = 0

    def estimate(self, image: torch.Tensor) -> torch.Tensor:
        self.estimated += 1
        return torch.full(image.shape[-2:], float(self.estimated))


def jpeg(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


def pixels(jpeg_image: bytes) -> npt.NDArray[np.int16]:
    image = Image.open(io.BytesIO(jpeg_image))
    assert (image.format, image.mode) == ("JPEG", "RGB")
    return np.asarray(image, dtype=np.int16)


def halves(left: float, right: float) -> torch.Tensor:
    """A 32x64 depth map with one value in each half."""
    return torch.tensor([[left, right]]).float().repeat_interleave(32, 1).expand(32, 64)


def sample(key: str, payload: bytes, attachment: bytes | None = None) -> MagicMock:
    received = MagicMock(key_expr=key, attachment=attachment)
    received.payload.to_bytes.return_value = payload
    return received


def subscriber(samples: list[MagicMock]) -> MagicMock:
    pending = MagicMock()
    pending.try_recv.side_effect = lambda: samples.pop(0) if samples else None
    return pending


class DepthImageEncoderTests(unittest.TestCase):
    def assert_halves_have_colors(
        self,
        depth: torch.Tensor,
        left: Sequence[float],
        right: Sequence[float],
        **config: object,
    ) -> None:
        encoder = DepthImageEncoder(
            IMAGE_CONFIG.model_copy(update=config), depth.device
        )

        image = pixels(encoder.encode(depth))

        self.assertEqual(image.shape, (32, 64, 3))
        # The columns next to the edge between the halves are left out.
        for half, color in ((image[:, :24], left), (image[:, 40:], right)):
            self.assertLessEqual(np.abs(half - color).max(), JPEG_TOLERANCE)

    def test_image_is_an_rgb_jpeg_of_the_size_of_the_depth(self) -> None:
        encoder = DepthImageEncoder(IMAGE_CONFIG, torch.device("cpu"))

        image = pixels(encoder.encode(torch.rand(30, 40) * 5))

        self.assertEqual(image.shape, (30, 40, 3))

    def test_value_range_maps_to_the_ends_of_the_colormap(self) -> None:
        self.assert_halves_have_colors(halves(0, 5), LOW_COLOR, HIGH_COLOR)

    def test_values_between_the_ends_follow_the_colormap(self) -> None:
        middle_color = matplotlib.colormaps["turbo"](0.5, bytes=True)[:3]

        self.assert_halves_have_colors(halves(2.5, 5), middle_color, HIGH_COLOR)

    def test_reversed_colormap_swaps_the_ends(self) -> None:
        self.assert_halves_have_colors(
            halves(0, 5), HIGH_COLOR, LOW_COLOR, colormap="turbo_r"
        )

    def test_values_outside_the_range_are_clipped(self) -> None:
        self.assert_halves_have_colors(halves(-3, 1000), LOW_COLOR, HIGH_COLOR)

    def test_non_finite_values_map_to_the_ends(self) -> None:
        # The finite values 0 and 5 also give the frame a range of its own.
        depth = halves(0, 5).clone()
        depth[:8, :32] = NAN
        depth[8:16, :32] = -INF
        depth[:16, 32:] = INF
        for value_range in ((0, 5), None):
            with self.subTest(value_range=value_range):
                self.assert_halves_have_colors(
                    depth, LOW_COLOR, HIGH_COLOR, value_range=value_range
                )

    def test_frame_range_is_used_without_a_value_range(self) -> None:
        self.assert_halves_have_colors(
            halves(100, 101), LOW_COLOR, HIGH_COLOR, value_range=None
        )

    def test_frame_without_a_range_maps_to_the_low_end(self) -> None:
        for value in (NAN, INF, -INF, 2.0):
            with self.subTest(value=value), warnings.catch_warnings():
                warnings.simplefilter("error")

                self.assert_halves_have_colors(
                    halves(value, value), LOW_COLOR, LOW_COLOR, value_range=None
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_image_matches_the_cpu_image(self) -> None:
        depth = torch.linspace(-1, 6, 64).expand(32, 64).clone()
        depth[0, :3] = torch.tensor([NAN, INF, -INF])
        images = [
            pixels(
                DepthImageEncoder(IMAGE_CONFIG, torch.device(device)).encode(
                    depth.to(device)
                )
            )
            for device in ("cpu", "cuda")
        ]

        self.assertEqual(images[0].shape, images[1].shape)
        self.assertLess(np.abs(images[0] - images[1]).mean(), JPEG_TOLERANCE)


class PublishDepthTests(unittest.TestCase):
    def publish_until_idle(
        self, streams: list[tuple[MagicMock, MagicMock, MagicMock]]
    ) -> str:
        """Run `publish_depth` until a round without samples; return its stderr."""
        stderr = StringIO()
        with (
            patch("node.time.sleep", side_effect=StopLoop) as sleep,
            redirect_stderr(stderr),
            self.assertRaises(StopLoop),
        ):
            publish_depth(
                streams,
                CountingEstimator(),
                DepthImageEncoder(IMAGE_CONFIG, torch.device("cpu")),
            )
        sleep.assert_called_once_with(IDLE_POLL_INTERVAL_SECONDS)
        return stderr.getvalue()

    def test_image_and_raw_depth_are_published_with_the_source_metadata(self) -> None:
        source = sample("camera/front", jpeg((40, 30)), b'{"yaw_deg": 90.0}')
        image_publisher, raw_publisher = MagicMock(), MagicMock()

        self.publish_until_idle(
            [(subscriber([source]), image_publisher, raw_publisher)]
        )

        for publisher in (image_publisher, raw_publisher):
            publisher.put.assert_called_once_with(
                ANY, attachment=source.attachment, timestamp=source.timestamp
            )
        image = pixels(image_publisher.put.call_args.args[0])
        self.assertEqual(image.shape, (30, 40, 3))
        depth = np.load(
            io.BytesIO(raw_publisher.put.call_args.args[0]), allow_pickle=False
        )
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_array_equal(depth, np.ones((30, 40), dtype=np.float32))

    def test_codec_path_raises_no_warning(self) -> None:
        image_publisher, raw_publisher = MagicMock(), MagicMock()
        source = sample("camera/front", jpeg((40, 30)))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.publish_until_idle(
                [(subscriber([source]), image_publisher, raw_publisher)]
            )

        image_publisher.put.assert_called_once()
        raw_publisher.put.assert_called_once()

    def test_undecodable_image_is_skipped_without_stopping_other_streams(self) -> None:
        png = io.BytesIO()
        Image.new("RGB", (40, 30)).save(png, format="PNG")
        payloads = {
            "garbage": b"not an image",
            "truncated": jpeg((40, 30))[:200],
            "empty": b"",
            "PNG": png.getvalue(),
        }
        for case, payload in payloads.items():
            with self.subTest(case=case):
                broken = (MagicMock(), MagicMock())
                intact = (MagicMock(), MagicMock())

                stderr = self.publish_until_idle(
                    [
                        (subscriber([sample("camera/a", payload)]), *broken),
                        (subscriber([sample("camera/b", jpeg((40, 30)))]), *intact),
                    ]
                )

                for publisher in broken:
                    publisher.put.assert_not_called()
                for publisher in intact:
                    publisher.put.assert_called_once()
                self.assertIn("[WARNING]", stderr)
                self.assertIn("camera/a", stderr)

    def test_out_of_memory_while_decoding_is_not_skipped(self) -> None:
        errors = [
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB."),
            torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB."),
        ]
        for error in errors:
            stderr = StringIO()
            with (
                self.subTest(error=type(error).__name__),
                patch("node.decode_jpeg", side_effect=error),
                redirect_stderr(stderr),
                self.assertRaisesRegex(RuntimeError, "out of memory"),
            ):
                publish_depth(
                    [
                        (
                            subscriber([sample("camera/front", jpeg((40, 30)))]),
                            MagicMock(),
                            MagicMock(),
                        )
                    ],
                    CountingEstimator(),
                    DepthImageEncoder(IMAGE_CONFIG, torch.device("cpu")),
                )
            self.assertEqual(stderr.getvalue(), "")

    def test_streams_take_turns_one_frame_at_a_time(self) -> None:
        published: list[tuple[str, bytes | None]] = []

        def publisher(output: str) -> MagicMock:
            recording = MagicMock()
            recording.put.side_effect = lambda _, *, attachment, timestamp: (
                published.append((output, attachment))
            )
            return recording

        streams = []
        for name in ("a", "b"):
            samples = [
                sample(f"camera/{name}", jpeg((8, 8)), f"{name}{index}".encode())
                for index in range(3)
            ]
            streams.append((subscriber(samples), publisher("image"), publisher("raw")))

        self.publish_until_idle(streams)

        self.assertEqual(
            published,
            [
                (output, attachment)
                for attachment in (b"a0", b"b0", b"a1", b"b1", b"a2", b"b2")
                for output in ("image", "raw")
            ],
        )


class RunTests(unittest.TestCase):
    def test_model_is_loaded_before_each_stream_is_declared(self) -> None:
        events = MagicMock()
        events.load_estimator.return_value.device = torch.device("cpu")
        session = events.open.return_value.__enter__.return_value
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch("node.load_estimator", events.load_estimator),
            patch("node.zenoh.open", events.open),
            patch("node.publish_depth", side_effect=StopLoop) as publish,
            self.assertRaises(StopLoop),
        ):
            run(MagicMock(), EXAMPLE_CONFIG)

        self.assertEqual(
            [name for name, _, _ in events.mock_calls[:2]], ["load_estimator", "open"]
        )
        self.assertEqual(
            [call.args[0] for call in session.declare_subscriber.call_args_list],
            ["camera/front", "spherical_camera/view/main"],
        )
        self.assertEqual(
            session.declare_publisher.call_args_list,
            [
                call(
                    key,
                    encoding=encoding,
                    congestion_control=zenoh.CongestionControl.DROP,
                    reliability=zenoh.Reliability.BEST_EFFORT,
                )
                for key, encoding in (
                    ("depth/front", JPEG_ENCODING),
                    ("depth/front/raw", NPY_ENCODING),
                    ("depth/view_main", JPEG_ENCODING),
                    ("depth/view_main/raw", NPY_ENCODING),
                )
            ],
        )
        streams, estimator, image_encoder = publish.call_args.args
        self.assertEqual([len(stream) for stream in streams], [3, 3])
        self.assertIs(estimator, events.load_estimator.return_value)
        self.assertIsInstance(image_encoder, DepthImageEncoder)
        for key in ("camera/front", "depth/front", "depth/front/raw"):
            self.assertIn(f"'{key}'", stdout.getvalue())


class MainTests(unittest.TestCase):
    def test_missing_node_config_is_reported_as_an_error(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            status = main(["--node-config", "config/missing.json5"])

        self.assertEqual(status, 1)
        self.assertIn("[ERROR]", stderr.getvalue())
        self.assertIn("config/missing.json5", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
