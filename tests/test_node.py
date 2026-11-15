import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import numpy.typing as npt
import zenoh
from PIL import Image

from config import load_node_config
from estim import DepthEstimator
from node import IDLE_POLL_INTERVAL_SECONDS, NPY_ENCODING, main, publish_depth, run

EXAMPLE_CONFIG = load_node_config(Path("config/node-config.example.json5"))


class StopLoop(Exception):
    pass


class CountingEstimator(DepthEstimator):
    """Fill the depth of the n-th estimated image with n."""

    def __init__(self) -> None:
        self.estimated = 0

    def estimate(self, image: Image.Image) -> npt.NDArray[np.float32]:
        self.estimated += 1
        return np.full((image.height, image.width), self.estimated, dtype=np.float32)


def jpeg(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


def sample(key: str, payload: bytes, attachment: bytes | None = None) -> MagicMock:
    received = MagicMock(key_expr=key, attachment=attachment)
    received.payload.to_bytes.return_value = payload
    return received


def subscriber(samples: list[MagicMock]) -> MagicMock:
    pending = MagicMock()
    pending.try_recv.side_effect = lambda: samples.pop(0) if samples else None
    return pending


def publish_until_idle(streams: list[tuple[MagicMock, MagicMock]]) -> str:
    stderr = StringIO()
    with (
        patch("node.time.sleep", side_effect=StopLoop) as sleep,
        redirect_stderr(stderr),
        unittest.TestCase().assertRaises(StopLoop),
    ):
        publish_depth(streams, CountingEstimator())
    sleep.assert_called_once_with(IDLE_POLL_INTERVAL_SECONDS)
    return stderr.getvalue()


class PublishDepthTests(unittest.TestCase):
    def test_depth_is_published_as_npy_with_the_source_metadata(self) -> None:
        source = sample("camera/front", jpeg((40, 30)), b'{"yaw_deg": 90.0}')
        publisher = MagicMock()

        publish_until_idle([(subscriber([source]), publisher)])

        payload = publisher.put.call_args.args[0]
        depth = np.load(io.BytesIO(payload), allow_pickle=False)
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_array_equal(depth, np.ones((30, 40), dtype=np.float32))
        self.assertEqual(
            publisher.put.call_args.kwargs,
            {"attachment": source.attachment, "timestamp": source.timestamp},
        )

    def test_undecodable_image_is_skipped_without_stopping_other_streams(self) -> None:
        payloads = {"garbage": b"not an image", "truncated": jpeg((40, 30))[:200]}
        for case, payload in payloads.items():
            with self.subTest(case=case):
                broken_publisher, intact_publisher = MagicMock(), MagicMock()

                stderr = publish_until_idle(
                    [
                        (subscriber([sample("camera/a", payload)]), broken_publisher),
                        (
                            subscriber([sample("camera/b", jpeg((40, 30)))]),
                            intact_publisher,
                        ),
                    ]
                )

                broken_publisher.put.assert_not_called()
                intact_publisher.put.assert_called_once()
                self.assertIn("[WARNING]", stderr)
                self.assertIn("camera/a", stderr)

    def test_streams_take_turns_one_frame_at_a_time(self) -> None:
        published: list[bytes | None] = []
        streams = []
        for name in ("a", "b"):
            publisher = MagicMock()
            publisher.put.side_effect = lambda _, *, attachment, timestamp: (
                published.append(attachment)
            )
            samples = [
                sample(f"camera/{name}", jpeg((8, 8)), f"{name}{index}".encode())
                for index in range(3)
            ]
            streams.append((subscriber(samples), publisher))

        publish_until_idle(streams)

        self.assertEqual(published, [b"a0", b"b0", b"a1", b"b1", b"a2", b"b2"])


class RunTests(unittest.TestCase):
    def test_model_is_loaded_before_each_stream_is_declared(self) -> None:
        events = MagicMock()
        session = events.open.return_value.__enter__.return_value
        with (
            redirect_stdout(StringIO()),
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
                    encoding=NPY_ENCODING,
                    congestion_control=zenoh.CongestionControl.DROP,
                    reliability=zenoh.Reliability.BEST_EFFORT,
                )
                for key in ("depth/front", "depth/view_main")
            ],
        )
        streams, estimator = publish.call_args.args
        self.assertEqual(len(streams), 2)
        self.assertIs(estimator, events.load_estimator.return_value)


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
