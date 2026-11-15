# Usage

## Running the node

```console
$ uv run node.py
```

The node loads the model first, which includes the download on the first start, and then opens the Zenoh session. Press `Ctrl-C` to stop.

## Node configuration

See [`config/node-config.example.json5`](../../config/node-config.example.json5) for a complete example. Unknown, missing, duplicate, or mistyped fields are rejected at startup.

| Key | Description |
| --- | --- |
| `zenoh_key_prefix` | Concrete Zenoh key under which every depth stream is published |
| `estimator.backend` | `transformers` or `depth_anything_3`; one model is loaded per process |
| `estimator.model_id` | Hugging Face model ID or a local directory |
| `estimator.revision` | Branch, tag, or commit hash of the model; defaults to `main` |
| `estimator.device` | PyTorch device, such as `cuda`, `cuda:1`, or `cpu` |
| `estimator.dtype` | `transformers` only: `float32`, `float16`, or `bfloat16`; defaults to `float32` |
| `estimator.process_res` | `depth_anything_3` only: longer side of the processed image in pixels; defaults to `504` |
| `streams` | Map from a stream name to the concrete Zenoh key of the image stream to estimate depth from; at least one entry |
| `publisher.congestion_control` | `drop`, `block`, or `block_first` |
| `publisher.reliability` | `best_effort` or `reliable` |

A stream name is a single key segment. Its depth is published on `{zenoh_key_prefix}/{name}`, so the example publishes the depth of `camera/front` on `depth/front`. Source keys must be unique, and no depth key may equal a source key.

On CUDA, `float16` reduces the GPU memory and usually the inference time of the `transformers` backend. The `depth_anything_3` backend has no `dtype` because the model selects mixed precision by itself.

## Models

The node works with models whose weights are public. The meaning of the published values depends on the model:

| `backend` | `model_id` | Output | License |
| --- | --- | --- | --- |
| `transformers` | `depth-anything/Depth-Anything-V2-Small-hf` | Relative inverse depth (larger = nearer) | Apache-2.0 |
| `transformers` | `depth-anything/Depth-Anything-V2-Base-hf`, `-Large-hf` | Relative inverse depth (larger = nearer) | CC-BY-NC-4.0 |
| `transformers` | `depth-anything/Depth-Anything-V2-Metric-Indoor-{Small,Base,Large}-hf`, `-Metric-Outdoor-{Small,Base,Large}-hf` | Meters (scale learned for the training cameras) | See the model card |
| `transformers` | `apple/DepthPro-hf` | Meters (estimates the field of view itself; slow) | Apple AMLR (research only) |
| `depth_anything_3` | `depth-anything/DA3-SMALL`, `DA3-BASE` | Relative depth (larger = farther) | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3MONO-LARGE` | Relative depth (larger = farther) | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3METRIC-LARGE` | Canonical depth; meters = value × focal_px / 300, with the focal length of the source frame in pixels | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3-LARGE-1.1`, `DA3-GIANT-1.1` | Relative depth (larger = farther) | CC-BY-NC-4.0 |
| `depth_anything_3` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Meters | CC-BY-NC-4.0 |

Other Hugging Face depth-estimation models, such as ZoeDepth, DPT, and GLPN, also work with the `transformers` backend. Relative depth is defined only up to an unknown scale, and for inverse depth also an unknown shift, that can change from frame to frame.

These models assume perspective images. Equirectangular panoramas are not handled properly, so use perspective sources such as camera frames or the views of a spherical camera.

## Frame scheduling

The node keeps only the newest frame of each stream and visits the streams in turn, estimating at most one frame per stream in each round. Frames that arrive while the model is busy replace older waiting frames, so the output never lags behind by a queue. There is no fixed output rate: a stream is published at the rate of its source or at the rate the model can sustain for all streams together, whichever is lower.

## Zenoh interface

| Key | Interaction | Payload |
| --- | --- | --- |
| Each key in `streams` | Subscription | Image in any format Pillow can read, JPEG in practice |
| `{prefix}/{name}` | Publication | Depth map in NumPy NPY format (`application/x-npy`): a float32 array of shape `(height, width)` with the size of the source frame |

Each depth sample carries the attachment and the timestamp of the frame it was estimated from. A view of spherical-camera-zenoh-node therefore keeps its direction JSON, and a subscriber can match depth to image by timestamp when the source publishes one.

```python
import io

import numpy as np
import zenoh

with zenoh.open(zenoh.Config()) as session:
    for sample in session.declare_subscriber("depth/front"):
        depth = np.load(io.BytesIO(sample.payload.to_bytes()), allow_pickle=False)
        print(depth.shape, depth.dtype, depth.min(), depth.max())
```

## Failures

- An invalid configuration, a model that cannot be downloaded or loaded, and an unusable device end the node at startup with an `[ERROR]` line and a nonzero status.
- A payload that cannot be decoded as an image is skipped with a `[WARNING]` line; the other streams continue.
- An error during inference, such as running out of GPU memory, ends the node with a nonzero status. Use a process supervisor when automatic restart is needed.
- While no source publishes, the node waits and publishes nothing.

## Viewer

```console
$ uv run --group example examples/viewer.py depth/front
```

The viewer colors each frame between its own smallest and largest finite value, from dark for small values to bright for large values. The window title shows the key, the minimum and maximum of the frame, and the value under the mouse cursor. Press `Q` or `Esc`, or close the window, to quit. `examples/viewer-zenoh-config.json5` connects to `tcp/127.0.0.1:7448`; pass `--zenoh-config` to reach another host.
