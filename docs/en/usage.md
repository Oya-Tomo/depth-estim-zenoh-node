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
| `zenoh_key_prefix` | Concrete Zenoh key under which every output is published |
| `estimator.backend` | `transformers` or `depth_anything_3`; one model is loaded per process |
| `estimator.model_id` | Hugging Face model ID or a local directory |
| `estimator.revision` | Branch, tag, or commit hash of the model; defaults to `main` |
| `estimator.device` | PyTorch device, such as `cuda`, `cuda:1`, or `cpu`, on which every frame is processed |
| `estimator.dtype` | `transformers` only: `float32`, `float16`, or `bfloat16`; defaults to `float32` |
| `estimator.process_res` | `depth_anything_3` only: longer side of the image the model processes, in pixels; defaults to `504` |
| `streams` | Map from a stream name to the concrete Zenoh key of the JPEG stream to estimate depth from; at least one entry |
| `image.colormap` | Matplotlib colormap name of the colorized image, such as `turbo`, `viridis`, `magma`, or `Spectral`; the suffix `_r` reverses it |
| `image.value_range` | `[low, high]`: depth values at the two ends of the colormap; `null` uses the minimum and maximum of each frame; defaults to `null` |
| `image.jpeg_quality` | JPEG quality of the colorized image, 1–100 |
| `publisher.congestion_control` | `drop`, `block`, or `block_first` |
| `publisher.reliability` | `best_effort` or `reliable` |

A stream name is a single key segment. The stream is published on `{zenoh_key_prefix}/{name}` and `{zenoh_key_prefix}/{name}/raw`, so the example publishes the depth of `camera/front` on `depth/front` and `depth/front/raw`. Source keys must be unique, and no output key may equal a source key.

On CUDA, `float16` reduces the GPU memory and usually the inference time of the `transformers` backend. The `depth_anything_3` backend has no `dtype` because the model selects mixed precision by itself.

## Colorized image

Depth values are mapped linearly onto the colormap between the two ends of the range. Values outside the range take the color of the nearer end; NaN and negative infinity take the low end, and positive infinity the high end.

With `value_range: null`, each frame is stretched between its own smallest and largest finite value, so every frame uses the whole colormap, but the same depth can get different colors from frame to frame. A fixed range keeps the colors comparable and suits models that output meters, such as `[0, 5]` indoors. The raw output is not affected by these settings.

## Models

The node works with models whose weights are public. The meaning of the raw values depends on the model:

| `backend` | `model_id` | Raw values | License |
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

## Input format and device

The node accepts JPEG frames only. A payload in another format such as PNG, an empty payload, or a damaged JPEG is skipped with a `[WARNING]` line. On CUDA, CMYK JPEG cannot be decoded and is skipped as well, while a JPEG that was cut short may still be decoded with the missing part filled in.

Every step of the frame path runs on `estimator.device`: JPEG decoding (nvJPEG on CUDA), depth estimation, colorization, and JPEG encoding. With a GPU, the CPU handles only the compressed images and the one copy of the depth that becomes the raw array.

## Frame scheduling

The node keeps only the newest frame of each stream and visits the streams in turn, estimating at most one frame per stream in each round. Frames that arrive while the model is busy replace older waiting frames, so the output never lags behind by a queue. There is no fixed output rate: a stream is published at the rate of its source or at the rate the model can sustain for all streams together, whichever is lower.

For reference, a 640x480 frame takes about 17 ms in total with `Depth-Anything-V2-Metric-Indoor-Small-hf` (`float16`) and about 25 ms with `DA3-SMALL` on an RTX 4050 Laptop GPU. Two streams from faster sources are therefore each published at about 29 Hz and 19 Hz, respectively.

## Zenoh interface

| Key | Interaction | Payload |
| --- | --- | --- |
| Each key in `streams` | Subscription | JPEG image (`image/jpeg`) |
| `{prefix}/{name}` | Publication | Depth colorized with `image.colormap` as a JPEG image (`image/jpeg`) |
| `{prefix}/{name}/raw` | Publication | Depth in NumPy NPY format (`application/x-npy`): a float32 array of shape `(height, width)` |

Both outputs have the size of the source frame and carry the attachment and the timestamp of the frame they were estimated from. A view of spherical-camera-zenoh-node therefore keeps its direction JSON, and a subscriber can match depth to image, or the colorized image to the raw array, by timestamp when the source publishes one.

```python
import io

import numpy as np
import zenoh

with zenoh.open(zenoh.Config()) as session:
    for sample in session.declare_subscriber("depth/front/raw"):
        depth = np.load(io.BytesIO(sample.payload.to_bytes()), allow_pickle=False)
        print(depth.shape, depth.dtype, depth.min(), depth.max())
```

## Failures

- An invalid configuration, a model that cannot be downloaded or loaded, and an unusable device end the node at startup with an `[ERROR]` line and a nonzero status.
- A payload that cannot be decoded as a JPEG image is skipped with a `[WARNING]` line; the other streams continue.
- An error during estimation, or running out of GPU memory at any step, ends the node with a nonzero status. Use a process supervisor when automatic restart is needed.
- While no source publishes, the node waits and publishes nothing.

## Viewer

```console
$ uv run --group example examples/viewer.py depth/front
```

The viewer displays any JPEG stream, such as the colorized depth or a camera stream. The window takes the size of the frames and shows the key in its title; a payload that cannot be decoded is skipped with a `[WARNING]` line. Press `Q` or `Esc`, or close the window, to quit. `examples/viewer-zenoh-config.json5` connects to `tcp/127.0.0.1:7448`; pass `--zenoh-config` to reach another host.
