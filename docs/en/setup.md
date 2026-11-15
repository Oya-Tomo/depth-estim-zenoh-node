# Setup

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU with CUDA is recommended; `device: "cpu"` works but is slow
- Image streams over Zenoh, such as those of camera-zenoh-node or spherical-camera-zenoh-node
- A desktop display for the viewer example

```console
$ uv sync
```

## Depth Anything 3 backend

The `depth_anything_3` backend needs the official `depth-anything-3` package, which is kept in a separate dependency group:

```console
$ uv sync --group depth-anything-3
```

The group is optional because the package brings heavy dependencies (open3d, xformers, pycolmap, and others) that the `transformers` backend does not need. A plain `uv sync` removes the group again. The package requires `numpy<2`; because one lockfile covers every group, the whole project runs on NumPy 1.x and Python 3.12. The group also lists `addict`, which `depth-anything-3` 0.1.1 imports without declaring it.

xformers may warn at startup that memory-efficient attention is unavailable; Depth Anything 3 still runs.

## Models and the Hugging Face cache

The node downloads the configured model on the first start and reuses it from the Hugging Face cache (`~/.cache/huggingface` by default, or `HF_HOME`) afterwards.

| Environment variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Access token for gated or private models |
| `HF_HUB_OFFLINE=1` | Use only the cache and never contact the Hub |

`estimator.model_id` can also be a local directory in the format of the chosen backend.

## Configuration files

```console
$ cp config/zenoh-config.example.json5 config/zenoh-config.json5
$ cp config/node-config.example.json5 config/node-config.json5
```

| File | Purpose |
| --- | --- |
| `config/zenoh-config.json5` | Zenoh mode, endpoints, and scouting |
| `config/node-config.json5` | Depth model, image streams, and publisher QoS |

Both runtime files are ignored by Git. The node reads them from the current working directory, so start it from the repository root or pass `--zenoh-config` and `--node-config`. [Usage](usage.md#node-configuration) describes every key of the node configuration.

## Zenoh connection

The example runs as a peer that listens on TCP port 7448 and uses multicast scouting. The port differs from the camera nodes' 7447 so that both can run on one host. With multicast scouting the node finds the camera nodes on the same network by itself; where multicast is unavailable, list a camera node in `connect.endpoints`, such as `tcp/192.168.1.20:7447`. To join an existing router instead, use `mode: "client"` and list the router in `connect.endpoints`. See the [Zenoh deployment guide](https://zenoh.io/docs/getting-started/deployment/).

## Development checks

```console
$ uv sync --all-groups
$ uv run ruff format --check .
$ uv run ruff check .
$ uv run pyright
$ uv run python -m unittest discover -s tests -v
```
