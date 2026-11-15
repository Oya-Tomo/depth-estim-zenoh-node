# depth-estim-zenoh-node

[日本語](README_ja.md)

A Zenoh node that estimates depth from image streams. It subscribes to image frames such as the JPEG streams published by camera-zenoh-node and spherical-camera-zenoh-node, runs a depth model selected by its Hugging Face model ID, and publishes each depth map as a float32 array together with the attachment and timestamp of its source frame.

## Quick start

```console
$ uv sync
$ cp config/zenoh-config.example.json5 config/zenoh-config.json5
$ cp config/node-config.example.json5 config/node-config.json5
```

To use the `depth_anything_3` backend, install its dependency group instead of the plain `uv sync`:

```console
$ uv sync --group depth-anything-3
```

Set `estimator` and `streams` in `config/node-config.json5`, then start the node from the repository root. The model is downloaded into the Hugging Face cache on the first start.

```console
$ uv run node.py
```

Display a depth stream:

```console
$ uv run --group example examples/viewer.py depth/front
```

## Documentation

- [Setup](docs/en/setup.md)
- [Usage and Zenoh interface](docs/en/usage.md)
