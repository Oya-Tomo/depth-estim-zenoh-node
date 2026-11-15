# depth-estim-zenoh-node

[日本語](README_ja.md)

A Zenoh node that estimates depth from JPEG image streams. It subscribes to frames such as those published by camera-zenoh-node and spherical-camera-zenoh-node, runs a depth model selected by its Hugging Face model ID, and publishes the depth of each frame twice: colorized as a JPEG image to look at, and as a float32 array to compute with. Both outputs have the size of the source frame and carry its attachment and timestamp. On a GPU, every frame stays on the device from JPEG decoding to JPEG encoding.

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

Display the colorized depth:

```console
$ uv run --group example examples/viewer.py depth/front
```

The depth values themselves are published on `depth/front/raw`; [Usage](docs/en/usage.md#zenoh-interface) shows how to read them.

## Documentation

- [Setup](docs/en/setup.md)
- [Usage and Zenoh interface](docs/en/usage.md)
