# depth-estim-zenoh-node

[English](README.md)

画像streamから深度を推定するZenohノードです。camera-zenoh-nodeやspherical-camera-zenoh-nodeがpublishするJPEGなどの画像frameをsubscribeし、Hugging Faceのmodel IDで選んだ深度モデルで推定します。深度mapはfloat32配列としてpublishし、元frameのattachmentとtimestampをそのまま付けます。

## クイックスタート

```console
$ uv sync
$ cp config/zenoh-config.example.json5 config/zenoh-config.json5
$ cp config/node-config.example.json5 config/node-config.json5
```

`depth_anything_3` backendを使う場合は、`uv sync`の代わりに専用のdependency groupを入れます。

```console
$ uv sync --group depth-anything-3
```

`config/node-config.json5`の`estimator`と`streams`を設定し、リポジトリrootからノードを起動します。モデルは初回起動時にHugging Faceのcacheへダウンロードされます。

```console
$ uv run node.py
```

深度streamを表示します。

```console
$ uv run --group example examples/viewer.py depth/front
```

## ドキュメント

- [セットアップ](docs/ja/setup.md)
- [使い方とZenoh interface](docs/ja/usage.md)
