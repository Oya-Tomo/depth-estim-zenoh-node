# depth-estim-zenoh-node

[English](README.md)

JPEG画像streamから深度を推定するZenohノードです。camera-zenoh-nodeやspherical-camera-zenoh-nodeがpublishするframeをsubscribeし、Hugging Faceのmodel IDで選んだ深度モデルで推定します。推定した深度は、見るためのカラーマップ付きJPEG画像と、計算に使うfloat32配列の2通りでpublishします。どちらも元frameと同じ大きさで、元frameのattachmentとtimestampが付きます。GPUを使う場合、frameはJPEGのdecodeからencodeまでGPU上で処理されます。

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

カラーマップ付きの深度画像を表示します。

```console
$ uv run --group example examples/viewer.py depth/front
```

深度の値そのものは`depth/front/raw`にpublishされます。読み方は[使い方](docs/ja/usage.md#zenoh-interface)を参照してください。

## ドキュメント

- [セットアップ](docs/ja/setup.md)
- [使い方とZenoh interface](docs/ja/usage.md)
