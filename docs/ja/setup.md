# セットアップ

## 必要なもの

- Python 3.12と[uv](https://docs.astral.sh/uv/)
- CUDAが使えるNVIDIA GPUを推奨します。`device: "cpu"`でも動きますが低速です
- camera-zenoh-nodeやspherical-camera-zenoh-nodeなどがZenohにpublishするJPEG画像stream
- viewer exampleを使う場合はdesktop display

```console
$ uv sync
```

JPEGのdecodeとencodeには[TorchCodec](https://github.com/meta-pytorch/torchcodec)を使います。CUDA deviceではnvJPEGで処理されます。

## Depth Anything 3 backend

`depth_anything_3` backendには公式の`depth-anything-3` packageが必要です。このpackageは別のdependency groupに分けてあります。

```console
$ uv sync --group depth-anything-3
```

groupを任意にしているのは、このpackageが`transformers` backendには不要な重い依存（open3d、xformers、pycolmapなど）を持ち込むためです。groupを付けずに`uv sync`を実行すると、groupのpackageは削除されます。このpackageは`numpy<2`を要求します。lockfileは全groupをまとめて解決するため、project全体がNumPy 1.xとPython 3.12で動きます。groupには`addict`も入れてあります。`depth-anything-3`がimportしているのに、依存として宣言していないためです。

起動時に、xformersのmemory-efficient attentionが使えないという警告、3DGSのrenderingにはgsplatが必要だという通知、OmegaConfの非推奨警告が出ることがあります。いずれも深度推定には影響しません。

## モデルとHugging Faceのcache

ノードは設定されたモデルを初回起動時にダウンロードし、以後はHugging Faceのcache（既定では`~/.cache/huggingface`、または`HF_HOME`）から読み込みます。

| 環境変数 | 用途 |
| --- | --- |
| `HF_TOKEN` | gatedまたはprivateなモデルにアクセスするためのtoken |
| `HF_HUB_OFFLINE=1` | cacheだけを使い、Hubへ接続しない |

`estimator.model_id`には、選んだbackendの形式で保存したローカルdirectoryも指定できます。

## 設定ファイル

```console
$ cp config/zenoh-config.example.json5 config/zenoh-config.json5
$ cp config/node-config.example.json5 config/node-config.json5
```

| ファイル | 用途 |
| --- | --- |
| `config/zenoh-config.json5` | Zenohのmode、endpoint、scouting |
| `config/node-config.json5` | 深度モデル、画像stream、カラーマップ画像、publisher QoS |

どちらの実行時ファイルもGitの管理対象外です。ノードはcurrent working directoryから読み込むため、リポジトリrootで起動するか、`--zenoh-config`と`--node-config`を指定してください。ノード設定の各キーは[使い方](usage.md#ノード設定)で説明しています。

## Zenohの接続

exampleはTCP port 7448でlistenし、multicast scoutingを使うpeerとして動作します。portをカメラノードの7447と変えてあるので、同じhostで両方を動かせます。multicast scoutingが使えるnetworkでは、同じnetwork上のカメラノードを自動で見つけます。使えない場合は、`connect.endpoints`にカメラノードを指定します（例: `tcp/192.168.1.20:7447`）。既存のrouterへ接続する場合は`mode: "client"`にし、`connect.endpoints`にrouterを指定します。詳しくは[Zenoh deployment guide](https://zenoh.io/docs/getting-started/deployment/)を参照してください。

## 開発時の検証

```console
$ uv sync --all-groups
$ uv run ruff format --check .
$ uv run ruff check .
$ uv run pyright
$ uv run python -m unittest discover -s tests -v
```

testはモデルをダウンロードしません。Depth Anything 3のtestは`depth-anything-3` groupを入れたときだけ、CUDAのtestはGPUがあるときだけ実行されます。
