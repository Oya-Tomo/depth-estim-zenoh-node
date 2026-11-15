# 使い方

## ノードの起動

```console
$ uv run node.py
```

ノードは最初にモデルを読み込み（初回起動時はダウンロードも行います）、その後でZenoh sessionを開きます。`Ctrl-C`で停止します。

## ノード設定

完全な例は[`config/node-config.example.json5`](../../config/node-config.example.json5)にあります。未知、欠落、重複、型が不正なfieldは起動時に拒否されます。

| キー | 説明 |
| --- | --- |
| `zenoh_key_prefix` | すべての深度streamをpublishするconcreteなZenoh key |
| `estimator.backend` | `transformers`または`depth_anything_3`。1つのprocessが読み込むモデルは1つです |
| `estimator.model_id` | Hugging Faceのmodel ID、またはローカルdirectory |
| `estimator.revision` | モデルのbranch、tag、commit hash。省略時は`main` |
| `estimator.device` | PyTorchのdevice。`cuda`、`cuda:1`、`cpu`など |
| `estimator.dtype` | `transformers`のみ。`float32`、`float16`、`bfloat16`。省略時は`float32` |
| `estimator.process_res` | `depth_anything_3`のみ。処理する画像の長辺のpixel数。省略時は`504` |
| `streams` | stream名から、深度を推定する画像streamのconcreteなZenoh keyへの対応。1つ以上必要です |
| `publisher.congestion_control` | `drop`、`block`、`block_first` |
| `publisher.reliability` | `best_effort`または`reliable` |

stream名は単一のkey segmentです。深度は`{zenoh_key_prefix}/{name}`にpublishされるので、exampleでは`camera/front`の深度が`depth/front`に出ます。入力keyは重複できません。また、深度のkeyが入力keyと同じになる設定も拒否されます。

CUDAで`transformers` backendを使う場合、`float16`にするとGPU memoryの使用量が減り、多くの場合は推論も速くなります。`depth_anything_3` backendに`dtype`がないのは、モデル自身がmixed precisionを選ぶためです。

## モデル

重みが公開されているモデルを使えます。publishされる値の意味はモデルによって異なります。

| `backend` | `model_id` | 出力 | ライセンス |
| --- | --- | --- | --- |
| `transformers` | `depth-anything/Depth-Anything-V2-Small-hf` | 相対的な逆深度（大きいほど近い） | Apache-2.0 |
| `transformers` | `depth-anything/Depth-Anything-V2-Base-hf`、`-Large-hf` | 相対的な逆深度（大きいほど近い） | CC-BY-NC-4.0 |
| `transformers` | `depth-anything/Depth-Anything-V2-Metric-Indoor-{Small,Base,Large}-hf`、`-Metric-Outdoor-{Small,Base,Large}-hf` | メートル（scaleは学習時のカメラに合わせて学習されたもの） | model cardを参照 |
| `transformers` | `apple/DepthPro-hf` | メートル（画角もモデルが推定する。低速） | Apple AMLR（研究用途のみ） |
| `depth_anything_3` | `depth-anything/DA3-SMALL`、`DA3-BASE` | 相対深度（大きいほど遠い） | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3MONO-LARGE` | 相対深度（大きいほど遠い） | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3METRIC-LARGE` | canonical depth。メートル = 値 × focal_px / 300。focal_pxは元frameのpixel単位の焦点距離 | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3-LARGE-1.1`、`DA3-GIANT-1.1` | 相対深度（大きいほど遠い） | CC-BY-NC-4.0 |
| `depth_anything_3` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | メートル | CC-BY-NC-4.0 |

ZoeDepth、DPT、GLPNなど、Hugging Faceにあるほかの深度推定モデルも`transformers` backendで動きます。相対深度はscaleが不定で、逆深度ではshiftも不定です。これらはframeごとに変わることがあります。

どのモデルも透視投影の画像を前提にしています。equirectangularのパノラマは正しく扱えないため、カメラのframeや全天球カメラのviewなど、透視投影の画像を入力にしてください。

## frameの処理順

ノードは各streamの最新frameだけを保持し、streamを順番に回って、1巡につき各streamから最大1 frameを推定します。推定中に届いたframeは待機中の古いframeを置き換えるので、出力が溜まって遅れていくことはありません。出力rateは固定ではありません。各streamは、入力のrateと、全streamを合わせてモデルが処理できるrateのうち、低いほうでpublishされます。

## Zenoh interface

| キー | 種類 | Payload |
| --- | --- | --- |
| `streams`の各key | Subscription | Pillowが読める形式の画像。実際にはJPEG |
| `{prefix}/{name}` | Publication | NumPyのNPY形式（`application/x-npy`）の深度map。shapeが`(height, width)`のfloat32配列で、大きさは元frameと同じ |

各深度sampleには、推定に使ったframeのattachmentとtimestampが付きます。そのため、spherical-camera-zenoh-nodeのviewでは向きのJSONがそのまま残ります。入力側がtimestampを付けていれば、subscriberはtimestampで深度と画像を対応付けられます。

```python
import io

import numpy as np
import zenoh

with zenoh.open(zenoh.Config()) as session:
    for sample in session.declare_subscriber("depth/front"):
        depth = np.load(io.BytesIO(sample.payload.to_bytes()), allow_pickle=False)
        print(depth.shape, depth.dtype, depth.min(), depth.max())
```

## 失敗時の動作

- 設定が不正なとき、モデルをダウンロードまたは読み込みできないとき、deviceが使えないときは、起動時に`[ERROR]`を1行出力して非0 statusで終了します。
- 画像としてdecodeできないpayloadは、`[WARNING]`を出力して読み飛ばします。ほかのstreamの処理は続きます。
- GPU memory不足など推論中のerrorが起きると、ノードは非0 statusで終了します。自動restartが必要な場合はprocess supervisorを使ってください。
- 入力が何もpublishされていない間は、何もpublishせずに待ちます。

## Viewer

```console
$ uv run --group example examples/viewer.py depth/front
```

viewerはframeごとに、有限な値の最小から最大までを色に割り当てます。小さい値ほど暗く、大きい値ほど明るく表示されます。window titleにはkey、そのframeの最小値と最大値、mouse cursor位置の値が出ます。`Q`か`Esc`を押すか、windowを閉じると終了します。`examples/viewer-zenoh-config.json5`は`tcp/127.0.0.1:7448`へ接続します。別hostへ接続する場合は`--zenoh-config`を指定してください。
