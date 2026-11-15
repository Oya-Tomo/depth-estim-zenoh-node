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
| `zenoh_key_prefix` | すべての出力をpublishするconcreteなZenoh key |
| `estimator.backend` | `transformers`または`depth_anything_3`。1つのprocessが読み込むモデルは1つです |
| `estimator.model_id` | Hugging Faceのmodel ID、またはローカルdirectory |
| `estimator.revision` | モデルのbranch、tag、commit hash。省略時は`main` |
| `estimator.device` | frameを処理するPyTorchのdevice。`cuda`、`cuda:1`、`cpu`など |
| `estimator.dtype` | `transformers`のみ。`float32`、`float16`、`bfloat16`。省略時は`float32` |
| `estimator.process_res` | `depth_anything_3`のみ。モデルが処理する画像の長辺のpixel数。省略時は`504` |
| `streams` | stream名から、深度を推定するJPEG streamのconcreteなZenoh keyへの対応。1つ以上必要です |
| `image.colormap` | カラーマップ画像に使うMatplotlibのカラーマップ名。`turbo`、`viridis`、`magma`、`Spectral`など。末尾に`_r`を付けると反転します |
| `image.value_range` | カラーマップの両端に対応する深度の値`[low, high]`。`null`ではframeごとの最小値と最大値を使います。省略時は`null` |
| `image.jpeg_quality` | カラーマップ画像のJPEG品質。1〜100 |
| `publisher.congestion_control` | `drop`、`block`、`block_first` |
| `publisher.reliability` | `best_effort`または`reliable` |

stream名は単一のkey segmentです。各streamは`{zenoh_key_prefix}/{name}`と`{zenoh_key_prefix}/{name}/raw`にpublishされるので、exampleでは`camera/front`の深度が`depth/front`と`depth/front/raw`に出ます。入力keyは重複できません。また、出力keyが入力keyと同じになる設定も拒否されます。

CUDAで`transformers` backendを使う場合、`float16`にするとGPU memoryの使用量が減り、多くの場合は推論も速くなります。`depth_anything_3` backendに`dtype`がないのは、モデル自身がmixed precisionを選ぶためです。

## カラーマップ画像

深度の値は、範囲の両端の間で線形にカラーマップへ割り当てられます。範囲外の値は近いほうの端の色になります。NaNと負の無限大は下端、正の無限大は上端の色です。

`value_range: null`では、frameごとに有限な値の最小から最大までをカラーマップ全体に広げます。どのframeでもカラーマップを使い切れますが、同じ深度でもframeによって色が変わります。範囲を固定すると色をframe間で比べられるので、メートルを出力するモデルに向いています（屋内なら`[0, 5]`など）。この設定はraw出力には影響しません。

## モデル

重みが公開されているモデルを使えます。raw出力の値の意味はモデルによって異なります。

| `backend` | `model_id` | rawの値 | ライセンス |
| --- | --- | --- | --- |
| `transformers` | `depth-anything/Depth-Anything-V2-Small-hf` | 相対的な逆深度（大きいほど近い） | Apache-2.0 |
| `transformers` | `depth-anything/Depth-Anything-V2-Base-hf`、`-Large-hf` | 相対的な逆深度（大きいほど近い） | CC-BY-NC-4.0 |
| `transformers` | `depth-anything/Depth-Anything-V2-Metric-Indoor-{Small,Base,Large}-hf`、`-Metric-Outdoor-{Small,Base,Large}-hf` | メートル（scaleは学習データのカメラに合わせて学習されたもの） | model cardを参照 |
| `transformers` | `apple/DepthPro-hf` | メートル（画角もモデルが推定する。低速） | Apple AMLR（研究用途のみ） |
| `depth_anything_3` | `depth-anything/DA3-SMALL`、`DA3-BASE` | 相対深度（大きいほど遠い） | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3MONO-LARGE` | 相対深度（大きいほど遠い） | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3METRIC-LARGE` | canonical depth。メートル = 値 × focal_px / 300。focal_pxは元frameのpixel単位の焦点距離 | Apache-2.0 |
| `depth_anything_3` | `depth-anything/DA3-LARGE-1.1`、`DA3-GIANT-1.1` | 相対深度（大きいほど遠い） | CC-BY-NC-4.0 |
| `depth_anything_3` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | メートル | CC-BY-NC-4.0 |

ZoeDepth、DPT、GLPNなど、Hugging Faceにあるほかの深度推定モデルも`transformers` backendで動きます。相対深度はscaleが不定で、逆深度ではshiftも不定です。これらはframeごとに変わることがあります。

どのモデルも透視投影の画像を前提にしています。equirectangularのパノラマは正しく扱えないため、カメラのframeや全天球カメラのviewなど、透視投影の画像を入力にしてください。

## 入力の形式と処理device

入力はJPEGだけを受け付けます。PNGなどほかの形式のpayload、空のpayload、壊れたJPEGは、`[WARNING]`を出力して読み飛ばします。CUDAでは、CMYKのJPEGもdecodeできないため読み飛ばします。逆に、途中で切れたJPEGは、欠けた部分が埋められてdecodeされることがあります。

frameの処理は、JPEGのdecode（CUDAではnvJPEG）、深度推定、カラーマップの適用、JPEGのencodeまで、すべて`estimator.device`上で行います。GPUを使う場合、CPUが扱うのは圧縮された画像と、raw配列にするための深度のcopy 1回だけです。

## frameの処理順

ノードは各streamの最新frameだけを保持し、streamを順番に回って、1巡につき各streamから最大1 frameを推定します。推定中に届いたframeは待機中の古いframeを置き換えるので、出力が溜まって遅れていくことはありません。出力rateは固定ではありません。各streamは、入力のrateと、全streamを合わせてモデルが処理できるrateのうち、低いほうでpublishされます。

目安として、RTX 4050 Laptop GPUで640x480のframe 1枚にかかる時間は、`Depth-Anything-V2-Metric-Indoor-Small-hf`（`float16`）で合計約17 ms、`DA3-SMALL`で約25 msです。入力がこれより速いときに2 streamを処理すると、各streamのrateは前者で約29 Hz、後者で約19 Hzになります。

## Zenoh interface

| キー | 種類 | Payload |
| --- | --- | --- |
| `streams`の各key | Subscription | JPEG画像（`image/jpeg`） |
| `{prefix}/{name}` | Publication | `image.colormap`で色を付けた深度のJPEG画像（`image/jpeg`） |
| `{prefix}/{name}/raw` | Publication | NumPyのNPY形式（`application/x-npy`）の深度。shapeが`(height, width)`のfloat32配列 |

どちらの出力も元frameと同じ大きさで、推定に使ったframeのattachmentとtimestampが付きます。そのため、spherical-camera-zenoh-nodeのviewでは向きのJSONがそのまま残ります。入力側がtimestampを付けていれば、subscriberはtimestampで深度と元画像を、またカラーマップ画像とraw配列を対応付けられます。

```python
import io

import numpy as np
import zenoh

with zenoh.open(zenoh.Config()) as session:
    for sample in session.declare_subscriber("depth/front/raw"):
        depth = np.load(io.BytesIO(sample.payload.to_bytes()), allow_pickle=False)
        print(depth.shape, depth.dtype, depth.min(), depth.max())
```

## 失敗時の動作

- 設定が不正なとき、モデルをダウンロードまたは読み込みできないとき、deviceが使えないときは、起動時に`[ERROR]`を出力して非0 statusで終了します。
- JPEG画像としてdecodeできないpayloadは、`[WARNING]`を出力して読み飛ばします。ほかのstreamの処理は続きます。
- 推定中にerrorが起きたとき、またはどこかの処理でGPU memoryが不足したときは、ノードは非0 statusで終了します。自動restartが必要な場合はprocess supervisorを使ってください。
- 入力が何もpublishされていない間は、何もpublishせずに待ちます。

## Viewer

```console
$ uv run --group example examples/viewer.py depth/front
```

viewerは、カラーマップ付きの深度画像やカメラの画像など、任意のJPEG streamを表示します。windowはframeの大きさに合わせて変わり、titleにはkeyが出ます。decodeできないpayloadは`[WARNING]`を出力して読み飛ばします。`Q`か`Esc`を押すか、windowを閉じると終了します。`examples/viewer-zenoh-config.json5`は`tcp/127.0.0.1:7448`へ接続します。別hostへ接続する場合は`--zenoh-config`を指定してください。
