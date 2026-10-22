# Microduck ACT

CPU版MuJoCoで、Microduckがボールを探して接近・ドリブルする動作を学習するためのプロジェクトです。
キーボード操作によるデモ収集、LeRobot ACTの学習、階層型ポリシーの推論に対応しています。
ACTはカメラ画像とロボットの状態から速度指令を予測し、学習済みの歩行ONNXポリシーが
BAM XL330 M6を通じて14個のサーボを制御します。

## セットアップ

Linux、Python 3.12、Git、uvを使用します。ビューアにはデスクトップ環境が必要です。
カメラプレビューにはTkも必要です。Ubuntuなどでは通常、OSパッケージの`python3-tk`を使用します。
ACTの本格的な学習にはCUDA対応GPUを推奨します。CPUは小規模な動作確認向けです。

以下のコマンドは、このREADMEがあるディレクトリで実行してください。

```bash
uv sync --locked
uv sync --locked --project lerobot_env
uv run scripts/prepare_assets.py --accept-model-license
```

実行前に[LICENSE-MODELS.md](LICENSE-MODELS.md)を確認してください。
`--accept-model-license`はロボットモデルの利用条件への同意を示します。
準備処理は、Pollen Roboticsの次のリビジョンからロボットXMLと参照先の38個のメッシュだけを取得します。

`b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`

取得した資産はGit管理対象外の`src/`以下にキャッシュされ、公開ソースには同梱しません。
必要に応じてXMLに頭部カメラを追加し、メッシュ参照をローカルキャッシュの絶対パスへ変更します。
このディレクトリを移動した場合は準備コマンドを再実行してください。
初回の準備にはインターネット接続が必要ですが、通常のシミュレーション中には資産を取得しません。
BAMも、元環境で動作確認済みのコミットに固定しています。

## 学習済みポリシーの取得

歩行と右キックの学習済みONNXポリシーは、以下の公式配布先からダウンロードしてください。

[Pollen Robotics / microduck-policies（Hugging Face）](https://huggingface.co/pollen-robotics/microduck-policies/tree/main)

| ファイル | 用途 |
| --- | --- |
| `alpha_walking.onnx` | ACTが出力した速度指令に従って歩行する低レベル制御 |
| `ball_kick_right.onnx` | 右キック指令を受けたときの低レベル制御 |

以下のコマンドで、このREADMEと同じディレクトリへ保存できます。
同名ファイルがある場合は上書きするため、既存のポリシーを残したい場合は別の保存先を指定してください。

```bash
curl --fail --location \
  https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/alpha_walking.onnx \
  --output alpha_walking.onnx

curl --fail --location \
  https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/ball_kick_right.onnx \
  --output ball_kick_right.onnx
```

別の場所に保存した場合は、以降の`--walking`と`--kick-right`にそのパスを指定してください。
配布ページではApache-2.0と表示されています。利用時には配布元の最新のライセンスと説明も確認してください。
ロボットのSTLメッシュに適用されるCC BY-NC-SA 4.0とは別のライセンスです。

ONNXファイルと、このプロジェクトで学習するACTチェックポイントは同梱・自動取得しません。
公式ONNXは低レベル制御用であり、ボール探索用ACTの学習済みモデルではありません。
歩行ポリシーには、共有の61次元観測形式と組み込み済みの観測正規化が必要です。
旧51次元のポリシーで置き換えないでください。

## デモの収集

```bash
uv run scripts/infer_policy.py \
  --walking alpha_walking.onnx --kick-right ball_kick_right.onnx \
  --new-cmd-obs \
  --lerobot-dataset-root data/ball_keyboard \
  --lerobot-repo-id local/ball-keyboard
```

矢印キーの押し続け操作にはカメラプレビューを使用し、MuJoCoビューアは表示専用にしてください。
上下キーは前進・後退、左右キーは旋回です。左右移動ではありません。矢印キーを離すと停止指令になります。

| キー | 操作 |
| --- | --- |
| `X` | エピソードの録画開始 |
| `C` | エピソードを保存して録画終了 |
| `0` | ロボットをリセットし、ボールをカメラに映る位置へ配置 |
| `9` | ロボットをリセットし、ボールをカメラに映らない背後へ配置 |
| `L` | 右キックONNXポリシーを実行 |
| `Q` | 終了。録画中の未保存エピソードは破棄 |

録画中はリセットできません。互換性のある既存データセットを指定した場合は、上書きせず追記します。
探索デモを別に保存する場合は、保存先を`data/ball_search_keyboard`、
repo IDを`local/ball-search-keyboard`へ変更してください。

探索デモは`9`、`X`の順に押し、旋回して発見、接近、接触するところまで記録します。
一部のエピソードでは、接触後にボールを見失い、探索・再接近する流れも含めてください。
操作を考えるための長い停止は入れすぎないようにします。意図した停止は有効な教師データです。
体当たりのデモでは`kick_right`を常にゼロにできますが、現在の収集・推論CLIは
その場合でも右キックONNXの読み込みを必要とします。

LeRobot v3形式で、50 HzのRGB画像`(320, 240, 3)`、48次元のロボット状態、
4次元のアクション`[vx, vy, yaw_rate, kick_right]`を保存します。
旧14次元の関節アクションを使うACTデータセットやチェックポイントとは互換性がありません。

## データの結合と学習

元の2つのデータセットは残したまま、新しい出力先へ結合します。

```bash
uv run --project lerobot_env lerobot-edit-dataset \
  --new_repo_id=local/ball-combined \
  --new_root=data/ball_combined \
  --operation.type=merge \
  --operation.repo_ids='["local/ball-keyboard","local/ball-search-keyboard"]' \
  --operation.roots='["data/ball_keyboard","data/ball_search_keyboard"]' \
  --operation.concatenate_videos=false \
  --operation.concatenate_data=false \
  --push_to_hub=false

uv run --project lerobot_env lerobot-train \
  --dataset.repo_id=local/ball-combined \
  --dataset.root=data/ball_combined \
  --policy.type=act \
  --policy.chunk_size=20 --policy.n_action_steps=20 \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=100000 --save_freq=10000 \
  --output_dir=logs/ball_act --job_name=ball_act
```

結合データでファインチューニングする場合は、既存チェックポイントの重みを初期値にして
新しい学習を開始します。旧オプティマイザとステップ数を復元する再開とは異なり、
`--resume=true`は指定しません。

```bash
uv run --project lerobot_env lerobot-train \
  --dataset.repo_id=local/ball-combined \
  --dataset.root=data/ball_combined \
  --policy.path=logs/ball_act/checkpoints/last/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=30000 --save_freq=5000 \
  --output_dir=logs/ball_search_finetune --job_name=ball_search_finetune
```

新しい学習ごとに、未使用の出力ディレクトリを指定してください。
途中のチェックポイントも使い、見えるボールへの接近と、見えないボールの探索の両方を比較します。
上記のステップ数は初期の試行目安であり、学習の収束を保証する値ではありません。

## 推論

```bash
uv run scripts/infer_policy.py \
  --walking alpha_walking.onnx --kick-right ball_kick_right.onnx \
  --new-cmd-obs \
  --lerobot-policy logs/ball_search_finetune/checkpoints/last/pretrained_model \
  --lerobot-action-steps=1 --camera-preview
```

最初から学習したモデルを使う場合は、指定先を
`logs/ball_act/checkpoints/last/pretrained_model`へ変更してください。
保存された前処理・後処理を含むチェックポイントのディレクトリ全体を指定します。

ACTは制御ステップごとに指令を再予測します。関節の目標値を生成するのは引き続き歩行ONNXです。
`0`で見える位置からの接近、`9`で見えない位置からの探索を確認できます。
`achieved/cmd`の速度ログで、実速度と指令速度を切り分けられます。
デモを追加しても、ボールがないときに必ず停止する保証はありません。安全制御用の仕組みではありません。

## スモークテスト

ロボット資産の準備後に実行します。

```bash
MUJOCO_GL=egl uv run python -m unittest discover -s tests -v
```

独立構成のMuJoCo/BAMシーン、カメラ、ボールが見える・見えない配置、4次元ラベル形式を確認します。
学習済みポリシーのタスク成功率を検証するテストではありません。

## 派生元とライセンス

[Pollen Robotics Microduck RL](https://github.com/pollen-robotics/microduck_rl)を元にしています。
移植元はリビジョン`4960bb332acd81ff340e902b4da77960c9038374`にローカル変更を加えた作業環境です。
取得するロボット資産のリビジョンは、セットアップ節に記載したものへ別途固定しています。

- コードとMJCF記述はApache-2.0です。[LICENSE](LICENSE)と既存の帰属表示を保持してください。
- `scripts/infer_policy.py`は、キーボード指令ACT、正立カメラプレビュー、エピソード録画、探索用リセット、独立構成のシーン参照に対応するため変更しています。
- データセット記録処理とACTワーカーは、変更済みワークスペースで開発したものを移植し、4次元アクション形式と追記動作を維持しています。
- `scenes/scene_ball.xml`と`scenes/ball.xml`は、独立構成の参照先と、調整済みのボール接触・転がり摩擦設定に合わせた派生ファイルです。
- ダウンロードするSTLメッシュはCopyright Pollen Robotics、CC BY-NC-SA 4.0です。非商用・継承の条件を含みます。[LICENSE-MODELS.md](LICENSE-MODELS.md)と[ライセンス](https://creativecommons.org/licenses/by-nc-sa/4.0/)を確認してください。ダウンロード方式にしても制限はなくなりません。
- LeRobot 0.6.1はApache-2.0です。その他の依存ライブラリには、それぞれのライセンスが適用されます。
- 利用者が用意する重み、データセット、生成メディアには、それぞれに適用される権利・利用条件の確認が必要です。このプロジェクトのコードライセンスが、それらへの権利を与えるわけではありません。

モデル用ライセンス文書は、上流で定義された対象ファイルの範囲を維持しています。
ダウンロードしたSTLメッシュをApache-2.0として扱わないでください。
元のGit履歴、メッシュ、記録データ、重みは、このプロジェクトの公開対象に含めません。
特にGitではなくZIPなどで公開する場合は、Git管理対象外のファイルが混入していないか確認してください。