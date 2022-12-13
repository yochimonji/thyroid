# Thyroid

## 各プログラム実行コマンド例

オプションパラメータの詳細は[parse.py](utils/parse.py)

### 訓練

```sh
python run_model.py -n <train_name> -A <trainA_path>
```

### テスト

```sh
python predict.py -p result/<train_name>/params.json -d <dataroot> -t <test_name>
```

### GradCAMで可視化  

テスト実行後にGradCAMで可視化する。params_pathのjsonファイルにはパラメータtestとtest_nameが必要。パラメータtestとtest_nameはpredict.pyを実行した際に生成されるparams.jsonに追加される。

```sh
python run_gradcam.py -p result/<train_name>/<test_name>/params.json
```

## resultフォルダ説明

`result`以下のフォルダ・ファイルについて説明する。

result以下のフォルダはrun_model.py, predict.py, run_gradcam.pyを実行することで生成される。例えば以下のオプションでrun_model.py, predict.py, run_gradcam.pyを実行した場合

```bash
python run_model.py -n <train_name> -A <trainA_path>
python predict.py -p result/<train_name>/params.json -t <test_name> -d <test_path>
python run_gradcam.py -p result/<train_name>/<test_name>/params.json
```

`result/<train_name>/<test_name>/`のように訓練条件ごと、テスト条件ごとにフォルダが生成される。

各階層で生成されるフォルダ・ファイルについて説明する。

- `result/<train_name>/weight/`：訓練時の重みが保存されるフォルダ。
- `result/<train_name>/params.json`：訓練時のパラメータを保存したファイル。
- `result/<train_name>/<test_name>/params.json`：テスト時のパラメータを保存したファイル。
- `result/<train_name>/<test_name>/confusion_matrix.csv`：テストデータの混同行列のファイル。
- `result/<train_name>/<test_name>/path_real_pred.csv`：テストデータのパス、本物のラベル、予測したラベルのファイル。
- `result/<train_name>/<test_name>/score_soft_voting.csv`：weightごとの予測結果をSoft Votingした際のスコアのファイル。
- `result/<train_name>/<test_name>/score.csv`：weightごとの予測結果の平均と標準偏差のスコアのファイル。
- `result/<train_name>/<test_name>/y_preds_all_score.csv`：weightごとの予測結果をすべて保存した結果とスコアのファイル。
- `result/<train_name>/<test_name>/gradcam/`：テストデータのGradCAMの可視化結果を保存したフォルダ。現在はテストで使用したすべての画像を可視化している。

