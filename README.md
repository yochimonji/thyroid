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
