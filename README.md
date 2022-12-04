# Thyroid

## 各プログラム実行コマンド例

train

```sh
python run_model.py -p <params_path>
```

test

```sh
python predict.py -p <params_path> -d <dataroot_dir_path>
```

GradCAMで可視化  
params_pathのjsonファイルにはパラメータtestとtest_nameが必要

```sh
python run_gradcam.py -p <params_path>
```
