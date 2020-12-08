# 標準ライブラリ
import random
import json
import sys
import os
from datetime import datetime

# 外部ライブラリ
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 自作ライブラリ
from utils.utils import ImageTransform, make_datapath_list, show_wrong_img
from utils.dataset import ArrangeNumDataset, ConcatDataset
from model.model import CustomResNet, CustomEfficientNet, eval_net, train_net

# 乱数シード値を固定して再現性を確保
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# jsonファイルを読み込んでパラメータを設定する
# jsonから読み込むことでpyファイルの書き換えをしなくてよいのでGitが汚れない
f = open("./config/params.json", "r")
params = json.load(f)
f.close()
data_path = params["data_path"]
labels = params["labels"]
dataset_params = params["dataset_params"]
tissue_dataset_params = params["tissue_dataset_params"]
num_estimate = params["num_estimate"]
batch_size = params["batch_size"]
epochs = params["epochs"]
net_params = params["net_params"]
loss_weight_flag = params["loss_weight_flag"]
optim_params = params["optim_params"]

# GPUが使用可能ならGPU、不可能ならCPUを使う
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

# 訓練とテストのファイルリストを取得する
train_list = make_datapath_list(data_path+"train")
test_list = make_datapath_list(data_path+"test")

# 訓練とテストのデータセットを作成する
train_dataset = ArrangeNumDataset(train_list, 
                                  labels,
                                  phase="train",
                                  transform=ImageTransform(mean=dataset_params["train_mean"],
                                                           std=dataset_params["train_std"],
                                                           grayscale_flag=dataset_params["grayscale_flag"],
                                                           normalize_per_img=dataset_params["normalize_per_img"]), 
                                  arrange=dataset_params["arrange"])
test_dataset = ArrangeNumDataset(test_list, 
                                 labels,
                                 phase="val",
                                 transform=ImageTransform(mean=dataset_params["test_mean"],
                                                          std=dataset_params["test_std"],
                                                          grayscale_flag=dataset_params["grayscale_flag"],
                                                          normalize_per_img=dataset_params["normalize_per_img"]),
                                 arrange=dataset_params["arrange"])

if tissue_dataset_params["use"]:
    tissue_list = make_datapath_list(data_path+"tissue array")
    tissue_dataset = ArrangeNumDataset(tissue_list,
                                       labels,
                                       phase=tissue_dataset_params["phase"],
                                       transform=ImageTransform(mean=tissue_dataset_params["mean"],
                                                                std=tissue_dataset_params["std"],
                                                                grayscale_flag=dataset_params["grayscale_flag"],
                                                                normalize_per_img=dataset_params["normalize_per_img"]),
                                       arrange=dataset_params["arrange"])
    if tissue_dataset_params["phase"] == "train":
        train_dataset = ConcatDataset(train_dataset, tissue_dataset)
    elif tissue_dataset_params["phase"] == "val":
        test_dataset = ConcatDataset(test_dataset, tissue_dataset)
    else:
        print("ValueError:tissue_dataset_params['phase']=={}は正しくありません。".format(tissue_dataset_params["phase"]))
        sys.exit()

print("len(train_dataset)", len(train_dataset))
print("len(test_dataset)", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2)

eval_accs = []  # estimateごとの推論時の正答率リスト
net_weights = []  # estimateごとのネットワークの重みリスト

for i in range(num_estimate):
    print("学習・推論：{}/{}".format(i+1, num_estimate))
    # 使用するネットワークを設定する
    if "resnet" in net_params["name"]:
        net = CustomResNet(only_fc=net_params["only_fc"],
                        pretrained=net_params["pretrained"],
                        model_name=net_params["name"])
    elif "efficientnet" in net_params["name"]:
        net = CustomEfficientNet(only_fc=net_params["only_fc"],
                            pretrained=net_params["pretrained"],
                            model_name=net_params["name"])
    else:  # ネットワーク名が間違っていたらエラー
        print("net_params['name']=={} : 定義されていないnameです".format(net_params['name']))
        sys.exit()

    # 損失関数のクラス数に合わせてweightをかけるか決める
    if loss_weight_flag:
        loss_weights = torch.tensor(train_dataset.weights).float().to(device)  # deviceに送らないと動かない
    else:
        loss_weights = None
    loss_fn=nn.CrossEntropyLoss(weight=loss_weights)
    print("loss_fn.weight:", loss_fn.weight)

    # 使用する最適化手法を設定する
    if "adam" == optim_params["name"]:
        optimizer = optim.Adam(net.get_params_lr(lr_fc=optim_params["lr_fc"], lr_not_fc=optim_params["lr_not_fc"]),
                            weight_decay=optim_params["weight_decay"])
    elif "sgd" == optim_params["name"]:
        optimizer = optim.SGD(net.get_params_lr(lr_fc=optim_params["lr_fc"], lr_not_fc=optim_params["lr_not_fc"]),
                            momentum=optim_params["momentum"],
                            weight_decay=optim_params["weight_decay"])
    else:  # 最適化手法の名前が間違えていたらエラー
        print("optim_params['name']=={} : 定義されていないnameです".format(optim_params['name']))
        sys.exit()
    print("optimizer:", optimizer)

    # 学習
    train_net(net(), train_loader, test_loader, optimizer=optimizer,
            loss_fn=loss_fn, epochs=epochs, device=device)
    # 推論
    ys, ypreds = eval_net(net(), test_loader, device=device)

    # 正答率とネットワークの重みをリストに追加
    ys = ys.cpu().numpy()
    ypreds = ypreds.cpu().numpy()
    eval_accs.append(accuracy_score(ys, ypreds))
    net_weights.append(net().state_dict())

# eval_accの中央値のインデックスを求める
acc_median = np.median(eval_accs)
acc_median_index = np.argmin(np.abs(np.array(eval_accs) - acc_median))
print("eval_accs:", eval_accs)
print("acc_median:", acc_median)
print("acc_median_index:", acc_median_index)

# 推論結果表示
net_weight_median = net_weights[acc_median_index]
net().load_state_dict(net_weight_median)
ys, ypreds = eval_net(net(), test_loader, device=device)
ys = ys.cpu().numpy()
ypreds = ypreds.cpu().numpy()
print(confusion_matrix(ys, ypreds))
print(classification_report(ys, ypreds,
                            target_names=labels,
                            digits=3))

# ネットワークとjsonのパラメータを保存
dt_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
torch.save(net_weight_median, "weight/weight_{}_{}.pth".format(dt_now, int(acc_median*100)))
f = open("config/params_{}_{}.json".format(dt_now, int(acc_median*100)), "w")
json.dump(params, f)
f.close()