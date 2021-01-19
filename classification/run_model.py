# 標準ライブラリ
import random
import json

# 外部ライブラリ
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, classification_report

# 自作ライブラリ
import utils
from utils import ImageTransform
from utils.dataset import ArrangeNumDataset
from model import create_net, eval_net, train_net

# 乱数シード値を固定して再現性を確保
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# jsonファイルを読み込んでパラメータを設定する
# よく呼び出すパラメータを変数に代入
params = utils.load_params()
data_path = params["data_path"]
dataset_params = params["dataset_params"]
tissue_dataset_params = params["tissue_dataset_params"]
optim_params = params["optim_params"]

# GPUが使用可能ならGPU、不可能ならCPUを使う
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print("使用デバイス：", device)

ys = []
ypreds = []
eval_recalls = []  # estimateごとのrecallのリスト
net_weights = []  # estimateごとのネットワークの重みリスト

for i in range(params["num_estimate"]):
    print("\n学習・推論：{}/{}".format(i+1, params["num_estimate"]))

    # 訓練とテストのデータセットを作成する
    train_dataset = ArrangeNumDataset(params, "train",
                                    transform=ImageTransform(params=params,
                                                            mean=dataset_params["train_mean"],
                                                            std=dataset_params["train_std"]))
    test_dataset = ArrangeNumDataset(params, "test",
                                    transform=ImageTransform(params=params,
                                                            mean=dataset_params["test_mean"],
                                                            std=dataset_params["test_std"]))
    print("train_datasetの各クラスのデータ数： {}\t計：{}".format(train_dataset.data_num, train_dataset.data_num.sum()))
    print("test_datasetの各クラスのデータ数：  {}\t計：{}".format(test_dataset.data_num, test_dataset.data_num.sum()))

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"],
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"],
                            shuffle=False, num_workers=4)

    # 損失関数のクラス数に合わせてweightをかけるか決める
    if params["loss_weight_flag"]:
        loss_weight = train_dataset.weight.to(device)  # deviceに送らないと動かない
        print("loss_weight:", loss_weight.cpu())
    else:
        loss_weight = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)
    
    net = create_net(params)

    # 使用する最適化手法を設定する
    if "Adam" == optim_params["name"]:
        optimizer = optim.Adam(net.get_params_lr(lr_not_pretrained=optim_params["lr_not_pretrained"], lr_pretrained=optim_params["lr_pretrained"]),
                            weight_decay=optim_params["weight_decay"])
    elif "SGD" == optim_params["name"]:
        optimizer = optim.SGD(net.get_params_lr(lr_not_pretrained=optim_params["lr_not_pretrained"], lr_pretrained=optim_params["lr_pretrained"]),
                            momentum=optim_params["momentum"],
                            weight_decay=optim_params["weight_decay"])

    # 学習
    train_net(net, train_loader, test_loader, optimizer=optimizer,
            loss_fn=loss_fn, epochs=params["epochs"], device=device)
    # 推論
    y, ypred = eval_net(net, test_loader, device=device)

    # 正答率とネットワークの重みをリストに追加
    ys.append(y.cpu().numpy())
    ypreds.append(ypred.cpu().numpy())
    eval_recall = recall_score(ys[-1], ypreds[-1], average=None, zero_division=0)
    eval_recalls.append(eval_recall)
    print("テストの各クラスrecall：\n{}\n平均：{}".format(np.round(eval_recall*100, decimals=1), np.round(eval_recall.mean()*100, decimals=1)))
    net_weights.append(net.cpu().state_dict())

# weightを保存するために
# eval_recallsのmeanに最も近いインデックスを求める
recall_mean_all = np.mean(eval_recalls)
recall_means_per_estimate = np.mean(eval_recalls, axis=1)
recall_mean_index = np.argmin(np.abs(recall_means_per_estimate - recall_mean_all))
print("各感度の{}回平均".format(params["num_estimate"]))
print(params["labels"])
print(np.round(np.mean(eval_recalls, axis=0)*100, decimals=1))
print("各感度の{}回平均の平均：{}".format(params["num_estimate"], np.round(recall_mean_all*100, decimals=1)))
# param,weight保存、混合行列表示用のインデックス
print("↑に近い各感度の{}回平均のインデックス:".format(params["num_estimate"]), recall_mean_index)

# 推論結果表示
y = ys[recall_mean_index]
ypred = ypreds[recall_mean_index]
print(confusion_matrix(y, ypred))
print(classification_report(y, ypred,
                            target_names=params["labels"],
                            digits=3,
                            zero_division=0))

# 各種パラメータと結果の保存
utils.save_params(params, net_weights[recall_mean_index])
utils.save_result(ys, ypreds, params)