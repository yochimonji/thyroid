# 標準ライブラリ
import random
import json

# 外部ライブラリ
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import recall_score

# 自作ライブラリ
import utils
from utils.dataset import ArrangeNumDataset
from model import create_net, eval_net, train_net

# 乱数シード値を固定して再現性を確保
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# jsonファイルを読み込んでパラメータを設定する
# よく呼び出すパラメータを変数に代入
params = utils.load_params()
optim_params = params["optim_params"]

# GPUが使用可能ならGPU、不可能ならCPUを使う
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

ys = []
ypreds = []
net_weights = []  # estimateごとのネットワークの重みリスト

for i in range(params["num_estimate"]):
    print("\n学習・推論：{}/{}".format(i+1, params["num_estimate"]))

    # 訓練とテストのデータセットを作成する
    train_dataset = ArrangeNumDataset(params=params, phase="train")
    test_dataset = ArrangeNumDataset(params=params, phase="test")

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"],
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"],
                            shuffle=False, num_workers=4)

    # 損失関数のクラス数に合わせてweightをかけるか決める
    if params["imbalance"] == "loss_weight":
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
    recall = recall_score(ys[-1], ypreds[-1], average=None, zero_division=0) * 100
    print("テストの各クラスrecall：\n{}\n平均：{}".format(np.round(recall, decimals=1), np.round(recall.mean(), decimals=1)))
    net_weights.append(net.cpu().state_dict())

# 各種パラメータと結果の表示と保存
utils.print_recall(params, ys, ypreds)
utils.save_result(params, ys, ypreds, net_weights)