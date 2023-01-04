# 標準ライブラリ
import os
import random
from collections import Counter

# 外部ライブラリ
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

# 自作ライブラリ
from model import create_net, train_net
from model.loss import create_loss
from utils import ImageTransform, save_params, save_weights
from utils.dataset import (
    CustomImageDataset,
    arrange_data_num_per_label,
    make_datapath_list,
    make_label_list,
)
from utils.parse import argparse_train


def main():
    # 乱数シード値を固定して再現性を確保
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # オプション引数をparamsに格納
    params = argparse_train()

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:" + params["gpu_id"] if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    path_list = make_datapath_list(params["trainA"], params["labels"])
    label_list = make_label_list(path_list, params["labels"])
    if params["trainB"]:
        B_path_list = make_datapath_list(params["trainB"], params["labels"])
        B_label_list = make_label_list(B_path_list, params["labels"])
        path_list += B_path_list
        label_list += B_label_list

    transform = ImageTransform(params)

    net_weights = []  # estimateごとのネットワークの重みリスト

    for i in range(params["num_estimate"]):
        print("\n学習: {}/{}".format(i + 1, params["num_estimate"]))

        # 訓練とテストのデータセットを作成する
        path_list, label_list = arrange_data_num_per_label(params["imbalance"], path_list, label_list)
        train_dataset = CustomImageDataset(path_list, label_list, transform, phase="train")
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4)
        print("クラスごとのデータ数", sorted(Counter(label_list).items()))

        # 損失関数のクラス数に合わせてweightをかけるか決める
        if params["imbalance"] == "inverse_class_freq":
            loss_weight = train_dataset.weight.to(device)  # deviceに送らないと動かない
            print("loss_weight:", loss_weight.cpu())
        else:
            loss_weight = None
        loss_fn = create_loss(params["loss_name"], weight=loss_weight, focal_gamma=params["focal_gamma"])

        net = create_net(params)

        # 使用する最適化手法を設定する
        if "Adam" == params["optim_name"]:
            optimizer = optim.Adam(
                net.get_params_lr(lr_not_pretrained=params["lr_not_pretrained"], lr_pretrained=params["lr_pretrained"]),
                weight_decay=params["weight_decay"],
            )
        elif "SGD" == params["optim_name"]:
            optimizer = optim.SGD(
                net.get_params_lr(lr_not_pretrained=params["lr_not_pretrained"], lr_pretrained=params["lr_pretrained"]),
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
            )

        # 学習
        train_net(net, train_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=params["epochs"], device=device)
        net_weights.append(net.cpu().state_dict())

    dir_path = os.path.join("result", params["name"])
    if not os.path.exists(os.path.join(dir_path, "weight")):
        os.makedirs(os.path.join(dir_path, "weight"))
    save_params(params)
    save_weights(net_weights, dir_path)


if __name__ == "__main__":
    main()
