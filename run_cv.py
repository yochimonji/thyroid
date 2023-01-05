import os
import random
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import optim
from torch.utils.data import DataLoader

from model import create_net, eval_net, train_net
from model.loss import create_loss
from utils import (
    ImageTransform,
    calc_score,
    print_and_save_result,
    print_score,
    save_params,
)
from utils.dataset import (
    CustomImageDataset,
    arrange_data_num_per_label,
    make_datapath_list,
    make_group_list,
    make_label_list,
)
from utils.parse import argparse_cv


def main():
    # 乱数シード値を固定して再現性を確保
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # オプション引数をparamsに格納
    params = argparse_cv()

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:" + params["gpu_id"] if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    path_list = make_datapath_list(params["trainA"], params["labels"])
    label_list = make_label_list(path_list, params["labels"])
    group_list = make_group_list(path_list, params["trainA"])
    if params["trainB"]:
        B_path_list = make_datapath_list(params["trainB"], params["labels"])
        B_label_list = make_label_list(B_path_list, params["labels"])
        B_group_list = make_group_list(B_path_list, params["trainB"])
        path_list += B_path_list
        label_list += B_label_list
        group_list += B_group_list

    transform = ImageTransform(params)

    ys = []
    ypreds = []

    gss = GroupShuffleSplit(n_splits=params["cv_n_split"], test_size=0.33, random_state=7)

    for cv_num, (train_indices, val_indices) in enumerate(gss.split(path_list, label_list, group_list)):
        print("\n交差検証: {}/{}".format(cv_num + 1, params["cv_n_split"]))

        train_path_list = [path_list[i] for i in train_indices]
        train_label_list = [label_list[i] for i in train_indices]
        val_path_list = [path_list[i] for i in val_indices]
        val_label_list = [label_list[i] for i in val_indices]

        train_path_list, train_label_list = arrange_data_num_per_label(
            params["imbalance"], train_path_list, train_label_list
        )
        print("訓練のクラスごとのデータ数: ", sorted(Counter(train_label_list).items()))

        train_dataset = CustomImageDataset(train_path_list, train_label_list, transform, phase="train")
        val_dataset = CustomImageDataset(val_path_list, val_label_list, transform, phase="test")

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4)

        loss_fn = create_loss(
            params["loss_name"],
            imbalance=params["imbalance"],
            label_list=train_label_list,
            focal_gamma=params["focal_gamma"],
            class_balanced_beta=params["class_balanced_beta"],
            device=device,
        )
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
        print("学習")
        train_net(net, train_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=params["epochs"], device=device)

        # 推論
        print("\n推論")
        y, ypred = eval_net(net, val_loader, probability=False, device=device)

        ys.append(y.cpu().numpy())
        ypreds.append(ypred.cpu().numpy())

        score = calc_score(ys, ypreds, len(params["labels"]), need_std=False)
        print_score(score, params["labels"], need_std=False)

    dir_path = os.path.join("result", params["name"])
    print_and_save_result(ys, ypreds, params["labels"], dir_path, is_cv=True)
    save_params(params, file_name="params_cv.json")


if __name__ == "__main__":
    main()
