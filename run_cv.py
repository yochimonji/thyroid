import itertools
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader

from model import eval_net, train_net
from utils import ImageTransform, make_datapath_list, make_label_list
from utils.dataset import ArrangeNumDataset
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
    print(len(path_list), len(label_list))

    # ys = []
    # ypreds = []
    # val_indices_after_skf = []
    # batch_size = 128
    # num_workers = 8
    # only_fc = True  # 転移学習：True, FineTuning：False
    # pretrained = True
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("使用デバイス：", device)

    skf = StratifiedKFold(n_splits=params["cv_n_split"], shuffle=True, random_state=0)

    for cv_num, (train_indices, val_indices) in enumerate(skf.split(path_list, label_list)):
        print(cv_num, len(train_indices), len(val_indices))

    #     print("交差検証：{}/{}".format(cv_num + 1, skf.get_n_splits()))

    #     train_list = [file_list[i] for i in train_indices]
    #     val_list = [file_list[i] for i in val_indices]

    #     train_dataset = ArrangeNumDataset(
    #         train_list, label_list, phase="train", arrange=None, transform=ImageTransform()
    #     )
    #     val_dataset = ArrangeNumDataset(val_list, label_list, phase="test", arrange=None, transform=ImageTransform())

    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #     net = InitEfficientNet(only_fc=only_fc, pretrained=pretrained, model_name="efficientnet-b3")
    #     optimizer = optim.Adam(net.get_params_lr())
    #     weights = torch.tensor(train_dataset.weights).float().cuda()
    #     loss_fn = nn.CrossEntropyLoss(weight=weights)

    #     train_net(net(), train_loader, val_loader, optimizer=optimizer, epochs=2, device=device, loss_fn=loss_fn)
    #     ys_ypreds = eval_net(net(), val_loader, device=device)
    #     ys.append(ys_ypreds[0])
    #     ypreds.append(ys_ypreds[1])
    #     val_indices_after_skf.append(val_indices)

    # ys = torch.cat(ys).cpu().numpy()
    # ypreds = torch.cat(ypreds).cpu().numpy()
    # val_indices_after_skf = list(itertools.chain.from_iterable(val_indices_after_skf))

    # print("accuracy_score:", accuracy_score(ys, ypreds))
    # print(confusion_matrix(ys, ypreds))
    # print(classification_report(ys, ypreds, target_names=label_list, digits=3))


if __name__ == "__main__":
    main()
