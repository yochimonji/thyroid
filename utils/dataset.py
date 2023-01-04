import glob
import os
from collections import Counter

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from torch.utils.data import Dataset

from utils import ImageTransform


class CustomImageDataset(Dataset):
    def __init__(self, path_list: list[str], label_list: list[int], transform: ImageTransform, phase: str = "train"):
        self.path_list = path_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase
        self.weight = compute_class_weight(self.label_list)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img_path = self.path_list[index]
        img = Image.open(img_path)
        img = self.transform(img, self.phase)
        label = self.label_list[index]

        return img, label


def arrange_data_num_per_label(
    imbalance: str, path_list: list[str], label_list: list[int]
) -> tuple[list[str], list[int]]:
    """クラスごとのデータ数を揃える

    Args:
        imbalance (str): データ数を揃える方式。"undersampling" or "oversampling".
        path_list (list[str]): データ数を揃えるパスのリスト。
        label_list (list[int]): データ数を揃えるラベルのリスト。

    Returns:
        tuple[list[str], list[int]]: データ数を揃えたパスのリストとラベルのリストのタプル。
    """
    if imbalance == "undersampling":
        path_array = np.array(path_list).reshape(len(path_list), 1)
        rus = RandomUnderSampler(random_state=0)
        path_array_resampled, label_list_resampled = rus.fit_resample(path_array, label_list)
        path_list_resampled = list(path_array_resampled.reshape(len(label_list_resampled)))
        return path_list_resampled, label_list_resampled
    elif imbalance == "oversampling":
        path_array = np.array(path_list).reshape(len(path_list), 1)
        ros = RandomOverSampler(random_state=0)
        path_array_resampled, label_list_resampled = ros.fit_resample(path_array, label_list)
        path_list_resampled = list(path_array_resampled.reshape(len(label_list_resampled)))
        return path_list_resampled, label_list_resampled
    else:
        return path_list, label_list


def compute_class_weight(label_list: list[int]) -> torch.Tensor:
    label_count = Counter(label_list)
    sorted_label_count = sorted(label_count.items())
    tensor_label_count = torch.tensor(sorted_label_count)[:, 1]
    weight = len(label_list) / tensor_label_count
    return weight


def make_datapath_list(dir_path: str, labels: list[str]) -> list[str]:
    """path以下のフォルダ名がlabelsと一致するすべてのフォルダからtifファイルのパスリスト取得

    Args:
        dir_path (str): 探索するフォルダ
        labels (list[str]): dir_path以下に存在するフォルダ名のリスト

    Returns:
        list[str]: tifファイルのリスト
    """
    if dir_path[-1] != "/":
        dir_path += "/"
    search_path_list: list[str] = []
    for label in labels:
        search_path_list.append(os.path.join(dir_path, label, "**/*.tif"))

    path_list: list[str] = []
    # recursive=True:子ディレクトリも再帰的に探索する
    for search_path in search_path_list:
        for path in glob.glob(search_path, recursive=True):
            path_list.append(path)

    return path_list


def make_label_list(path_list: list[str], labels: list[str]) -> list[int]:
    """tifファイルのリストからtifファイルと組になるlabelのリスト生成する

    Args:
        path_list (list[str]): tifファイルのリスト
        labels (list[str]): ラベルの一覧のリスト

    Returns:
        list[str]: tifファイルと組になるlabelのindexのリスト
    """
    label_list: list[int] = []
    for path in path_list:
        for label in labels:
            if label in path:
                label_list.append(labels.index(label))
                break
    return label_list


def make_group_list(path_list: list[str], path_root: str) -> list[str]:
    group_list = []
    for path in path_list:
        group = path.replace(path_root, "")
        group = group[: group.rfind("/")]
        group_list.append(group)
    return group_list
