import itertools
import random

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from torch.utils.data import Dataset

from utils import ImageTransform, make_datapath_list


# データ数を調整したDatasetを作成するクラス
# オーバー・アンダーサンプリング用
class ArrangeNumDataset(Dataset):
    def __init__(self, params, phase):
        self.params = params
        self.labels = params["labels"]
        self.phase = phase
        self.transform = ImageTransform(params)
        self.file_list = self.make_file_list()  # データ数調整後のファイルリスト。self.label_listと対。
        self.label_list = self.make_label_list()  # データ数調整後のラベルリスト。self.file_listと対。
        self.data_num = np.bincount(np.array(self.label_list))  # クラスごとのデータ数
        if len(self.data_num) < len(self.labels):
            self.data_num = np.concatenate([self.data_num, np.array([0] * (len(self.labels) - len(self.data_num)))])
        self.weight = self.calc_weight()  # 損失関数の重み調整用の重み。
        print("{}の各データ数： {}\t計: {}".format(self.phase, self.data_num, self.data_num.sum()))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img, self.phase)

        label = self.label_list[index]

        return img, label

    def make_file_list(self):
        if self.phase == "train":
            file_list = []
            path_trainA = self.params["trainA"]
            path_trainB = self.params["trainB"]
            if path_trainA:
                file_list.extend(make_datapath_list(path_trainA, self.labels))
            if path_trainB:
                file_list.extend(make_datapath_list(path_trainB, self.labels))
        if self.phase == "test":
            file_list = make_datapath_list(self.params["test"], self.labels)

        arrange = self.params["imbalance"]
        # データ数の調整ありの場合
        if ((arrange == "oversampling") or (arrange == "undersampling")) and (self.phase == "train"):
            arrange_file_list = []
            file_dict = self.make_file_dict(file_list)

            # undersampling(+bagging)を行う場合
            if arrange == "undersampling":
                min_file_num = float("inf")
                for val in file_dict.values():
                    min_file_num = min(min_file_num, len(val))
                for val in file_dict.values():
                    # データの重複あり(baggingする場合はこっち)
                    # arrange_file_list.append(random.choices(val, k=min_file_num))
                    # データの重複なし(baggingしない場合はこっち)
                    arrange_file_list.append(random.sample(val, min_file_num))

            # oversamplingを行う場合
            elif arrange == "oversampling":
                max_file_num = 0
                for val in file_dict.values():
                    max_file_num = max(max_file_num, len(val))
                for val in file_dict.values():
                    arrange_file_list.append(random.choices(val, k=max_file_num))  # 重複あり
                # random.sampleは再標本化後の数値kがもとの要素数より大きいと使えない

            file_list = sorted(list(itertools.chain.from_iterable(arrange_file_list)))
        return file_list

    # key:ラベル、value:ファイルパスリストの辞書を作成
    def make_file_dict(self, file_list):
        label_dict = {}
        for label in self.labels:
            label_dict[label] = list()
        for file in file_list:
            for key in label_dict:
                if ("/" + key + "/") in file:
                    label_dict[key].append(file)
        return label_dict

    # self.fileリストと対になるラベルのリストを作成する
    def make_label_list(self):
        label_list = []
        for file in self.file_list:
            for label in self.labels:
                if ("/" + label + "/") in file:
                    label_list.append(self.labels.index(label))

        return label_list

    # ラベル数に応じてweightを計算する
    # 戻り値がnp.arrayなのに注意。PyTorchで使う場合、Tensorに変換する必要あり
    def calc_weight(self):
        data_num_sum = self.data_num.sum()
        weight = []
        for n in self.data_num:
            if n == 0:
                weight.append(0)
            else:
                weight.append(data_num_sum / n)
        weight = torch.tensor(weight).float()
        return weight


class CustomImageDataset(Dataset):
    def __init__(self, path_list: list[str], label_list: list[int], transform: ImageTransform, phase: str = "train"):
        self.path_list = path_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img_path = self.path_list[index]
        img = Image.open(img_path)
        img = self.transform(img, self.phase)
        label = self.label_list[index]

        return img, label


# 複数のデータセットを結合し、1つのデータセットとするクラス
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.label_list = self.make_label_list()
        self.weight = self.calc_weight()

    def __len__(self):
        length = 0
        for dataset in self.datasets:
            length += dataset.__len__()
        return length

    def __getitem__(self, index):
        for dataset in self.datasets:
            length = dataset.__len__()
            if index < length:
                img, label = dataset.__getitem__(index)
                break
            else:
                index -= length
        return img, label

    def make_label_list(self):
        label_list = []
        for dataset in self.datasets:
            label_list.extend(dataset.make_label_list())
        return label_list

    def calc_weight(self):
        data_num = np.bincount(np.array(self.label_list))
        data_num_sum = data_num.sum()
        weight = []
        for n in data_num:
            weight.append(data_num_sum / n)
        weight = torch.tensor(weight).float()
        return weight


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
