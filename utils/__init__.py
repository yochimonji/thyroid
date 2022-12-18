import glob
import itertools
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.metrics import precision_recall_fscore_support as scores
from sklearn.metrics import recall_score
from torchvision import transforms


# 与えられた角度をランダムに一つ選択する
# 0, 90, 180, 270度で回転
# angles:回転したい角度のリスト
class MyRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)


# 画像に変換処理を行う
# ResNetで転移学習するとき、sizeは224×224、defaultのmean,stdで標準化する
class ImageTransform:
    def __init__(self, params):
        self.grayscale_flag = params["grayscale_flag"]
        self.normalize_per_img = params["normalize_per_img"]
        self.multi_net = params["multi_net"]
        mean = params["mean"]
        std = params["std"]
        if self.multi_net:
            self.mean = np.hstack((mean, mean))
            self.std = np.hstack((std, std))
        else:
            self.mean = mean
            self.std = std

        size = params["img_resize"]
        self.transform_rgb = {
            "train": transforms.Compose(
                [  # 他の前処理をまとめる
                    transforms.Resize((size, size)),  # リサイズ, 最初にしたほうが処理が軽い
                    # scaleのサイズとratioのアスペクト比でクロップ後、sizeにリサイズ
                    # transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                    # transforms.RandomCrop(size),  # ランダムにクロップ後、sizeにリサイズ
                    transforms.RandomHorizontalFlip(),  # 50%の確率で左右対称に変換
                    transforms.RandomVerticalFlip(),  # 50%の確率で上下対象に変換
                    MyRotationTransform([0, 90, 180, 270]),  # [0, 90, 180, 270]度で回転
                    transforms.ToTensor()  # ndarrayをTensorに変換、0〜1に正規化
                    # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
                ]
            ),
            "test": transforms.Compose(
                [  # 他の前処理をまとめる
                    transforms.Resize((size, size)),
                    # transforms.CenterCrop(size),
                    transforms.ToTensor()  # ndarrayをTensorに変換
                    # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
                ]
            ),
        }
        self.transform_gray = {
            "train": transforms.Compose(
                [  # 他の前処理をまとめる
                    transforms.Resize((size, size)),  # リサイズ, 最初にしたほうが処理が軽い
                    transforms.Grayscale(num_output_channels=3),
                    # scaleのサイズとratioのアスペクト比でクロップ後、sizeにリサイズ
                    # transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                    # transforms.RandomCrop(size),  # ランダムにクロップ後、sizeにリサイズ
                    transforms.RandomHorizontalFlip(),  # 50%の確率で左右対称に変換
                    transforms.RandomVerticalFlip(),  # 50%の確率で上下対象に変換
                    MyRotationTransform([0, 90, 180, 270]),  # [0, 90, 180, 270]度で回転
                    transforms.ToTensor()  # ndarrayをTensorに変換、0〜1に正規化
                    # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
                ]
            ),
            "test": transforms.Compose(
                [  # 他の前処理をまとめる
                    transforms.Resize((size, size)),
                    transforms.Grayscale(num_output_channels=3),
                    # transforms.CenterCrop(size),
                    transforms.ToTensor()  # ndarrayをTensorに変換
                    # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
                ]
            ),
        }

    def __call__(self, img, phase):
        if self.multi_net:
            transform_rgb_img = self.transform_rgb[phase](img)
            transform_gray_img = self.transform_gray[phase](img)
            transform_img = torch.cat((transform_rgb_img, transform_gray_img), dim=0)
        else:
            if self.grayscale_flag:
                transform_img = self.transform_gray[phase](img)
            else:
                transform_img = self.transform_rgb[phase](img)

        if self.normalize_per_img:
            self.mean = torch.mean(transform_img, dim=(1, 2))
            self.std = torch.std(transform_img, dim=(1, 2))
        normalize = transforms.Normalize(self.mean, self.std)

        return normalize(transform_img)


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


def make_label_list(path_list: list[str], labels: list[str]) -> list[str]:
    """tifファイルのリストからtifファイルと組になるlabelのリスト生成する

    Args:
        path_list (list[str]): tifファイルのリスト
        labels (list[str]): ラベルの一覧のリスト

    Returns:
        list[str]: tifファイルと組になるlabelのリスト
    """
    label_list: list[str] = []
    for path in path_list:
        for label in labels:
            if label in path:
                label_list.append(label)
                break
    return label_list


# img_pathの画像をそのまま・train変換・val変換で表示
# 変換した画像を確認する
def show_transform_img(img_path, transform):
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

    img = Image.open(img_path)
    print("original_img\tmax: {}\tmin: {}".format(np.max(img), np.min(img)))

    ax[0].imshow(img)
    ax[0].set_title("original")

    img_transform_train = transform(img, phase="train")
    img_transform_train = img_transform_train.numpy().transpose((1, 2, 0))
    print("train_img\tmax: {}\tmin: {}".format(np.max(img_transform_train), np.min(img_transform_train)))
    # 標準化で0より下の値になるため0~1にクリップ
    img_transform_train = np.clip(img_transform_train, 0, 1)
    ax[1].imshow(img_transform_train)
    ax[1].set_title("train_transform")

    img_transform_val = transform(img, phase="test")
    img_transform_val = img_transform_val.numpy().transpose((1, 2, 0))
    # 標準化で0より下の値になるため0~1にクリップ
    img_transform_val = np.clip(img_transform_val, 0, 1)
    ax[2].imshow(img_transform_val)
    ax[2].set_title("val_transform")


# 識別を間違えた画像を表示する
# dataset:transformしていないdataset
# ys,ypreds:本物ラベルリスト、予測ラベルリスト
# indices:シャッフル前のys,ypredsのインデックス（順番）。シャッフルしていない場合None
# y,ypred:表示したいラベル番号
def show_wrong_img(dataset, ys, ypreds, indices=None, y=None, ypred=None):
    if indices is None:
        indices = range(dataset.__len__())

    # miss.shape:(len(dataset), 3)
    miss = np.stack([ys, ypreds, indices], axis=1)
    miss = miss[miss[:, 0] != miss[:, 1]]  # ミス画像のみ残す
    if y is not None:
        miss = miss[miss[:, 0] == y]  # 本物のラベルがyのみ残す
    if ypred is not None:
        miss = miss[miss[:, 1] == ypred]  # 予測のラベルがypredのみ残す

    print("wrong_img_num:", len(miss))
    for (y, ypred, index) in miss:
        img = dataset[index][0]
        plt.imshow(img)
        plt.title("real:{}  prediction:{}".format(dataset.label_list[y], dataset.label_list[ypred]))
        plt.show()


# 結果（主にrecall）を表示する
def print_recall(params, ys, ypreds):
    # recall計算
    recalls = []
    for y, ypred in zip(ys, ypreds):
        recalls.append(recall_score(y, ypred, average=None, zero_division=0))

    # recallの？回平均と各recallの平均二乗誤差が最小のインデックスを求める
    min_error = float("inf")
    min_error_index = 0
    recall_means_by_type = np.mean(recalls, axis=0)
    for i, recall in enumerate(recalls):
        error = mean_squared_error(recall_means_by_type, recall)
        if error < min_error:
            min_error = error
            min_error_index = i

    # 結果表示
    print("各感度の{}回平均".format(params["num_estimate"]))
    print(params["labels"])
    print(np.round(recall_means_by_type * 100, decimals=1))
    print("各感度の{}回平均の平均：{}".format(params["num_estimate"], np.round(np.mean(recalls) * 100, decimals=1)))
    print("↑に近い各感度の{}回平均のインデックス:".format(params["num_estimate"]), min_error_index)

    y = ys[min_error_index]
    ypred = ypreds[min_error_index]
    print(confusion_matrix(y, ypred))
    print(classification_report(y, ypred, target_names=params["labels"], digits=3, zero_division=0))


def calc_confusion_matrix_df(params, y, preds):
    total_confusion_matrix = None
    for i, pred in enumerate(preds):
        if total_confusion_matrix is None:
            total_confusion_matrix = confusion_matrix(y, pred, labels=range(len(params["labels"])))
        else:
            total_confusion_matrix += confusion_matrix(y, pred, labels=range(len(params["labels"])))
    multi_columns = pd.MultiIndex.from_product([["Prediction"], params["labels"]])
    multi_index = pd.MultiIndex.from_product([["Actual"], params["labels"]])
    confusion_matrix_df = pd.DataFrame(
        np.rint(total_confusion_matrix / 10).astype(int), index=multi_index, columns=multi_columns
    )
    return confusion_matrix_df


def save_weights(params, weights):
    # フォルダ作成
    path = os.path.join("result", params["name"])
    if not os.path.exists(os.path.join(path, "weight")):
        os.makedirs(os.path.join(path, "weight"))

    # ネットワークの重み保存
    for i, weight in enumerate(weights):
        torch.save(weight, os.path.join(path, "weight", "weight" + str(i) + ".pth"))


def save_params(params: dict):
    # フォルダ作成
    if params["phase"] == "train":
        path = os.path.join("result", params["name"])
    elif params["phase"] == "test":
        path = os.path.join("result", params["name"], params["test_name"])
    if not os.path.exists(path):
        os.makedirs(path)
    params.pop("gpu_id")

    # パラメータ保存
    with open(os.path.join(path, "params.json"), "w") as params_file:
        json.dump(params, params_file, indent=4)


def mean_and_std_score(all_score_list, y_class_num, need_all_mean=True, need_std=True):
    all_score_array = np.array(all_score_list)
    score_array = all_score_array.mean(axis=0)
    if need_all_mean:
        mean_score = score_array[score_array.nonzero()].sum() / y_class_num
        score_array = np.concatenate([score_array, [mean_score]])

    if need_std:
        std_array = np.std(all_score_array, axis=0, ddof=1)
        if need_all_mean:
            mean_std = std_array[std_array.nonzero()].sum() / y_class_num
            std_array = np.concatenate([std_array, [mean_std]])
        return score_array, std_array
    else:
        return score_array, None


def calc_score(params, y, preds, need_mean=True, need_std=True):
    all_precision_list = []
    all_recall_list = []
    all_f1_score_list = []
    accuracy_list = []
    for i, pred in enumerate(preds):
        result = scores(y, pred, labels=range(len(params["labels"])), zero_division=0)
        all_precision_list.append(result[0] * 100)
        all_recall_list.append(result[1] * 100)
        all_f1_score_list.append(result[2] * 100)
        accuracy_list.append(accuracy_score(y, pred) * 100)

    y_class_num = np.unique(y).size
    precision_array, precision_std = mean_and_std_score(all_precision_list, y_class_num, need_mean, need_std)
    recall_array, recall_std = mean_and_std_score(all_recall_list, y_class_num, need_mean, need_std)
    f1_score_array, f1_score_std = mean_and_std_score(all_f1_score_list, y_class_num, need_mean, need_std)
    accuracy_array, accuracy_std = mean_and_std_score(
        accuracy_list, y_class_num, need_all_mean=False, need_std=need_std
    )

    score = {
        "Precision": precision_array,
        "Precision_Std": precision_std,
        "Recall": recall_array,
        "Recall_Std": recall_std,
        "F1 Score": f1_score_array,
        "F1 Score_Std": f1_score_std,
        "Accuracy": accuracy_array,
        "Accuracy_Std": accuracy_std,
    }
    return score


def format_score_line(score_array, std_array=None):
    formatted_line = ""
    if std_array is None:
        for score in score_array:
            formatted_line += f"{score:.2f}\t"
    else:
        for score, std in zip(score_array, std_array):
            formatted_line += f"{score:.2f}±{std:.2f}\t"
    return formatted_line


def print_score(params, score, need_mean=True, need_std=True):
    print(f"\n{params['num_estimate']}回平均", end="\t")
    for label in params["labels"]:
        print(label, end="\t\t" if need_std else "\t")
    if need_mean:
        print("平均")
    else:
        print("")
    print(f"Precision\t{format_score_line(score['Precision'], score['Precision_Std'])}")
    print(f"Recall\t\t{format_score_line(score['Recall'], score['Recall_Std'])}")
    print(f"F1 Score\t{format_score_line(score['F1 Score'], score['F1 Score_Std'])}")
    if need_std:
        print(f"Accuracy\t{score['Accuracy']:.2f}±{score['Accuracy_Std']:.2f}")
    else:
        print(f"Accuracy\t{score['Accuracy']:.2f}")


def save_score(params, score, path, need_mean=True, need_std=True):
    if need_std:
        precision = format_score_line(score["Precision"], score["Precision_Std"]).split()
        recall = format_score_line(score["Recall"], score["Recall_Std"]).split()
        f1_score = format_score_line(score["F1 Score"], score["F1 Score_Std"]).split()
        accuracy = f"{score['Accuracy']:.2f}±{score['Accuracy_Std']:.2f}"
    else:
        precision = format_score_line(score["Precision"], None).split()
        recall = format_score_line(score["Recall"], None).split()
        f1_score = format_score_line(score["F1 Score"], None).split()
        accuracy = f"{score['Accuracy']:.2f}"

    index = params["labels"].copy()
    if need_mean:
        index.append("平均")

    df = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1 Score": f1_score, "Accuracy": accuracy}, index=index
    ).T
    df.to_csv(path)


def save_y_preds_all_score(params, y, preds, path):
    y_preds_df = pd.DataFrame(dict(y=y))
    all_score_array = None
    for i, pred in enumerate(preds):
        # 各施行の予測値のdf作成
        y_preds_df[f"pred_{i}"] = pred

        # 各施行のスコアを計算
        score = calc_score(params, y, [pred], True, False)
        score_array = np.hstack([score["Precision"], score["Recall"], score["F1 Score"], score["Accuracy"]])
        score_array = format_score_line(score_array).split()
        if all_score_array is None:
            all_score_array = score_array
        else:
            all_score_array = np.vstack([all_score_array, score_array])

    # 計算したスコアをフォーマットして、予測値dfの末尾に連結できる形式にする
    score_name_list = ["Precision", "Recall", "F1 Score"]
    label_name_list = params["labels"].copy()
    label_name_list.append("Mean")
    score_index = []
    for score_name, label_name in itertools.product(score_name_list, label_name_list):
        score_index.append(f"{score_name}_{label_name}")
    score_index.append("Accuracy")
    score_df = pd.DataFrame(
        all_score_array, columns=score_index, index=[f"pred_{i}" for i in range(params["num_estimate"])]
    ).T

    result_df = pd.concat([y_preds_df, score_df])
    result_df.to_csv(os.path.join(path, "y_preds_all_score.csv"))


def calc_voting_score(params, y, preds, need_mean=True):
    sum_preds = np.sum(preds, 0)
    voting_preds = np.argmax(sum_preds, 1)
    result = scores(y, voting_preds, labels=range(len(params["labels"])), zero_division=0)

    y_class_num = np.unique(y).size
    precision_array, _ = mean_and_std_score([result[0] * 100], y_class_num, need_mean, False)
    recall_array, _ = mean_and_std_score([result[1] * 100], y_class_num, need_mean, False)
    f1_score_array, _ = mean_and_std_score([result[2] * 100], y_class_num, need_mean, False)
    accuracy_array, _ = mean_and_std_score([accuracy_score(y, voting_preds) * 100], y_class_num, False, False)

    score = {
        "Precision": precision_array,
        "Recall": recall_array,
        "F1 Score": f1_score_array,
        "Accuracy": accuracy_array,
    }
    return score


# ys:推論回数 × データ数  2次元配列
# ypreds:推論回数 × データ数 × ラベル数  3次元配列
def print_and_save_result(params, y, preds, need_mean=True, need_std=True, need_confusion_matrix=True):
    # フォルダを作製する
    path = os.path.join("result", params["name"], params["test_name"])
    if not os.path.exists(path):
        os.makedirs(path)

    preds_non_probability = np.argmax(preds, 2)

    if need_confusion_matrix:
        confusion_matrix_df = calc_confusion_matrix_df(params, y, preds_non_probability)
        print(confusion_matrix_df)
        confusion_matrix_df.to_csv(os.path.join(path, "confusion_matrix.csv"))

    # 結果の表示と保存
    score = calc_score(params, y, preds_non_probability, need_mean, need_std)
    print_score(params, score, need_mean, need_std)
    save_score(params, score, os.path.join(path, "score.csv"), need_mean, need_std)
    save_y_preds_all_score(params, y, preds_non_probability, path)

    # SoftVotingの計算と保存
    voting_score = calc_voting_score(params, y, preds, need_mean)
    save_score(params, voting_score, os.path.join(path, "score_soft_voting.csv"), need_mean, False)


def save_path_y_ypred(paths: list[str], ys: list, ypreds: list, labels: list[str], save_dir: str):
    """パス、ラベル、予測をCSVファイルに保存する。

    Args:
        paths (list[str]): 保存するパスの一覧。
        ys (list): 保存する本物のラベル（数値）の一覧。
        ypreds (list): 保存する予測したラベル（数値）の一覧。
        labels (list[str]): ラベルのリスト。ys, ypredsのラベル名はlabels[ys[i]], labels[ypreds[i]]となる。
        save_dir (str): 保存するフォルダのパス。

    Raises:
        ValueError: paths, ys, ypredsの配列長が異なる場合に発生。
    """
    if len(paths) != len(ys) or len(paths) != len(ypreds):
        raise ValueError(
            f"Number of paths and y, ypred are not same: \
                len(paths)={len(paths)}. len(y)={len(ys)}. len(ypred)={len(ypreds)}"
        )

    # 理解しやすいようにys, ypredsを癌の名前の一覧に変換する
    ys_label = list(map(lambda y: labels[y], ys))
    ypreds_label = list(map(lambda ypred: labels[np.argmax(ypred)], ypreds))

    ys_ypreds_array = np.array([ys_label, ypreds_label]).T
    ys_ypreds_df = pd.DataFrame(ys_ypreds_array, columns=["real", "pred"], index=paths)
    ys_ypreds_df.to_csv(os.path.join(save_dir, "path_real_pred.csv"))
