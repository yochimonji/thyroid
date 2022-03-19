import glob
import itertools
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error)
from sklearn.metrics import precision_recall_fscore_support as scores
from sklearn.metrics import recall_score
from torchvision import transforms


# 与えられた角度をランダムに一つ選択する
# 0, 90, 180, 270度で回転
# angles:回転したい角度のリスト
class MyRotationTransform():
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)


# 画像に変換処理を行う
# ResNetで転移学習するとき、sizeは224×224、defaultのmean,stdで標準化する
class ImageTransform():
    def __init__(self, params):
        self.grayscale_flag = params["transform_params"]["grayscale_flag"]
        self.normalize_per_img = params["transform_params"]["normalize_per_img"]
        self.multi_net = params["net_params"]["multi_net"]
        mean = params["transform_params"]["mean"]
        std = params["transform_params"]["std"]
        if self.multi_net:
            self.mean = np.hstack((mean, mean))
            self.std = np.hstack((std, std))
        else:
            self.mean = mean
            self.std = std

        size = params["transform_params"]["img_resize"]
        self.transform_rgb = {
            "train": transforms.Compose([  # 他の前処理をまとめる
                transforms.Resize((size, size)),  # リサイズ, 最初にしたほうが処理が軽い
                # scaleのサイズとratioのアスペクト比でクロップ後、sizeにリサイズ
                # transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                # transforms.RandomCrop(size),  # ランダムにクロップ後、sizeにリサイズ
                transforms.RandomHorizontalFlip(),  # 50%の確率で左右対称に変換
                transforms.RandomVerticalFlip(),  # 50%の確率で上下対象に変換
                MyRotationTransform([0, 90, 180, 270]),  # [0, 90, 180, 270]度で回転
                transforms.ToTensor()  # ndarrayをTensorに変換、0〜1に正規化
                # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
            ]),
            "test": transforms.Compose([  # 他の前処理をまとめる
                transforms.Resize((size, size)),
                # transforms.CenterCrop(size),
                transforms.ToTensor()  # ndarrayをTensorに変換
                # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
            ])
        }
        self.transform_gray = {
            "train": transforms.Compose([  # 他の前処理をまとめる
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
            ]),
            "test": transforms.Compose([  # 他の前処理をまとめる
                transforms.Resize((size, size)),
                transforms.Grayscale(num_output_channels=3),
                # transforms.CenterCrop(size),
                transforms.ToTensor()  # ndarrayをTensorに変換
                # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
            ])
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


# path以下にあり、labelsと一致するすべてのディレクトリからtifファイルのパスリスト取得
def make_datapath_list(path, labels):
    if path[-1] != "/":
        path += "/"
    search_path_list = []
    for label in labels:
        search_path_list.append(os.path.join(path, '*'+label+'*/**/*.tif'))

    path_list = []
    # recursive=True:子ディレクトリも再帰的に探索する
    for search_path in search_path_list:
        for path in glob.glob(search_path, recursive=True):
            path_list.append(path)

    return path_list


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
        plt.title("real:{}  prediction:{}".format(
            dataset.label_list[y], dataset.label_list[ypred]))
        plt.show()


# jsonファイルを読み込んでパラメータを設定する
# jsonから読み込むことでpyファイルの書き換えをしなくてよいのでGitが汚れない
def load_params(path="config/params.json"):
    if len(sys.argv) == 2:
        if os.path.exists(sys.argv[1]):
            path = sys.argv[1]
        else:
            print("Error:指定した引数のパスにファイルが存在しません")
            sys.exit()
    with open(path, "r") as file:
        params = json.load(file)
    check_params(params)
    print_params(params)
    return params


def check_params(params):
    net_name = params["net_params"]["name"]
    optim_name = params["optim_params"]["name"]
    grayscale_flag = params["transform_params"]["grayscale_flag"]
    multi_net = params["net_params"]["multi_net"]
    transfer_learning = params["net_params"]["transfer_learning"]
    pretrained = params["net_params"]["pretrained"]

    # 誤っているparamsがあれば終了する
    if not(("resnet" in net_name) or ("efficientnet" in net_name)):
        print("ParamsError:net_params['name']=='{}'は定義されていない".format(net_name))
        sys.exit()
    if not((optim_name == "Adam") or (optim_name == "SGD")):
        print("optim_params['name']=='{}'は定義されていない".format(optim_name))
        sys.exit()
    if grayscale_flag and multi_net:
        print("grayscale==True and multi_net==Trueはできません")
        sys.exit()
    if transfer_learning and (not pretrained):
        print("transfer_learning==True and pretrained=Falseはできません")
        sys.exit()


def print_params(params, nest=0):
    for param in params:
        print("\t"*nest, param, end=":")
        if type(params[param]) == dict:
            print("{")
            print_params(params[param], nest=nest+1)
            print("}\n")
        else:
            print("\t", params[param])


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
    print(np.round(recall_means_by_type*100, decimals=1))
    print("各感度の{}回平均の平均：{}".format(params["num_estimate"], np.round(np.mean(recalls)*100, decimals=1)))
    print("↑に近い各感度の{}回平均のインデックス:".format(params["num_estimate"]), min_error_index)

    y = ys[min_error_index]
    ypred = ypreds[min_error_index]
    print(confusion_matrix(y, ypred))
    print(classification_report(y, ypred, target_names=params["labels"],
                                digits=3, zero_division=0))


# 各種パラメータ、結果、ネットワークの重みを保存する
def save_params(params, weights):
    # フォルダ作成
    path = os.path.join("result", params["name"])
    if not os.path.exists(os.path.join(path, "weight")):
        os.makedirs(os.path.join(path, "weight"))

    # パラメータ保存
    with open(os.path.join(path, "params.json"), "w") as params_file:
        json.dump(params, params_file)

    # ネットワークの重み保存
    for i, weight in enumerate(weights):
        torch.save(weight, os.path.join(path, "weight", "weight" + str(i) + ".pth"))


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
        "Accuracy_Std": accuracy_std
    }
    return score


def format_score_line(score_array, std_array=None):
    formated_line = ""
    if std_array is None:
        for score in score_array:
            formated_line += f"{score:.2f}\t"
    else:
        for score, std in zip(score_array, std_array):
            formated_line += f"{score:.2f}±{std:.2f}\t"
    return formated_line


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


def save_score(params, score, need_mean=True, need_std=True):
    path = os.path.join("result", params["name"])
    if not os.path.exists(path):
        os.makedirs(path)
    precision = format_score_line(score['Precision'], score['Precision_Std']).split()
    recall = format_score_line(score['Recall'], score['Recall_Std']).split()
    f1_score = format_score_line(score['F1 Score'], score['F1 Score_Std']).split()

    if need_std:
        accuracy = f"{score['Accuracy']:.2f}±{score['Accuracy_Std']:.2f}"
    else:
        accuracy = f"{score['Accuracy']:.2f}"

    index = params["labels"].copy()
    if need_mean:
        index.append("平均")

    df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy
    }, index=index).T
    df.to_csv(os.path.join(path, "score.csv"))


def save_y_preds_all_score(params, y, preds):
    # フォルダを作製する
    path = os.path.join("result", params["name"])
    if not os.path.exists(path):
        os.makedirs(path)

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


# ys:推論回数　×　データ数　　2次元配列
# ypreds:推論回数　×　データ数　×　ラベル数　　3次元配列
def print_and_save_result(params, y, preds, need_mean=True, need_std=True, need_confusion_matrix=False):
    if need_confusion_matrix:
        for i, pred in enumerate(preds):
            print(f"Confusion Matrix {i+1}\n{confusion_matrix(y, pred, labels=range(len(params['labels'])))}\n")

    score = calc_score(params, y, preds, need_mean, need_std)

    # 結果の表示と保存
    print_score(params, score, need_mean, need_std)
    save_score(params, score, need_mean, need_std)
    save_y_preds_all_score(params, y, preds)
