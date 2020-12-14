import os
import glob
import random
import sys

import torch
from torch import nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



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
    def __init__(self, size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 grayscale_flag=False, normalize_per_img=False, multi_net=False):
        if grayscale_flag and multi_net:
            print("grayscale==True and multi_net==Trueはできません")
            sys.exit()

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
            "val": transforms.Compose([  # 他の前処理をまとめる
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
            "val": transforms.Compose([  # 他の前処理をまとめる
                transforms.Resize((size, size)),
                transforms.Grayscale(num_output_channels=3),
                # transforms.CenterCrop(size),
                transforms.ToTensor()  # ndarrayをTensorに変換
                # transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
            ])
        }
        if multi_net:
            self.mean = np.hstack((mean, mean))
            self.std = np.hstack((std, std))
        else:
            self.mean = mean
            self.std = std
        self.grayscale_flag = grayscale_flag
        self.normalize_per_img = normalize_per_img
        self.multi_net = multi_net
        
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
            self.mean = torch.mean(transform_img, dim=(1,2))
            self.std = torch.std(transform_img, dim=(1,2))
        normalize = transforms.Normalize(self.mean, self.std)

        return normalize(transform_img)


# path以下にあるすべてのディレクトリからtifファイルのパスリスト取得
def make_datapath_list(path):
    target_path = os.path.join(path+'/**/*.tif')
    print(target_path)
    path_list = []
    
    # recursive=True:子ディレクトリも再帰的に探索する
    for path in glob.glob(target_path, recursive=True):
        path_list.append(path)
    
    return path_list

    
# img_pathの画像をそのまま・train変換・val変換で表示
# 変換した画像を確認する
def show_transform_img(img_path, transform=ImageTransform()):
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    
    img = Image.open(img_path)
    print("original_img\tmax: {}\tmin: {}".format(np.max(img), np.min(img)))

    ax[0].imshow(img)
    ax[0].set_title("original")

    img_transform_train = transform(img, phase="train")
    img_transform_train = img_transform_train.numpy().transpose((1, 2, 0))
    print("train_img\tmax: {}\tmin: {}".format(np.max(img_transform_train), np.min(img_transform_train)))
#     標準化で0より下の値になるため0~1にクリップ
    img_transform_train = np.clip(img_transform_train, 0, 1)
    ax[1].imshow(img_transform_train)
    ax[1].set_title("train_transform")

    img_transform_val = transform(img, phase="val")
    img_transform_val = img_transform_val.numpy().transpose((1, 2, 0))
#     標準化で0より下の値になるため0~1にクリップ
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
    miss = miss[miss[:, 0]!=miss[:, 1]]  # ミス画像のみ残す
    if y is not None:
        miss = miss[miss[:, 0]==y]  # 本物のラベルがyのみ残す
    if ypred is not None:
        miss = miss[miss[:, 1]==ypred]  # 予測のラベルがypredのみ残す
        
    print("wrong_img_num:", len(miss))
    for (y, ypred, index) in miss:
        img = dataset[index][0]
        plt.imshow(img)
        plt.title("real:{}  prediction:{}".format(
            dataset.label_list[y], dataset.label_list[ypred]))
        plt.show()