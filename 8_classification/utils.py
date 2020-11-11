import random
import os
import glob
import itertools

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet50, resnet101, resnet152
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


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
    def __init__(self, size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = {
            "train": transforms.Compose([  # 他の前処理をまとめる
                transforms.Resize((size, size)),  # リサイズ
#                 # scaleのサイズとratioのアスペクト比でクロップ後、sizeにリサイズ
#                 transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
#                 transforms.RandomCrop(size),  # ランダムにクロップ後、sizeにリサイズ
                transforms.RandomHorizontalFlip(),  # 50%の確率で左右対称に変換
                transforms.RandomVerticalFlip(),  # 50%の確率で上下対象に変換
                MyRotationTransform([0, 90, 180, 270]),  # [0, 90, 180, 270]度で回転
                transforms.ToTensor(),  # ndarrayをTensorに変換
                transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
            ]),
            "val": transforms.Compose([  # 他の前処理をまとめる
                transforms.Resize((size, size)),
#                 transforms.CenterCrop(size),
                transforms.ToTensor(),  # ndarrayをTensorに変換
                transforms.Normalize(mean, std)  # 各色の平均値と標準偏差で標準化
            ])
        }
        
    def __call__(self, img, phase):
        return self.transform[phase](img)
    
    
# path以下にあるすべてのディレクトリからtifファイルのパスリスト取得
def make_datapath_list(path):
    target_path = os.path.join(path+'/**/*.tif')
    print(target_path)
    path_list = []
    
    # recursive=True:子ディレクトリも再帰的に探索する
    for path in glob.glob(target_path, recursive=True):
        path_list.append(path)
    
    return path_list


# データ数を調整したDatasetを作成するクラス
# オーバー・アンダーサンプリング用
class ArrangeNumDataset(Dataset):
    def __init__(self, file_list, label_list, phase=None, transform=None, arrange=None):
        # データ数の調整なしの場合
        if arrange == None:
            self.file_list = file_list
            
        else:
            self.file_list = []
            file_dict = self.make_file_dict(file_list, label_list)
            
            # undrersampling(+bagging)を行う場合
            if arrange == "undersampling":
                min_file_num = float("inf")
                for val in file_dict.values():
                    min_file_num = min(min_file_num, len(val))
                for val in file_dict.values():
#                     データの重複あり(baggingする場合はこっち)
#                     self.file_list.append(random.choices(val, k=min_file_num))
#                     データの重複なし(baggingしない場合はこっち)
                    self.file_list.append(random.sample(val, min_file_num))
            
            # oversamplingを行う場合
            elif arrange == "oversampling":
                max_file_num = 0
                for val in file_dict.values():
                    max_file_num = max(max_file_num, len(val))
                for val in file_dict.values():
                    self.file_list.append(random.choices(val, k=max_file_num)) # 重複あり
#                     random.sampleは再標本化後の数値kがもとの要素数より大きいと使えない
                
            self.file_list = list(itertools.chain.from_iterable(self.file_list))
            
        self.label_list = label_list
        self.transform = transform
        self.phase = phase
        self.labels = self.make_labels()  # self.fileリストと対になるラベルのリスト
        self.weights = self.calc_weights()  # ラベルリストからweightのリストを生成
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img, self.phase)
        
        label = self.labels[index]
            
        return img, label
    
    # key:ラベル、value:ファイルパスリストの辞書を作成
    def make_file_dict(self, file_list, label_list):
        labels = {}
        for label in label_list:
            labels[label] = list()
        for file in file_list:
            for key in labels.keys():
                if key in file:
                    labels[key].append(file)
        return labels
    
    # self.file_listのラベルリストを返却する
    def make_labels(self):
        labels = []
        for file in self.file_list:
            for label in self.label_list:
                if label in file:
                    labels.append(self.label_list.index(label))
            
        return labels
    
    # ラベル数に応じてweightを計算する
    # 戻り値がnp.arrayなのに注意。PyTorchで使う場合、Tensorに変換する必要あり
    def calc_weights(self):
        data_num = np.bincount(np.array(self.labels))
        data_num_sum = data_num.sum()
        weights = []
        for n in data_num:
            weights.append(data_num_sum / n)
        
        return weights
            
    
    
# img_pathの画像をそのまま・train変換・val変換で表示
# 変換した画像を確認する
def show_transform_img(img_path):
    img = Image.open(img_path)

    plt.imshow(img)
    plt.show()
    
    transform = ImageTransform()

    img_transform_train = transform(img, phase="train")
    img_transform_train = img_transform_train.numpy().transpose((1, 2, 0))
#     標準化で0より下の値になるため0~1にクリップ
    img_transform_train = np.clip(img_transform_train, 0, 1)
    plt.imshow(img_transform_train)
    plt.show()

    img_transform_val = transform(img, phase="val")
    img_transform_val = img_transform_val.numpy().transpose((1, 2, 0))
#     標準化で0より下の値になるため0~1にクリップ
    img_transform_val = np.clip(img_transform_val, 0, 1)
    plt.imshow(img_transform_val)
    plt.show()
    
    
# 初期化したネットワークを返却
def init_net(only_fc=True, pretrained=True):
#     net = resnet18(pretrained=pretrained)
#     net = resnet50(pretrained=pretrained)
    net = resnet101(pretrained=pretrained)  # 性能良い
#     net = resnet152(pretrained=pretrained)
    
    # 最終の全結合層のみ重みの計算をするか否か
    # する：FineTuning, しない：転移学習
    if only_fc:
        for p in net.parameters():
            p.requires_grad = False
            
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 8)
    
    return net


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
        
        
# ネットワークで推論を行う
# 戻り値：本物ラベル、予測ラベル
# loaderの設定によってはシャッフルされているので本物ラベルも返す必要がある
def eval_net(net, loader, device="cpu"):
    net.eval()
    ys = []
    ypreds = []
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            ypred = net(x).argmax(1)
        ys.append(y)
        ypreds.append(ypred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    
    return ys, ypreds


# ネットワークの訓練を行う
def train_net(net, train_loader, val_loader, optimizer,
              loss_fn=nn.CrossEntropyLoss(), epochs=10, only_fc=True, device="cpu"):
    
    net = net.to(device)
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for (x, y) in tqdm(train_loader, total=len(train_loader)):
            x = x.to(device)
            y = y.to(device)
            h = net(x)
            loss = loss_fn(h, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # lossは1イテレーションの平均値なので、x.size(0)をかけて足し合わせると合計値になる
            # len(dataset)/batch_sizeが割り切れないときのために必要
            train_loss += loss.item() * x.size(0)
            ypred = h.argmax(1)
            train_acc += (y == ypred).sum()  # 予測があっている数を計算
            
        # len(train_loader)：イテレーション数
        # len(train_loader.dataset)：データセット数
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        val_ys, val_ypreds = eval_net(net, val_loader, device=device)
        val_acc = (val_ys == val_ypreds).sum().float() / len(val_loader.dataset)
        
        print("epoch:{}/{}  train_loss: {:.3f}  train_acc: {:.3f}  val_acc: {:.3f}".format(
        epoch+1, epochs, train_loss, train_acc, val_acc), flush=True)
        
    torch.save(net.state_dict(), "weight/last_weight.pth")
    
    return net