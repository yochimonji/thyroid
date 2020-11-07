import random
import os
import glob

from torch.utils.data import Dataset
from torchvision import transforms
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
    def __init__(self, size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = {
            "train": transforms.Compose([  # 他の前処理をまとめる
                transforms.Resize((size, size)),  # リサイズ
#                 transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
#                 # scaleのサイズとratioのアスペクト比でクロップ後、sizeにリサイズ
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
    def __init__(self, file_list, target_list, phase="train", transform=None, arrange=None):
        # データ数の調整なしの場合
        if arrange == None:
            self.file_list = file_list
            
        else:
            self.file_list = []
            file_dict = self.make_file_dict(file_list, target_list)
            
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
            
        self.target_list = target_list
        self.transform = transform
        self.phase = phase
        self.targets = self.make_targets()
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img, self.phase)
        
        for target in self.target_list:
            if target in img_path:
                label = self.target_list.index(target)
            
        return img, label
    
    # key:ラベル、value:ファイルパスリストの辞書を作成
    def make_file_dict(self, file_list, target_list):
        targets = {}
        for target in target_list:
            targets[target] = list()
        for file in file_list:
            for key in targets.keys():
                if key in file:
                    targets[key].append(file)
        return targets
    
    # self.file_listのラベルリストを返却する
    def make_targets(self):
        targets = []
        for file in self.file_list:
            for target in self.target_list:
                if target in file:
                    targets.append(self.target_list.index(target))
            
        return targets