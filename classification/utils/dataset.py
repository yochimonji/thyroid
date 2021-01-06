import random
import itertools

from torch.utils.data import Dataset
from PIL import Image
import numpy as np



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
        self.phase = phase
        self.transform = transform
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
            if n == 0:
                weights.append(0)
            else:
                weights.append(data_num_sum / n)
        
        return weights
            

# 複数のデータセットを結合し、1つのデータセットとするクラス
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.labels = self.make_labels()
        self.weights = self.calc_weights()
    
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

    def make_labels(self):
        labels = []
        for dataset in self.datasets:
            labels.extend(dataset.make_labels())
        return labels

    def calc_weights(self):
        data_num = np.bincount(np.array(self.labels))
        data_num_sum = data_num.sum()
        weights = []
        for n in data_num:
            weights.append(data_num_sum / n)
        
        return weights