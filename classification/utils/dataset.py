import random
import itertools

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from utils import make_datapath_list


# データ数を調整したDatasetを作成するクラス
# オーバー・アンダーサンプリング用
class ArrangeNumDataset(Dataset):
    def __init__(self, params, phase, transform):
        self.params = params
        self.labels = params["labels"]
        self.phase = phase
        self.transform = transform
        self.file_list = self.make_file_list()
        self.make_label_list()  
        self.weights = self.calc_weights()  # ラベルリストからweightのリストを生成
        
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
        file_list =  make_datapath_list(self.params["data_path"][self.phase], self.labels)
        
        arrange = self.params["dataset_params"]["arrange"]
        # データ数の調整ありの場合
        if arrange:
            arrange_file_list = []
            file_dict = self.make_file_dict(file_list, labels)
            
            # undrersampling(+bagging)を行う場合
            if arrange == "undersampling":
                min_file_num = float("inf")
                for val in file_dict.values():
                    min_file_num = min(min_file_num, len(val))
                for val in file_dict.values():
#                     データの重複あり(baggingする場合はこっち)
#                     arrange_file_list.append(random.choices(val, k=min_file_num))
#                     データの重複なし(baggingしない場合はこっち)
                    arrange_file_list.append(random.sample(val, min_file_num))
            
            # oversamplingを行う場合
            elif arrange == "oversampling":
                max_file_num = 0
                for val in file_dict.values():
                    max_file_num = max(max_file_num, len(val))
                for val in file_dict.values():
                    arrange_file_list.append(random.choices(val, k=max_file_num)) # 重複あり
#                     random.sampleは再標本化後の数値kがもとの要素数より大きいと使えない
                
            file_list = list(itertools.chain.from_iterable(arrange_file_list))
        return file_list
    
    # key:ラベル、value:ファイルパスリストの辞書を作成
    def make_file_dict(self, file_list, labels):
        label_dict = {}
        for label in labels:
            label_dict[label] = list()
        for file in file_list:
            for key in label_dict:
                if key in file:
                    labels[key].append(file)
        return label_dict
    
    # self.fileリストと対になるラベルのリストを作成する
    def make_label_list(self):
        self.label_list = []
        for file in self.file_list:
            for label in self.labels:
                if label in file:
                    self.label_list.append(self.labels.index(label))
            
        self.label_list
    
    # ラベル数に応じてweightを計算する
    # 戻り値がnp.arrayなのに注意。PyTorchで使う場合、Tensorに変換する必要あり
    def calc_weights(self):
        self.data_num = np.bincount(np.array(self.label_list))
        self.data_num_sum = self.data_num.sum()
        weights = []
        for n in self.data_num:
            if n == 0:
                weights.append(0)
            else:
                weights.append(self.data_num_sum / n)
        
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