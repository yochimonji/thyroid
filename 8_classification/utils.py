import random
import os
import glob

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