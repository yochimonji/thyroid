# 画像の正方形化、画像の縮小、ほぼ背景のみの画像の削除を行う。
# 容量の問題から元の1024*1024pixelデータは削除され、新たに224*224pixelのデータが新規に作成されることに注意する。

import os
import sys
import pathlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from PIL import Image

from utils import make_datapath_list


# 画像の縮小は単純にPCのHDDの容量が驚きの500GBだったため行う。。。（全データ容量約570GB）
# アスペクト比を維持しながら長辺を224pixelにする。
def resize(image, size=224):
    if image.width > image.height:
        resize_image = image.resize((size, int(image.height / image.width * size)))
    else:
        resize_image = image.resize((int(image.width / image.height * size), size))
    return resize_image

# 画像の正方形化は一部の画像が長方形であり、このままではネットワークの学習に悪影響を与えるため行う。
# def padding(image):
#     return padding_image


# ほぼ背景のみの画像の削除は(今の所)90％以上が背景(白色)の画像を削除する。ほぼ背景のみの画像は情報量が少ない、もしくはノイズであるため。
# def remove_white_image(image):


if __name__ == '__main__':
    if (len(sys.argv) == 2) and (os.path.exists(sys.argv[1])):
        BASEPATH = pathlib.Path(sys.argv[1])
        SAVEBASEPATH = BASEPATH.parent / ('prepro_' + str(BASEPATH.name))
    else:
        print('Error:正しいパスを入力してください')
        sys.exit()

    LABELS = ['Normal', 'PTC HE', 'fvptc', 'FTC', 'med', 'poor', 'und']
    SIZE = 224  # リサイズする大きさ

    path_list = make_datapath_list(str(BASEPATH), LABELS)
    for path in path_list:
        image = Image.open(path)
        if (image.width != 1024) or (image.height != 1024):
            resize_image = resize(image=image, size=SIZE)

            save_path = pathlib.Path(path.replace(str(BASEPATH), str(SAVEBASEPATH)))
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            resize_image.save(save_path)