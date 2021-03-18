# 画像の正方形化、画像の縮小、ほぼ背景のみの画像の削除を行う。

import os
import sys
import pathlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils import make_datapath_list


# 画像の縮小は単純にPCのHDDの容量が驚きの500GBだったため行う。。。（全データ容量約570GB）
# アスペクト比を維持しながら長辺を224pixelにする。
# ちなみに1024*1024を224*224に縮小すると容量は約20分の１になる
def resize(image, size=224):
    width, height = image.size
    if width == height:
        resize_image = image.resize((size, size), Image.LANCZOS)
    elif width > height:
        resize_image = image.resize((size, int(height / width * size)), Image.LANCZOS)
    else:
        resize_image = image.resize((int(width / height * size), size), Image.LANCZOS)
    return resize_image

# 画像の正方形化は一部の画像が長方形であり、このままではネットワークの学習に悪影響を与えるため行う。
# 長方形の画像を引き伸ばすのではなく、白色で埋めて正方形にする。
def padding_square(image, background_color=(255, 255, 255)):
    width, height = image.size
    if width == height:
        square_image = image
    elif width > height:
        square_image = Image.new(image.mode, (width, width), background_color)
        square_image.paste(image)
    else:
        square_image = Image.new(image.mode, (height, height), background_color)
        square_image.paste(image)
    return square_image


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
    BACKGROUNDCOLOR = (255, 255, 255)

    path_list = make_datapath_list(str(BASEPATH), LABELS)
    print('前処理：縮小、正方形化、ほぼ背景のみの画像の削除を行います')
    for path in tqdm(path_list):
        image = Image.open(path)
        if image.size != (1024, 1024):
            result_image = resize(image=image, size=SIZE)
            result_image = padding_square(image=result_image, background_color=BACKGROUNDCOLOR)

            save_path = pathlib.Path(path.replace(str(BASEPATH), str(SAVEBASEPATH)))
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            result_image.save(save_path)