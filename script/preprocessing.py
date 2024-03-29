# 画像の正方形化、画像の縮小、ほぼ背景のみの画像の削除を行う。
# コマンド例
# python preprcessing.py DATAPATH SAVEPATH SIZE
# python preprocesing.py ../data/20210427福井大学症例/ ../data/20210427福井大学症例_resized_224/ 224

import os
import sys
import pathlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob


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

# ほぼ背景のみの画像(今の所90％以上が背景)はTrue, 細胞が多く写っている画像はFalse
# ほぼ背景のみの画像は情報量が少ない、もしくはノイズであるため。
def is_white_image(image, threshold):
    PERCENTAGE = 0.9
    gray_image = image.convert(mode='L')
    binary_image = np.array(gray_image) > threshold
    u, count = np.unique(binary_image, return_counts=True)  # u:[False, True], count:[Falseの数, Trueの数]
    if len(u) == 1:
        return u[0]
    else:
        per_white = count[1] / sum(count)
        if per_white >= PERCENTAGE:
            return True
        else:
            return False

if __name__ == '__main__':
    if (len(sys.argv) == 4) and (os.path.exists(sys.argv[1])):
        BASEPATH = sys.argv[1]
        SAVEBASEPATH = sys.argv[2]
        SAVEWHITEPATH = SAVEBASEPATH + 'white'
        SIZE = int(sys.argv[3])  # リサイズする大きさ
    else:
        print('Error:正しいコマンドを入力したください')
        sys.exit()

    search_path = BASEPATH + '/**/*.tif'
    path_list = []
    for path in glob.glob(search_path, recursive=True):
        path_list.append(path)

    print('前処理：縮小、正方形化、ほぼ背景のみの画像の削除を行います')
    for path in tqdm(path_list):
        if 'frozen' in path:
            background_color = (221, 207, 220)
            threshold = 203
        elif 'tissue' in path:
            background_color = (234, 227, 226)
            threshold = 220
        else:
            background_color = (234, 228, 224)  # 背景とほぼ同じ色
            threshold = 220

        image = Image.open(path)
        result_image = resize(image=image, size=SIZE)
        result_image = padding_square(image=result_image, background_color=background_color)
        if is_white_image(result_image, threshold=threshold):
            continue
            # save_path = pathlib.Path(path.replace(str(BASEPATH), str(SAVEWHITEPATH)))
        else:
            save_path = pathlib.Path(path.replace(str(BASEPATH), str(SAVEBASEPATH)))
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        result_image.save(save_path)