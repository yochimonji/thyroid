import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.utils import make_datapath_list


if __name__ == "__main__":
    data_path = "../data/"
    train_list = make_datapath_list(os.path.join(data_path, "train"))
    test_list = make_datapath_list(os.path.join(data_path, "test"))
    size = 224
    imgs = np.empty((0, size*size*3), float)
    print(imgs.shape)

    for data_list in (train_list, test_list):
        for img_path in data_list:
            img = Image.open(img_path)
            img = img.resize((size, size))
            img = np.ravel(img)
            imgs = np.vstack((imgs, img))

    print(imgs.shape)
    pca = PCA(whiten=True)
    imgs_pca = pca.fit_transform(imgs)
    print(imgs_pca.shape)