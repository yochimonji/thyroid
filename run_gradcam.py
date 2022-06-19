import os

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.models as models
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import utils
from model import create_net, eval_net
from utils.dataset import ArrangeNumDataset
from utils.parse import argparse_gradcam


def main():
    # 使用するファイルのパスを読み込む
    args = argparse_gradcam()
    params_path = str(args.params_path) if args.params_path else os.path.join(str(args.dir), "params.json")
    params = utils.load_params(params_path)
    image_dir_path = str(args.image_dir_path) if args.image_dir_path else str(params["data_path"]["test"])
    weight_path = str(args.weight_path) if args.weight_path else os.path.join(str(args.dir), "weight/weight0.pth")

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)


if __name__ == "__main__":
    main()
