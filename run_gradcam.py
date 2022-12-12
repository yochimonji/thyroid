import os
from glob import glob

import PIL
import torch
from gradcam import GradCAM
from gradcam.utils import visualize_cam
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from model import create_net
from utils.parse import argparse_gradcam


def main():
    # 使用するファイルのパスを読み込む
    params = argparse_gradcam()
    dataroot = params["test"]
    weight_path = os.path.join("result", params["name"], "weight/weight0.pth")
    save_dir = os.path.join("result", params["name"], params["test_name"], "gradcam")

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:" + params["gpu_id"] if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # 変換の準備
    trans_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    trans_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # ネットワークの準備
    net = create_net(params)
    net.to(device).eval()
    load_weight = torch.load(weight_path, map_location=device)
    net.load_state_dict(load_weight)

    # Grad-CAM
    target_layer = net.net.layer4
    gradcam = GradCAM(net, target_layer)

    # 可視化する画像のパスを貯める辞書
    image_path_list = glob(dataroot + "/**/*.tif", recursive=True)

    for path in tqdm(image_path_list, total=len(image_path_list)):
        # 推論とヒートマップ生成
        pil_img = PIL.Image.open(path)
        torch_img = trans_tensor(pil_img).to(device)
        normed_torch_img = trans_norm(torch_img)[None]
        logit = net(normed_torch_img)
        pred = params["labels"][logit.cpu()[0].argmax()]
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        # 保存するファイル名を生成
        save_file_name, _ = os.path.splitext(path.replace(dataroot, "").replace("_fake", "").strip("/"))
        save_file_name_suffix = save_file_name + "_pred_" + pred + ".tif"
        save_path = os.path.join(save_dir, save_file_name_suffix)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        images = [torch_img.cpu(), heatmap, result]
        grid_image = make_grid(images, nrow=3)
        transforms.ToPILImage()(grid_image).save(save_path)


if __name__ == "__main__":
    main()
