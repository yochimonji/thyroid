import json
import os
from glob import glob

import PIL
import torch
from gradcam import GradCAM
from gradcam.utils import visualize_cam
from torchvision import transforms
from torchvision.utils import make_grid

import utils
from model import create_net
from utils.parse import argparse_gradcam


def main():
    # 使用するファイルのパスを読み込む
    args = argparse_gradcam()
    params_path = str(args.params_path) if args.params_path else os.path.join(str(args.dir), "params.example.json")
    params = utils.load_params(params_path)
    image_dir_path = str(args.image_dir_path) if args.image_dir_path else str(params["data_path"]["test"])
    if image_dir_path[-1] != "/":
        image_dir_path += "/"
    weight_path = str(args.weight_path) if args.weight_path else os.path.join(str(args.dir), "weight/weight0.pth")
    save_dir = os.path.join(str(args.dir), "gradcam" + image_dir_path.replace("./data", "").replace("data", ""))
    os.makedirs(save_dir, exist_ok=True)

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    labels: list = params["labels"]
    image_paths = {}
    for l1 in labels:
        for l2 in labels:
            image_paths[l1 + "_" + l2] = []
    image_paths_all = glob(image_dir_path + "/**/*.tif", recursive=True)

    # ラベルごとに可視化する画像を決定
    for path in image_paths_all:
        pil_img = PIL.Image.open(path)
        torch_img = trans_tensor(pil_img).to(device)
        normed_torch_img = trans_norm(torch_img)[None]
        logit = net(normed_torch_img)

        temp_path = path.replace(image_dir_path, "")
        actual = temp_path[: temp_path.find("/")]
        pred = labels[logit.cpu()[0].argmax()]
        image_paths_key = actual + "_" + pred

        if len(image_paths[image_paths_key]) <= 5:
            image_paths[image_paths_key].append(path)

    # 可視化
    for key, path_list in image_paths.items():
        if path_list == []:
            continue
        images = []
        for path in path_list:
            pil_img = PIL.Image.open(path)
            torch_img = trans_tensor(pil_img).to(device)
            normed_torch_img = trans_norm(torch_img)[None]
            mask, _ = gradcam(normed_torch_img)
            heatmap, result = visualize_cam(mask, torch_img)
            images.extend([torch_img.cpu(), heatmap, result])
        grid_image = make_grid(images, nrow=3)

        filename = "actual_" + key.split("_")[0] + "_pred_" + key.split("_")[1] + ".png"
        transforms.ToPILImage()(grid_image).save(os.path.join(save_dir, filename))

    with open(os.path.join(save_dir, "visualized_image_paths.json"), "w") as file:
        json.dump(image_paths, file, indent=4)


if __name__ == "__main__":
    main()
