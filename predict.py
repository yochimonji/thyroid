# 標準ライブラリ
import os
from collections import Counter
from glob import glob

# 外部ライブラリ
import torch
from torch.utils.data import DataLoader

# 自作ライブラリ
from model import change_state_dict_model_to_net, create_net, eval_net
from utils import ImageTransform, print_and_save_result, save_params, save_path_y_ypred
from utils.dataset import CustomImageDataset, make_datapath_list, make_label_list
from utils.parse import argparse_test


def predict():
    # オプション引数からjsonファイルを読み込んでパラメータを設定する
    params = argparse_test()

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:" + params["gpu_id"] if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    path_list = make_datapath_list(params["test"], params["labels"])
    label_list = make_label_list(path_list, params["labels"])
    print("クラスごとのデータ数", sorted(Counter(label_list).items()))

    transform = ImageTransform(params)

    test_dataset = CustomImageDataset(path_list, label_list, transform, phase="test")
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4)

    net = create_net(params)

    if params["weight_dir"]:
        weight_path_list = glob(params["weight_dir"] + "/*.pth")
        params["num_estimate"] = len(weight_path_list)
        assert weight_path_list != []

    ys = []
    ypreds = []

    for i in range(params["num_estimate"]):
        print("\n推論: {}/{}".format(i + 1, params["num_estimate"]))
        # ネットワークに重みをロードする
        if params["weight_dir"]:
            weight_path = weight_path_list[i]
            load_weight = torch.load(weight_path, map_location=device)
            load_weight = change_state_dict_model_to_net(load_weight)
        else:
            weight_path = os.path.join("result", params["name"], "weight/weight" + str(i) + ".pth")
            load_weight = torch.load(weight_path, map_location=device)
        net.load_state_dict(load_weight)
        print("ネットワークに重みをロードしました")
        print("-----推論中-----")

        # 推論
        y, ypred = eval_net(net, test_loader, probability=True, device=device)

        ys.append(y.cpu().numpy())
        ypreds.append(ypred.cpu().numpy())

    dir_path = os.path.join("result", params["name"], params["test_name"])
    print_and_save_result(ys, ypreds, params["labels"], dir_path)
    save_path_y_ypred(path_list, ys[0], ypreds[0], params["labels"], dir_path)
    save_params(params)


if __name__ == "__main__":
    predict()
