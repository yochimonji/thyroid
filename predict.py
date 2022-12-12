# 標準ライブラリ
import os

# 外部ライブラリ
import torch
from torch.utils.data import DataLoader

# 自作ライブラリ
import utils
from model import create_net, eval_net
from utils.dataset import ArrangeNumDataset
from utils.parse import argparse_test


def predict():
    # オプション引数からjsonファイルを読み込んでパラメータを設定する
    params = argparse_test()

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:" + params["gpu_id"] if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    test_dataset = ArrangeNumDataset(params=params, phase="test")
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4)

    net = create_net(params)

    ys = []
    ypreds = []

    for i in range(params["num_estimate"]):
        print("\n推論: {}/{}".format(i + 1, params["num_estimate"]))
        # ネットワークに重みをロードする
        weight_path = os.path.join("result", params["name"], "weight/weight" + str(i) + ".pth")
        load_weight = torch.load(weight_path, map_location=device)
        net.load_state_dict(load_weight)
        print("ネットワークに重みをロードしました")
        print("-----推論中-----")

        # 推論
        y, ypred = eval_net(net, test_loader, probability=True, device=device)

        ys.append(y.cpu().numpy())
        ypreds.append(ypred.cpu().numpy())

    utils.print_and_save_result(params, ys[0], ypreds)
    utils.save_path_y_ypred(
        test_dataset.file_list,
        ys[0],
        ypreds[0],
        params["labels"],
        os.path.join("result", params["name"], params["test_name"]),
    )
    utils.save_params(params)


if __name__ == "__main__":
    predict()
