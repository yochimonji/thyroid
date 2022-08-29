# 標準ライブラリ
import os

# 外部ライブラリ
import torch
from torch.utils.data import DataLoader

# 自作ライブラリ
import utils
from model import create_net, eval_net
from utils.dataset import ArrangeNumDataset


def predict():
    # jsonファイルを読み込んでパラメータを設定する
    # よく呼び出すパラメータを変数に代入
    params = utils.load_params(phase="test")

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    test_dataset = ArrangeNumDataset(params=params, phase="test")
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4)

    net = create_net(params)

    ys = []
    ypreds = []

    for i in range(params["num_estimate"]):
        print("\n推論：{}/{}".format(i+1, params["num_estimate"]))
        # ネットワークに重みをロードする
        weight_path = os.path.join("result", params["name"], "weight/weight"+str(i)+".pth")
        load_weight = torch.load(weight_path, map_location=device)
        net.load_state_dict(load_weight)
        print("ネットワークに重みをロードしました")
        print("-----推論中-----")

        # 推論
        y, ypred = eval_net(net, test_loader, probability=True, device=device)

        ys.append(y.cpu().numpy())
        ypreds.append(ypred.cpu().numpy())

    utils.print_and_save_result(params, ys[0], ypreds)


if __name__ == "__main__":
    predict()
