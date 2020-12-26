# 標準ライブラリ
import json
import sys

# 外部ライブラリ
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report

# 自作ライブラリ
from utils import ImageTransform, make_datapath_list, show_wrong_img
from utils.dataset import ArrangeNumDataset, ConcatDataset
from model import CustomResNet, CustomResNetGray, ConcatMultiResNet, CustomEfficientNet, eval_net, train_net


def predict(file_name, probability=False):
    # jsonファイルを読み込む
    f = open("./config/params_"+file_name+".json")
    params = json.load(f)
    f.close()
    data_path = params["data_path"]
    labels = params["labels"]
    batch_size = params["batch_size"]
    img_resize = params["img_resize"]
    dataset_params = params["dataset_params"]
    net_params = params["net_params"]
    label_num = len(labels)

    # GPUが使用可能ならGPU、不可能ならCPUを使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    
    print(labels)

    # テスト画像のファイルリスト取得
    test_list = make_datapath_list(data_path+"test", labels=labels)

    # テストのデータセットを作成する
    test_dataset = ArrangeNumDataset(test_list, 
                                    labels,
                                    phase="val",
                                    transform=ImageTransform(size=img_resize,
                                                             mean=dataset_params["test_mean"],
                                                             std=dataset_params["test_std"],
                                                             grayscale_flag=dataset_params["grayscale_flag"],
                                                             normalize_per_img=dataset_params["normalize_per_img"],
                                                             multi_net=net_params["multi_net"]),
                                    arrange=dataset_params["arrange"])

    print("len(test_dataset)", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2)

    # 使用するネットワークを設定する
    if "resnet" in net_params["name"]:
        if net_params["multi_net"]:
            net = ConcatMultiResNet(transfer_learning=net_params["transfer_learning"],
                                    pretrained=net_params["pretrained"],
                                    model_name=net_params["name"],
                                    out_features=label_num)
        else:
            net = CustomResNet(transfer_learning=net_params["transfer_learning"],
                            pretrained=net_params["pretrained"],
                            model_name=net_params["name"],
                            out_features=label_num)
    elif "efficientnet" in net_params["name"]:
        net = CustomEfficientNet(transfer_learning=net_params["transfer_learning"],
                                 pretrained=net_params["pretrained"],
                                 model_name=net_params["name"],
                                 out_features=label_num)

    # ネットワークに重みをロードする
    load_weights = torch.load("./weight/weight_"+file_name+".pth", map_location=device)
    net.load_state_dict(load_weights)
    print("ネットワークに重みをロードしました")

    ys, ypreds = eval_net(net, test_loader, probability=True, device=device)
    ys = ys.cpu().numpy()
    ypreds = ypreds.cpu().numpy()

    return ys, ypreds

if __name__=="__main__":
    # コマンドライン引数を受け取る
    file_names = sys.argv[1:]

    if len(file_names) == 1:
        ys, ypreds = predict(file_names[0])
    else:
        ypreds = []
        recall_list = []
        for file_name in file_names:
            y, ypred = predict(file_name, probability=True)
            ypreds.append(ypred)
            recall_list.append(recall_score(y, ypred.argmax(axis=1), average="macro"))
        ys = y
        ypreds = np.average(ypreds, axis=0, weights=recall_list).argmax(axis=1)
        print(ypreds.shape)

    print("各腫瘍の感度：")
    print(recall_score(ys, ypreds, average=None))
    print("腫瘍の平均感度：", recall_score(ys, ypreds, average="macro"))
    print(confusion_matrix(ys, ypreds))