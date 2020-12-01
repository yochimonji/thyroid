import random
import json
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from utils.utils import ImageTransform, make_datapath_list, show_wrong_img
from utils.dataset import ArrangeNumDataset
from model.model import InitResNet, InitEfficientNet, eval_net, train_net


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# jsonファイルを読み込んでパラメータを設定する
# jsonから読み込むことでpyファイルの書き換えをしなくてよいのでGitが汚れない
f = open("./config/params.json", "r")
params = json.load(f)
f.close()

data_path = params["data_path"]
labels = params["labels"]
dataset_params = params["dataset_params"]
batch_size = params["batch_size"]
net_params = params["net_params"]
loss_weight_flag = params["loss_weight_flag"]
optim_params = params["optim_params"]

train_list = make_datapath_list(data_path+"train")
test_list = make_datapath_list(data_path+"test")

train_dataset = ArrangeNumDataset(train_list, 
                                  labels,
                                  phase="train",
                                  transform=ImageTransform(grayscale_flag=dataset_params["grayscale_flag"]), 
                                  arrange=dataset_params["arrange"])
test_dataset = ArrangeNumDataset(test_list, 
                                 labels,
                                 phase="val",
                                 transform=ImageTransform(grayscale_flag=dataset_params["grayscale_flag"]),
                                 arrange=dataset_params["arrange"])

# only_fc = True  # 転移学習：True, FineTuning：False
# pretrained = True  # 事前学習の有無
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
print("len(train_dataset)", len(train_dataset))
print("len(test_dataset)", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2)

if "resnet" in net_params["name"]:
    net = InitResNet(only_fc=net_params["only_fc"],
                     pretrained=net_params["pretrained"],
                     model_name=net_params["name"])
elif "efficientnet" in net_params["name"]:
    net = InitEfficientNet(only_fc=net_params["only_fc"],
                           pretrained=net_params["pretrained"],
                           model_name=net_params["name"])
else:
    print("net_params['name']=={} : 定義されていないnameです".format(net_params['name']))
    sys.exit()

if loss_weight_flag:
    weights = torch.tensor(train_dataset.weights).float().cuda()
else:
    weights = None
loss_fn=nn.CrossEntropyLoss(weight=weights)
print("loss_fn.weight:", loss_fn.weight)

if "adam" == optim_params["name"]:
    optimizer = optim.Adam(net.get_params_lr(),
                           weight_decay=optim_params["weight_decay"])
elif "sgd" == optim_params["name"]:
    optimizer = optim.SGD(net.get_params_lr(),
                          momentum=optim_params["momentum"],
                          weight_decay=optim_params["weight_decay"])
else:
    print("optim_params['name']=={} : 定義されていないnameです".format(optim_params['name']))
    sys.exit()
print("optimizer:", optimizer)

train_net(net(), train_loader, test_loader, optimizer=optimizer,
          loss_fn=loss_fn, epochs=10, device=device)

ys, ypreds = eval_net(net(), test_loader, device=device)
ys = ys.cpu().numpy()
ypreds = ypreds.cpu().numpy()
print(accuracy_score(ys, ypreds))
print(confusion_matrix(ys, ypreds))
print(classification_report(ys, ypreds,
                            target_names=labels,
                            digits=3))