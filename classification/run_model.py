import random
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from utils.utils import ImageTransform, ImageTransformGray, make_datapath_list, show_wrong_img
from utils.dataset import ArrangeNumDataset
from model.model import InitResNet, InitResNetGray, InitEfficientNet, eval_net, train_net


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

data_path = "./data/"

f = open("./config/params.json", "r")
json_data = json.load(f)

labels = json_data["labels"]
dataset_params = json_data["dataset_params"]

train_list = make_datapath_list(data_path+"train")
test_list = make_datapath_list(data_path+"test")

train_dataset = ArrangeNumDataset(train_list, 
                                  labels,
                                  phase="train",
                                  transform=ImageTransform(), 
                                  arrange=dataset_params["arrange"])
test_dataset = ArrangeNumDataset(test_list, 
                                 labels,
                                 phase="val",
                                 transform=ImageTransform(),
                                 arrange=dataset_params["arrange"])

batch_size = 256
num_workers = 2
only_fc = True  # 転移学習：True, FineTuning：False
pretrained = True  # 事前学習の有無
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
print("len(train_dataset)", len(train_dataset))
print("len(test_dataset)", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)

weights = torch.tensor(train_dataset.weights).float().cuda()
# weights = None
loss_fn=nn.CrossEntropyLoss(weight=weights)
print(loss_fn.weight)

net = InitResNet(only_fc=only_fc, pretrained=pretrained)
# net = InitResNetGray()
# net = InitEfficientNet(only_fc=only_fc, pretrained=pretrained,
#                        model_name="efficientnet-b5")

optimizer = optim.Adam(net.get_params_lr())
# optimizer = optim.SGD(net.get_params_lr(), momentum=0.9, weight_decay=5e-5)

train_net(net(), train_loader, test_loader, optimizer=optimizer,
          epochs=10, device=device, loss_fn=loss_fn)

ys, ypreds = eval_net(net(), test_loader, device=device)
ys = ys.cpu().numpy()
ypreds = ypreds.cpu().numpy()
print(accuracy_score(ys, ypreds))
print(confusion_matrix(ys, ypreds))
print(classification_report(ys, ypreds,
                            target_names=labels,
                            digits=3))

# original_test_dataset = ArrangeNumDataset(test_list, labels, phase="val")
# show_wrong_img(original_test_dataset, ys, ypreds, indices=None, y=0, ypred=None)