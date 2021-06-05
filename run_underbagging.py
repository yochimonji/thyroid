import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from utils.utils import ImageTransform, make_datapath_list, show_wrong_img
from utils.dataset import ArrangeNumDataset
from models import InitResNet, InitEfficientNet, eval_net, train_net

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


batch_size = 128
num_workers = 4
only_fc = True    # 転移学習：True, FineTuning：False
pretrained = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
data_path = "./data/"
label_list = ["Normal", "PTC", "UN", "fvptc", "ftc", "med", "poor", "und"]

train_list = make_datapath_list(data_path+"train")
test_list = make_datapath_list(data_path+"test")

test_dataset = ArrangeNumDataset(test_list,
                                 label_list,
                                 phase="test",
                                 transform=ImageTransform(),
                                 arrange=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

n_estimators = 3
last_weights = []
for i in range(n_estimators):
    print("{}/{}".format(i+1, n_estimators))
    train_dataset = ArrangeNumDataset(train_list, 
                                      label_list, 
                                      phase="train",
                                      transform=ImageTransform(),
                                      arrange="undersampling")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    net = InitResNet(only_fc=only_fc, pretrained=pretrained)
    optimizer = optim.Adam(net.get_params_lr())
    train_net(net(), train_loader, test_loader, optimizer=optimizer,
              epochs=2, device=device)
    last_weights.append(torch.load("weight/last_weight.pth"))

ypreds_vote = []
for weight in last_weights:
    net().load_state_dict(weight)
    ys, ypreds = eval_net(net(), test_loader, device=device)
    ypreds = ypreds.cpu().numpy()
    ypreds_vote.append(ypreds)

ypreds_vote, count = stats.mode(ypreds_vote, axis=0)
ypreds_vote = ypreds_vote.ravel()
ys = ys.cpu().numpy()
print(accuracy_score(ys, ypreds_vote))
print(confusion_matrix(ys, ypreds_vote))
print(classification_report(ys, ypreds_vote,
                            target_names=label_list,
                            digits=3))

# original_test_dataset = ArrangeNumDataset(test_list, label_list)
# show_wrong_img(original_test_dataset, ys, ypreds_vote, indices=None, y=0, ypred=None)