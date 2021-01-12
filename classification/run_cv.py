import itertools
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

from utils.utils import ImageTransform, make_datapath_list, show_wrong_img
from utils.dataset import ArrangeNumDataset
from model.model import InitResNet, InitEfficientNet, eval_net, train_net

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


data_path = "./data/"
label_list = ["Normal", "PTC HE", "UN", "fvptc", "FTC", "med", "poor", "und"]
file_list = make_datapath_list(data_path+"train")
dataset = ArrangeNumDataset(file_list, label_list, phase="train",
                            arrange=None, transform=None)

ys = []
ypreds = []
val_indices_after_skf = []
batch_size = 128
num_workers = 8
only_fc = True    # 転移学習：True, FineTuning：False
pretrained = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

for cv_num, (train_indices, val_indices) in enumerate(skf.split(
    dataset, dataset.labels)):
    
    print("交差検証：{}/{}".format(cv_num+1, skf.get_n_splits()))
    
    train_list = [file_list[i] for i in train_indices]
    val_list = [file_list[i] for i  in val_indices]
    
    train_dataset = ArrangeNumDataset(train_list,
                                      label_list,
                                      phase="train",
                                      arrange=None,
                                      transform=ImageTransform())
    val_dataset = ArrangeNumDataset(val_list,
                                    label_list,
                                    phase="test",
                                    arrange=None,
                                    transform=ImageTransform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    net = InitEfficientNet(only_fc=only_fc, pretrained=pretrained, model_name="efficientnet-b3")
    optimizer = optim.Adam(net.get_params_lr())
    weights = torch.tensor(train_dataset.weights).float().cuda()
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    
    train_net(net(), train_loader, val_loader, optimizer=optimizer,
              epochs=2, device=device, loss_fn=loss_fn)
    ys_ypreds = eval_net(net(), val_loader, device=device)
    ys.append(ys_ypreds[0])
    ypreds.append(ys_ypreds[1])
    val_indices_after_skf.append(val_indices)
    
ys = torch.cat(ys).cpu().numpy()
ypreds = torch.cat(ypreds).cpu().numpy()
val_indices_after_skf = list(itertools.chain.from_iterable(val_indices_after_skf))

print("accuracy_score:", accuracy_score(ys, ypreds))
print(confusion_matrix(ys, ypreds))
print(classification_report(ys, ypreds,
                            target_names=label_list,
                            digits=3))

# original_test_dataset = ArrangeNumDataset(train_list, label_list)
# show_wrong_img(original_test_dataset, ys, ypreds,
#                indices=val_indices_after_skf, y=0, ypred=None)