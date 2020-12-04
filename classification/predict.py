import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from utils.utils import ImageTransform, make_datapath_list, show_wrong_img
from utils.dataset import ArrangeNumDataset
from model.model import InitResNet, InitEfficientNet, eval_net, train_net


data_path = "./data/"
label_list = ["Normal", "PTC HE", "UN", "fvptc", "FTC", "med", "poor", "und"]

test_list = make_datapath_list(data_path+"test")

test_dataset = ArrangeNumDataset(test_list, 
                                 label_list,
                                 phase="val",
                                 transform=ImageTransform(), 
                                 arrange=None)

batch_size =32
num_workers = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
print("len(test_dataset)", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)

net = InitEfficientNet(model_name="efficientnet-b0")
load_path = "weight/last_weight.pth"
load_weights = torch.load(load_path)
net().load_state_dict(load_weights)

ys, ypreds = eval_net(net(), test_loader, device=device)
ys = ys.cpu().numpy()
ypreds = ypreds.cpu().numpy()
print(accuracy_score(ys, ypreds))
print(confusion_matrix(ys, ypreds))
print(classification_report(ys, ypreds,
                            target_names=label_list,
                            digits=3))

# original_test_dataset = ArrangeNumDataset(test_list, label_list, phase="val")
# show_wrong_img(original_test_dataset, ys, ypreds, indices=None, y=0, ypred=None)