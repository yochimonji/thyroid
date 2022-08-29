import copy
import os
import sys
from datetime import datetime

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm


def create_net(params):
    net_params = params["net_params"]
    out_features = len(params["labels"])

    if "resnet" in net_params["name"]:
        if net_params["multi_net"]:
            net = ConcatMultiResNet(
                transfer_learning=net_params["transfer_learning"],
                pretrained=net_params["pretrained"],
                model_name=net_params["name"],
                out_features=out_features,
            )
        else:
            net = CustomResNet(
                transfer_learning=net_params["transfer_learning"],
                pretrained=net_params["pretrained"],
                model_name=net_params["name"],
                out_features=out_features,
            )
    elif "efficientnet" in net_params["name"]:
        net = CustomEfficientNet(
            transfer_learning=net_params["transfer_learning"],
            pretrained=net_params["pretrained"],
            model_name=net_params["name"],
            out_features=out_features,
        )
    return net


class CustomResNet(nn.Module):
    def __init__(self, transfer_learning=True, pretrained=True, model_name="resnet18", out_features=8):
        super().__init__()
        self.transfer_learning = transfer_learning
        self.net = getattr(models, model_name)(pretrained=pretrained)
        fc_input_dim = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(fc_input_dim, out_features))
        self.set_grad()

    def forward(self, x):
        return self.net(x)

    # 最終の全結合層のみ重みの計算をするか否か
    # True：転移学習、False：FineTuning
    def set_grad(self):
        if self.transfer_learning:
            for name, param in self.net.named_parameters():
                # net.parameters()のrequires_gradの初期値はTrueだから
                # 勾配を求めたくないパラメータだけFalseにする
                if not ("fc" in name):
                    param.requires_grad = False

    # optimizerのためのparams,lrのdictのlistを生成する
    def get_params_lr(self, lr_not_pretrained=1e-3, lr_pretrained=1e-4):
        params_not_pretrained = []
        params_pretrained = []
        params_lr = []

        for name, param in self.net.named_parameters():
            if "fc" in name:
                params_not_pretrained.append(param)
            else:
                params_pretrained.append(param)

        params_lr.append({"params": params_not_pretrained, "lr": lr_not_pretrained})
        if not self.transfer_learning:
            params_lr.append({"params": params_pretrained, "lr": lr_pretrained})

        return params_lr


# ResNetの1チャネルのグレースケール用
# CustomResNetは3チャネル用
class CustomResNetGray(nn.Module):
    def __init__(self, transfer_learning=True, pretrained=True, model_name="resnet18", out_features=8):
        super().__init__()
        self.transfer_learning = transfer_learning
        self.net = getattr(models, model_name)(pretrained=pretrained)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_input_dim = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(fc_input_dim, out_features))
        self.set_grad()

    def forward(self, x):
        return self.net(x)

    # 最終の全結合層のみ重みの計算をするか否か
    # True：転移学習、False：FineTuning
    def set_grad(self):
        if self.transfer_learning:
            for name, param in self.net.named_parameters():
                # net.parameters()のrequires_gradの初期値はTrueだから
                # 勾配を求めたくないパラメータだけFalseにする
                if not (("fc" in name) or ("conv1.weight" == name)):
                    param.requires_grad = False

    def get_params_lr(self, lr_not_pretrained=1e-3, lr_pretrained=1e-4):
        params_not_pretrained = []
        params_pretrained = []
        params_lr = []

        for name, param in self.net.named_parameters():
            if ("fc" in name) or ("conv1.weight" == name):
                params_not_pretrained.append(param)
            else:
                params_pretrained.append(param)

        params_lr.append({"params": params_not_pretrained, "lr": lr_not_pretrained})
        if not self.transfer_learning:
            params_lr.append({"params": params_pretrained, "lr": lr_pretrained})

        return params_lr


class ConcatMultiResNet(nn.Module):
    def __init__(self, transfer_learning=True, pretrained=True, model_name="multi-resnet18", out_features=8):
        super().__init__()
        self.transfer_learning = transfer_learning
        self.rgb_feature_net = getattr(models, model_name)(pretrained=pretrained)
        fc_input_dim = self.rgb_feature_net.fc.in_features * 2
        self.rgb_feature_net.fc = nn.Identity()  # 恒等関数に変更
        self.gray_feature_net = copy.deepcopy(self.rgb_feature_net)
        self.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(fc_input_dim, out_features))
        self.set_grad()

    def forward(self, x):
        x_rgb = self.rgb_feature_net(x[:, :3, :, :])
        x_gray = self.gray_feature_net(x[:, 3:, :, :])
        x = torch.cat((x_rgb, x_gray), 1)
        x = self.fc(x)
        return x

    # 最終の全結合層のみ重みの計算をするか否か
    # True：転移学習、False：FineTuning
    def set_grad(self):
        if self.transfer_learning:
            # .parameters()のrequires_gradの初期値はTrueだから
            # 勾配を求めたくないパラメータだけFalseにする
            for net in (self.rgb_feature_net, self.gray_feature_net):
                for param in net.parameters():
                    param.requires_grad = False

    # optimizerのためのparams,lrのdictのlistを生成する
    def get_params_lr(self, lr_not_pretrained=1e-3, lr_pretrained=1e-4):
        params_not_pretrained = []
        params_pretrained = []
        params_lr = []

        for net in (self.rgb_feature_net, self.gray_feature_net):
            for param in net.parameters():
                params_pretrained.append(param)
        for param in self.fc.parameters():
            params_not_pretrained.append(param)

        params_lr.append({"params": params_not_pretrained, "lr": lr_not_pretrained})
        if not self.transfer_learning:
            params_lr.append({"params": params_pretrained, "lr": lr_pretrained})

        return params_lr


class CustomEfficientNet(nn.Module):
    def __init__(self, transfer_learning=True, pretrained=True, model_name="efficientnet-b0", out_features=8):
        super().__init__()
        self.transfer_learning = transfer_learning
        if pretrained:
            self.net = EfficientNet.from_pretrained(model_name)
        else:
            self.net = EfficientNet.from_name(model_name)
        fc_input_dim = self.net._fc.in_features
        self.net._fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(fc_input_dim, out_features))
        self.set_grad()

    def forward(self, x):
        return self.net(x)

    # 最終の全結合層のみ重みの計算をするか否か
    # True：転移学習、False：FineTuning
    def set_grad(self):
        if self.transfer_learning:
            for name, param in self.net.named_parameters():
                # net.parameters()のrequires_gradの初期値はTrueだから
                # 勾配を求めたくないパラメータだけFalseにする
                if not ("fc" in name):
                    param.requires_grad = False

    # optimizerのためのparams,lrのdictのlistを生成する
    def get_params_lr(self, lr_not_pretrained=1e-3, lr_pretrained=1e-4):
        params_not_pretrained = []
        params_pretrained = []
        params_lr = []

        for name, param in self.net.named_parameters():
            if "fc" in name:
                params_not_pretrained.append(param)
            else:
                params_pretrained.append(param)

        params_lr.append({"params": params_not_pretrained, "lr": lr_not_pretrained})
        if not self.transfer_learning:
            params_lr.append({"params": params_pretrained, "lr": lr_pretrained})

        return params_lr


# ネットワークで推論を行う
# 戻り値：本物ラベル、予測ラベル
# loaderの設定によってはシャッフルされているので本物ラベルも返す必要がある
def eval_net(net, loader, probability=False, device="cpu"):
    net = net.to(device)
    net.eval()
    ys = []
    ypreds = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            if probability:
                ypred = net(x)
            else:
                ypred = net(x).argmax(1)
        ys.append(y)
        ypreds.append(ypred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    return ys, ypreds


# ネットワークの訓練を行う
def train_net(net, train_loader, optimizer, loss_fn=nn.CrossEntropyLoss(), epochs=10, device="cpu"):
    dt_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join("./logs", dt_now))

    net = net.to(device)

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.0

        for (x, y) in tqdm(train_loader, total=len(train_loader)):
            x = x.to(device)
            y = y.to(device)
            h = net(x)
            loss = loss_fn(h, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lossは1イテレーションの平均値なので、x.size(0)をかけて足し合わせると合計値になる
            # len(dataset)/batch_sizeが割り切れないときのために必要
            train_loss += loss.item() * x.size(0)
            ypred = h.argmax(1)
            train_acc += (y == ypred).sum()  # 予測があっている数を計算

        # len(train_loader)：イテレーション数
        # len(train_loader.dataset)：データセット数
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)

        print(
            "epoch:{}/{}  train_loss: {:.3f}  train_acc: {:.3f}".format(epoch + 1, epochs, train_loss, train_acc),
            flush=True,
        )
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)

    if not os.path.exists("weight"):
        os.makedirs("weight")
    torch.save(net.state_dict(), "weight/last_weight.pth")

    return net
