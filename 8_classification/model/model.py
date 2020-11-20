import torch
from torch import nn
from torchvision.models import resnet18, resnet50, resnet101, resnet152
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


    
# 初期化したネットワークを返却
# set_gradとget_params_lrはもっといい描き方がある気がする
class InitResNet():
    def __init__(self, only_fc=True, pretrained=True):
        self.only_fc = only_fc
    #     self.net = resnet18(pretrained=pretrained)
#         self.net = resnet50(pretrained=pretrained)
        self.net = resnet101(pretrained=pretrained)  # 性能良い
    #     self.net = resnet152(pretrained=pretrained)

        fc_input_dim = self.net.fc.in_features
        self.net.fc = nn.Linear(fc_input_dim, 8)
        
        self.set_grad()
            
    def __call__(self):
        return self.net
    
    # 最終の全結合層のみ重みの計算をするか否か
    # True：転移学習、False：FineTuning
    def set_grad(self):
        if self.only_fc:
            for name, param in self.net.named_parameters():
                # net.parameters()のrequires_gradの初期値はTrueだから
                # 勾配を求めたくないパラメータだけFalseにする
                if not("fc" in name):
                    param.requires_grad = False
                    
    def get_params_lr(self):
        fc_params = []
        not_fc_params = []
        params_lr = []
        
        if self.only_fc:
            for name, param in self.net.named_parameters():
                if "fc" in name:
                    fc_params.append(param)
            params_lr.append({"params": fc_params, "lr": 1e-3})
                    
        else:
            for name, param in self.net.named_parameters():
                if "fc" in name:
                    fc_params.append(param)
                else:
                    not_fc_params.append(param)
            params_lr.append({"params": fc_params, "lr": 1e-3})
            params_lr.append({"params": not_fc_params, "lr": 1e-4})
            
        return params_lr
    
    
class InitEfficientNet():
    def __init__(self, only_fc=True, pretrained=True, model_name="efficientnet-b0"):
        self.only_fc = only_fc
        
        if pretrained:
            self.net = EfficientNet.from_pretrained(model_name)
        else:
            self.net = EfficientNet.from_name(model_name)

        fc_input_dim = self.net._fc.in_features
        self.net._fc = nn.Linear(fc_input_dim, 8)
        self.set_grad()
            
    def __call__(self):
        return self.net
    
    # 最終の全結合層のみ重みの計算をするか否か
    # True：転移学習、False：FineTuning
    def set_grad(self):
        if self.only_fc:
            for name, param in self.net.named_parameters():
                # net.parameters()のrequires_gradの初期値はTrueだから
                # 勾配を求めたくないパラメータだけFalseにする
                if not("fc" in name):
                    param.requires_grad = False
                    
    def get_params_lr(self):
        fc_params = []
        not_fc_params = []
        params_lr = []
        
        if self.only_fc:
            for name, param in self.net.named_parameters():
                if "fc" in name:
                    fc_params.append(param)
            params_lr.append({"params": fc_params, "lr": 1e-3})
                    
        else:
            for name, param in self.net.named_parameters():
                if "fc" in name:
                    fc_params.append(param)
                else:
                    not_fc_params.append(param)
            params_lr.append({"params": fc_params, "lr": 1e-3})
            params_lr.append({"params": not_fc_params, "lr": 1e-4})
            
        return params_lr


# ネットワークで推論を行う
# 戻り値：本物ラベル、予測ラベル
# loaderの設定によってはシャッフルされているので本物ラベルも返す必要がある
def eval_net(net, loader, device="cpu"):
    net = net.to(device)
    net.eval()
    ys = []
    ypreds = []
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            ypred = net(x).argmax(1)
        ys.append(y)
        ypreds.append(ypred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    
    return ys, ypreds


# ネットワークの訓練を行う
def train_net(net, train_loader, val_loader, optimizer,
              loss_fn=nn.CrossEntropyLoss(), epochs=10, device="cpu"):
    
    net = net.to(device)
    writer = SummaryWriter(log_dir="./logs")
    
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
        val_ys, val_ypreds = eval_net(net, val_loader, device=device)
        val_acc = (val_ys == val_ypreds).sum().float() / len(val_loader.dataset)
        
        print("epoch:{}/{}  train_loss: {:.3f}  train_acc: {:.3f}  val_acc: {:.3f}".format(
        epoch+1, epochs, train_loss, train_acc, val_acc), flush=True)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        
    torch.save(net.state_dict(), "weight/last_weight.pth")
    
    return net