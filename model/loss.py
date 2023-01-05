import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    # 元論文URL: https://arxiv.org/pdf/1708.02002.pdf
    # 実装参考URL: https://kyudy.hatenablog.com/entry/2019/05/20/105526
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, eps: float = 1e-7):
        """FocalLossの初期化

        Args:
            weight (torch.Tensor | None, optional): クラスごとにつけることができる重み. Defaults to None.
            gamma (float, optional): FocalLossのハイパーパラメータ. Defaults to 2.0.
            eps (float, optional): 0除算防止のパラメータ. Defaults to 1e-7.
        """
        super().__init__()
        self.weight: torch.Tensor | None = weight
        self.gamma: float = gamma
        self.eps: float = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, weight=self.weight, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma  # focal loss
        return loss.mean()


def create_loss(
    loss_name: str,
    imbalance: str | None = None,
    num_per_class: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    class_balanced_beta: float = 0.999,
    device: str = "cpu",
) -> torch.nn.CrossEntropyLoss | FocalLoss:
    """損失関数を作成する

    Args:
        loss_name (str): 損失関数名。現在はcrossentropy, focal。
        imbalance (str | None, optional): 損失関数で使用するクラスごとの重みの算出方法。 Defaults to None.
        num_per_class: (torch.Tensor | None, optional): クラスごとのデータ数. Defaults to None.
        focal_gamma (float, optional): Focal Lossのハイパーパラメータ。 Defaults to 2.0.
        class_balanced_beta (float, optional): Class-balanced Lossのハイパーパラメータ。 Defaults to 0.999.
        device (str, optional): 使用デバイス。 Defaults to "cpu".

    Returns:
        torch.nn.CrossEntropyLoss | FocalLoss: 損失関数
    """
    # 損失関数のweightを定義
    if num_per_class is not None:
        if imbalance == "class_balanced":
            weight = calc_class_balanced_weight(num_per_class, beta=class_balanced_beta).to(device)
        elif imbalance == "inverse_class_freq":
            weight = calc_inverse_class_freq_weight(num_per_class).to(device)  # deviceに送らないと動かない
        else:
            weight = None
        print("クラスごとのweight:", weight)
    else:
        weight = None

    # 損失関数を定義
    loss_fn: torch.nn.CrossEntropyLoss | FocalLoss
    if loss_name == "focal":
        loss_fn = FocalLoss(weight=weight, gamma=focal_gamma)
    else:  # elif loss_name == "crossentropy":
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    return loss_fn


def calc_inverse_class_freq_weight(num_per_class: torch.Tensor) -> torch.Tensor:
    """逆クラス頻度の重みを計算する

    Args:
        num_per_class (torch.Tensor): クラスごとのデータ数。

    Returns:
        torch.Tensor: 逆クラス頻度の重み。
    """
    weight = torch.sum(num_per_class) / num_per_class
    weight = weight / torch.mean(weight)
    return weight


def calc_class_balanced_weight(num_per_class: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """Class-balanced Lossの重みを計算する
    実装参考URL: https://qiita.com/myellow/items/75cd786a051d097efa81

    Args:
        num_per_class (torch.Tensor): クラスごとのデータ数
        beta (float, optional): Class-balanced Lossのハイパーパラメータ. Defaults to 0.999.

    Returns:
        torch.Tensor: Class-balanced Lossの重み。
    """
    # Class-balanced項を計算する
    weight = (1.0 - beta) / (1.0 - torch.pow(beta, num_per_class))
    weight = weight / torch.mean(weight)
    return weight
