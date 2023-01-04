from collections import Counter

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    # 元論文URL: https://arxiv.org/pdf/1708.02002.pdf
    # 実装参考URL: https://kyudy.hatenablog.com/entry/2019/05/20/105526
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, eps: float = 1e-7):
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
    label_list: list[int] | None = None,
    focal_gamma: float = 2.0,
    device: str = "cpu",
) -> torch.nn.CrossEntropyLoss | FocalLoss:
    """損失関数を作成する

    Args:
        loss_name (str): 損失関数名。現在はcrossentropy, focal。
        weight (torch.Tensor | None, optional): クラスごとの重み. Defaults to None.
        focal_gamma (float, optional): FocalLossのハイパーパラメータ. Defaults to 2.0.

    Returns:
        torch.nn.CrossEntropyLoss | FocalLoss: 損失関数
    """
    # 損失関数のweightを定義
    if imbalance and label_list:
        # else:  # if imbalance == "inverse_class_freq":
        weight = calc_inverse_class_freq_weight(label_list).to(device)  # deviceに送らないと動かない
        print("クラスごとのweight:", weight.cpu())
    else:
        weight = None

    # 損失関数を定義
    loss_fn: torch.nn.CrossEntropyLoss | FocalLoss
    if loss_name == "focal":
        loss_fn = FocalLoss(weight=weight, gamma=focal_gamma)
    else:  # elif loss_name == "crossentropy":
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    return loss_fn


def calc_inverse_class_freq_weight(label_list: list[int]) -> torch.Tensor:
    label_count = Counter(label_list)
    sorted_label_count = sorted(label_count.items())
    tensor_label_count = torch.tensor(sorted_label_count)[:, 1]
    weight = len(label_list) / tensor_label_count
    return weight
