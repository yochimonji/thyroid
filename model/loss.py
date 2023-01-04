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
    loss_name: str, weight: torch.Tensor | None = None, focal_gamma: float = 2.0
) -> torch.nn.CrossEntropyLoss | FocalLoss:
    """損失関数を作成する

    Args:
        loss_name (str): 損失関数名。現在はcrossentropy, focal。
        weight (torch.Tensor | None, optional): クラスごとの重み. Defaults to None.
        focal_gamma (float, optional): FocalLossのハイパーパラメータ. Defaults to 2.0.

    Returns:
        torch.nn.CrossEntropyLoss | FocalLoss: 損失関数
    """
    loss_fn: torch.nn.CrossEntropyLoss | FocalLoss
    if loss_name == "focal":
        loss_fn = FocalLoss(weight=weight, gamma=focal_gamma)
    else:  # elif loss_name == "crossentropy":
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    return loss_fn
