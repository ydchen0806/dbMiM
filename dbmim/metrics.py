from __future__ import annotations

import torch


def binary_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (logits.sigmoid() >= threshold)
    tgt = target.bool()
    inter = (pred & tgt).sum().item()
    union = (pred | tgt).sum().item()
    if union == 0:
        return 1.0
    return float(inter / union)


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (logits.sigmoid() >= threshold).float()
    tgt = target.float()
    num = 2.0 * (pred * tgt).sum().item()
    den = pred.sum().item() + tgt.sum().item()
    if den == 0:
        return 1.0
    return float(num / den)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
