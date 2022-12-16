import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1707.03237.pdf
    Based on this https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    """

    def __init__(
        self,
        weight: float,
        reduction: str = "mean"
        ):
        super().__init__()


        self.weight = weight
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, X: torch.Tensor, Y: torch.Tensor):

        with torch.no_grad():
            weights = torch.ones_like(Y) + Y*(self.weight - 1)

        loss = self.bce(X, Y)

        weighted_loss = loss*weights

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss
