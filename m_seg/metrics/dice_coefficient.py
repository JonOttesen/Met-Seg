import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCoefficient(nn.Module):

    def __init__(
        self,
        one_hot_encode: bool = True,
        treshold: float = 0.5,
        from_logits: bool = True,
        ignore_background: bool = True,
        ):
        super().__init__()
        self.one_hot_encode = one_hot_encode
        self.from_logits = from_logits
        self.treshold = treshold
        self.ignore_background = ignore_background

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X (torch.Tensor): prediction shape (batch, C, H, W)
            Y (torch.Tensor): ground truth shape (batch,(C,) H, W)
        returns:
            torch.Tensor: The Dice Coefficient for the input with shape (1, ) if mean else (batch, )
        """
        num_classes = X.shape[1]
        axes = tuple(range(2, len(X.shape)))
        X = X.clone()
        Y = Y.clone()

        if self.from_logits:
            if num_classes > 1:  # If there are more than one class use Softmax
                X = F.log_softmax(X, dim=1).exp()
            else:
                X = F.logsigmoid(X).exp()

        # Set predictions
        if num_classes == 1:
            X[X < self.treshold] = 0
            X[X >= self.treshold] = 1
        else:
            X = X.argmax(dim=1, keepdim=False)
            Y = F.one_hot(Y, num_classes=num_classes).permute(0, 3, 1, 2).type(torch.float32)
            X = F.one_hot(X, num_classes=num_classes).permute(0, 3, 1, 2).type(torch.float32)

        TP = (X*Y).sum(dim=axes)
        FP = (X*(1 - Y)).sum(dim=axes)
        FN = ((1 - X)*Y).sum(dim=axes)

        dice = (2*TP + 1) / (2*TP + FN + FP + 1 + 1e-10)

        if self.ignore_background:
            return dice[:, 1:].mean()
        return dice.mean()
