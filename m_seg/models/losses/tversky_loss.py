import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum


class TverskyLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        weights: torch.Tensor = None,
        smooth: float = 1.,
        one_hot_encode: bool = True,
        from_logits: bool = True,
        batch: bool = True,
        ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.one_hot_encode = one_hot_encode
        self.from_logits = from_logits
        self.batch = batch

        if weights is not None:
            assert isinstance(weights, torch.Tensor), "The weights must be of type torch.Tensor"
            self.register_buffer('weights', weights.unsqueeze(0))
        else:
            self.register_buffer('weights', None)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):

        # Use big brain log softmax trick to ensure better numerical stability
        if self.from_logits:
            if X.shape[1] > 1:  # If there are more than one class use Softmax
                X = F.log_softmax(X, dim=1).exp()
            else:
                X = F.logsigmoid(X).exp()

        num_classes = X.shape[1]

        # No point in tracking grads in this
        with torch.no_grad():
            if self.one_hot_encode:  # One hot encode for multiclass
                Y = F.one_hot(Y, num_classes=num_classes).permute(0, 3, 1, 2).type(torch.float32)
            elif len(Y.shape) != len(X.shape) and num_classes == 1:
                Y = Y.unsqueeze(1)  # Add channel dimension

        if self.batch:
            axes = (0, ) + tuple(range(2, len(X.shape)))    
        else:
            axes = tuple(range(2, len(X.shape)))

        TP = (X*Y).sum(dim=axes)
        FP = (X*(1 - Y)).sum(dim=axes)
        FN = ((1 - X)*Y).sum(dim=axes)

        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        if isinstance(self.weights, torch.Tensor):
            tversky = (tversky*self.weights).mean(dim=1)

        return -tversky.mean()
        
        
class FocalTverskyLoss(nn.Module):


    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 4./3,
        smooth: float = 1.,
        one_hot_encode: bool = True,
        from_logits: bool = True,
        batch: bool = True,
        ):
        super().__init__()
        self.tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            one_hot_encode=one_hot_encode,
            from_logits=from_logits,
            batch=batch,
            )
        self.gamma = gamma

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        loss = self.tversky(X, Y)
        return torch.pow(1 + loss, self.gamma)
