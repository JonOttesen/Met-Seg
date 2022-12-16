import torch
import torch.nn as nn
import torch.nn.functional as F


class Accuracy(torch.nn.Module):

    def __init__(
        self,
        mean: bool = True,
        treshold: float = 0.5,
        from_logits: bool = False,
        ):
        super().__init__()
        self.mean = mean
        self.treshold = treshold
        self.from_logits = from_logits

    def forward(self, X, Y):
        """
        Args:
            X (torch.Tensor): prediction shape (batch, C, H, W)
            Y (torch.Tensor): ground truth shape (batch, C, H, W)
        returns:
            torch.Tensor: The accuracy for the input with shape (1, ) if mean
        """
        num_classes = X.shape[1]
        X = X.clone()
        Y = Y.clone()

        if self.from_logits:
            if num_classes > 1:  # If there are more than one class use Softmax
                X = F.log_softmax(X, dim=1).exp()
            else:
                X = F.logsigmoid(X).exp()

        # Ensure correct shape, the shape error is most likely in the channel dim
        if len(X.shape) != len(Y.shape):
            Y = Y.unsqueeze(1)

        # Set predictions
        if num_classes == 1:
            X[X < self.treshold] = 0
            X[X >= self.treshold] = 1
        else:
            X = X.argmax(dim=1, keepdim=True)

        accuracy = X == Y  # Where prediction is equal to the ground truth

        if self.mean:
            return torch.mean(accuracy.type(torch.float32))
        else:
            return accuracy

