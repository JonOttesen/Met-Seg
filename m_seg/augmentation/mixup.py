import torch
import numpy as np

class MixUP(torch.utils.data.Dataset):
    """A mixup class for a given dataset loader

    Args:
        loader (DatasetLoader): The dataloader to use mixup on
        alpha (float): The mixing parameter for mixup
    """

    def __init__(
        self,
        loader: torch.utils.data.Dataset,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.alpha = alpha
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, i: int) -> torch.Tensor:
        # y is the slice choosen by the dataloader
        x_i, y_i = self.loader[i]
        x_j, y_j = self.loader[np.random.randint(len(self.loader))]
        lam = np.random.beta(self.alpha, self.alpha)
        return lam*x_i + (1-lam)*x_j, lam*y_i + (1-lam)*y_j

    def __iter__(self):
        self.current_index = 0
        self.max_length = len(self)
        return self

    def __next__(self):
        if not self.current_index < self.max_length:
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item
