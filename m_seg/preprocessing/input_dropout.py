from typing import Tuple, Dict, Any

import torch
import torchvision
import monai
from monai.config import KeysCollection
import numpy as np


class ImageInputDropout(monai.transforms.MapTransform):
    """
    ***torchvision.Transforms compatible***

    Input-level dropout
    """
    def __init__(self, keys: KeysCollection, prob: float, num_sequences: int):
        """
        Args:
            labels tuple(int): The total number of sequences, not the number of classes
        """
        self.num_sequences = num_sequences
        self.prob = prob
        self.keys = keys

    def __call__(self, data: Dict[Any, np.ndarray]):
        """
        Args:
            tensor (torch.Tensor): Tensor image to be one hot encoded

        Returns:
            torch.Tensor: One hot encoded labels

        """
        d = dict(data)
        probs = torch.rand(self.num_sequences)
        drop = probs < self.prob

        if drop.sum() == self.num_sequences:
            drop[torch.argmax(probs)] = False

        p = torch.sum(drop)/self.num_sequences

        for key in self.key_iterator(d):
            num_pr_seq = int(d[key].shape[0]/self.num_sequences)
            for i, droop in enumerate(drop):
                if droop:
                    for j in range(num_pr_seq):
                        d[key][i*num_pr_seq + j, ..., :] = 0

        for key in self.key_iterator(d):
            d[key] = d[key]*float(1/(1-p))

        return d

    def __repr__(self):
        return self.__class__.__name__ + '(num_sequences={0}, prob={1})'.format(self.num_sequences, self.prob)


class VolumeInputDropout(monai.transforms.MapTransform):
    """
    ***monai.transforms.Transform compatible***

    Input-level dropout
    """
    def __init__(self, keys: KeysCollection, prob: float, num_sequences: int):
        """
        Args:
            labels tuple(int): The total number of sequences, not the number of classes
        """
        super().__init__(keys)
        self.keys = keys
        self.prob = prob
        self.num_sequences = num_sequences

    def __call__(self, data: Dict[Any, np.ndarray]):
        """
        Args:
            tensor (torch.data): data image to be one hot encoded

        Returns:
            torch.data: One hot encoded labels

        """
        d = dict(data)
        probs = torch.rand(self.num_sequences)
        drop = probs < self.prob

        if drop.sum() == self.num_sequences:
            drop[torch.argmax(probs)] = False

        p = torch.sum(drop)/self.num_sequences

        for key in self.key_iterator(d):
            for i, droop in enumerate(drop):
                if droop:
                    d[key][i, ..., :] = 0

        for key in self.key_iterator(d):
            d[key] = d[key]*float(1/(1-p))
            # print(float(1/(1-p)), drop)

        return d

    def __repr__(self):
        return self.__class__.__name__ + '(num_sequences={0}, prob={1})'.format(self.num_sequences, self.prob)
