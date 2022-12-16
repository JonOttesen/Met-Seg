from typing import Dict, Optional, Tuple
import random
import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torchvision
import monai
import numpy as np
from tqdm import tqdm

from skimage import exposure

from ..containers import DatasetContainer

from ...logger import get_logger

@contextlib.contextmanager
def temp_seed(seed):
    """
    Source:
    https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)

class VolumeLoader(torch.utils.data.Dataset):
    """
    An iterable datasetloader for the dataset container to make my life easier
    """

    def __init__(self,
                 datasetcontainer: DatasetContainer,
                 transforms: monai.transforms.Compose = None,
                 channels: int = 4,
                 voxel_sizes: Optional[Tuple[int]] = None,
                 sequence_order: Dict[int, Tuple[str, bool]] = None,
                 ):
        """
        Args:
            datasetcontainer: The datasetcontainer that is to be loaded
            train_transforms: Transforms the data is gone through before model input
            train_transforms: Transforms the data is gone through before being ground truths
            open_func: Function to open file
            dataloader_compat: make datasetloader compatible with pytorch datasetloader
        """

        self.datasetcontainer = datasetcontainer
        self.transforms = transforms
        self.channels = channels
        self.sequence_order = sequence_order
        self.voxel_sizes = voxel_sizes

        self.logger = get_logger(name=__name__)

    def __len__(self):
        return len(self.datasetcontainer)

    def brain_normalization(self, volume: np.ndarray):
        indices = volume != 0
        mean = np.mean(volume[indices])
        std = np.std(volume[indices])
        volume[indices] = (volume[indices] - mean)/(std + 1e-5)
        return volume

    def brain_equalization(self, volume: np.ndarray):
        indicies = volume != 0
        volume_eq = np.copy(volume)
        volume_eq[indicies] = exposure.equalize_hist(volume[indicies])
        return volume_eq

    def __getitem__(self, index):
        entry = self.datasetcontainer[index]
        name = Path(entry.segmentation_path).parts[-2]

        images = np.zeros((self.channels, ) + tuple(entry.shape))
        gt = np.expand_dims(entry.open().get_fdata(), axis=0)

        # Order the sequences
        if self.sequence_order is not None:
            for i, instance in enumerate(entry):
                for item, order in self.sequence_order.items():
                    if str(order).lower() == str((instance.sequence_type.lower(), instance.contrast)).lower():
                        img = instance.open()
                        img_seq = img.get_fdata()
                        images[item] = img_seq

        voxels = (img.header["pixdim"][1], img.header["pixdim"][2], img.header["pixdim"][3])

        out = {'image': images, 'mask': gt, 'voxel_sizes': voxels}
        z = self.transforms(out)
        # z['mask'][:] = 1.

        return torch.from_numpy(z['image']), torch.from_numpy(z['mask'])

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