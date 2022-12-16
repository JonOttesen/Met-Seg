import random
import contextlib
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torchvision
import numpy as np
from tqdm import tqdm

from skimage import exposure, measure

from ..containers import DatasetContainer
from ...preprocessing.input_dropout import ImageInputDropout

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

class DatasetLoader(torch.utils.data.Dataset):
    """
    An iterable datasetloader for the dataset container to make my life easier
    """

    def __init__(self,
                 datasetcontainer: DatasetContainer,
                 transforms: torchvision.transforms = None,
                 slices: int = 0,
                 open_func: callable = None,
                 dataloader_compat: bool = True,
                 dim: int = -1,
                 check_empty_img: bool = True,
                 sequence_order: Dict[int, Tuple[str, bool]] = None,
                 seed: int = 42,
                 input_level_dropout: bool = False,
                 ):
        """
        Args:
            datasetcontainer: The datasetcontainer that is to be loaded
            train_transforms: Transforms the data is gone through before model input
            train_transforms: Transforms the data is gone through before being ground truths
            open_func: Function to open file
            dataloader_compat: make datasetloader compatible with pytorch datasetloader
        """
        self.dim = dim

        self.datasetcontainer = datasetcontainer
        self.transforms = transforms
        self.open_func = open_func
        self.dataloader_compat = dataloader_compat
        self.slices = slices
        self.check_empty_img = check_empty_img
        self.sequence_order = sequence_order
        self.seed = seed

        if input_level_dropout:
            self.input_level_dropout = ImageInputDropout(keys=["image"], num_sequences=len(sequence_order), prob=1/len(sequence_order))
        else:
            self.input_level_dropout = None

        self.logger = get_logger(name=__name__)

        # Checking if dataloader compatibility is enabled
        if dataloader_compat:
            self.logger.info('--------------------------------------------------------------')
            self.logger.info('torch.utils.data.DataLoader compatibility enabled(default=True), '\
                'the last index in shape is assumed to be the slice/image/sample (N, C, H, W).')

            # Checking if all entries have the shape attribute, if not, try to add them.
            if not datasetcontainer.shapes_given():
                self.logger.info('Image shape must be given in entry, for pytorch '\
                    'torch.utils.data.DataLoader compatibility.')
                self.logger.info('Trying to fetch shapes from dataset...')

                # fetching shapes from image files
                datasetcontainer.add_shapes(open_func=open_func)

                # Could not fetch shapes, raise error
                if not datasetcontainer.shapes_given():
                    self.logger.warning('Could not fetch shapes, '\
                        'insert manually, aborting program.')
                    raise AttributeError

                self.logger.info('All shapes fetched from files, will continue.')
            else:
                self.logger.info('All entries have the shape attribute, will continue.')


            # Create a dict that maps image index to file and image in file index
            self.index_to_file_and_image = self._set_index_to_file_and_image(container=datasetcontainer)

            self.logger.info('--------------------------------------------------------------')
        else:
            self.logger.info('Pytorch datasetloader compatibility disabled.\n'\
                'The outputs are therefore (batch, C, H, W)')
            self._index_to_file_and_image = None

    def _set_index_to_file_and_image(self, container):
        index_to_file_and_image = dict()
        self.no_seg = list()
        counter = 0
        for i, entry in tqdm(enumerate(container), total=len(container)):
            images = entry.shape[self.dim]

            if Path(entry.segmentation_path).suffix == ".h5":
                img = entry.instances[0].open()['data'][()]
                seg = entry.open()['data'][()]
            else:
                img = entry.instances[0].open().get_fdata()  # Assumes co-registration
                seg = entry.open().get_fdata()

            if self.dim == -1 or self.dim == len(entry.shape):
                img = np.sum(img, axis=(0, 1))
                seg = np.sum(seg, axis=(0, 1))
            elif self.dim == -2 or self.dim == (len(entry.shape) - 1):
                img = np.sum(img, axis=(0, 2))
                seg = np.sum(seg, axis=(0, 2))
            else:
                img = np.sum(img, axis=(1, 2))
                seg = np.sum(seg, axis=(1, 2))

            for j in range(images):
                if self.check_empty_img:
                    if img[j] != 0:
                        if seg[j] == 0:
                            self.no_seg.append(counter)
                        index_to_file_and_image[counter] = (i, j)
                        counter += 1
                else:
                    if seg[j] == 0:
                        self.no_seg.append(counter)
                    index_to_file_and_image[counter] = (i, j)
                    counter += 1

        return index_to_file_and_image

    def __len__(self):
        if self.dataloader_compat:
            return len(self.index_to_file_and_image)
        else:
            return len(self.datasetcontainer)

    def __getitem__(self, index):
        if self.dataloader_compat:
            index, image_index = self.index_to_file_and_image[index]  # Fetch image (image_index) from volume (index)
        else:
            index = index  # Index corresponds to a file, not image in files
            image_index = ()  # Fetch all images

        entry = self.datasetcontainer[index]
        sequences = len(entry.instances)
        sequence = dict()

        counter = 0
        shape = list(deepcopy(entry.shape))

        del shape[self.dim]

        train = np.zeros(shape=((2*self.slices + 1)*sequences, shape[0], shape[1]))

        stats = dict()
        for instance in entry.instances:
            sequence[str((instance.sequence_type.lower(), instance.contrast))] = list()
            img = instance.open(open_func=self.open_func)
            if Path(instance.image_path).suffix == ".h5":
                img = img['data']
            else:
                img = img.get_fdata()
                
            stats[str((instance.sequence_type.lower(), instance.contrast))] = (instance.mean, instance.std)

            for sliice in list(range(image_index - self.slices, image_index + self.slices + 1)):

                if 0 <= sliice < entry.shape[self.dim]:
                    if self.dim == -1 or self.dim == len(entry.shape):
                        img_slice = img[:, :, sliice]
                    elif self.dim == -2 or self.dim == (len(entry.shape) - 1):
                        img_slice = img[:, sliice]
                    else:
                        img_slice = img[sliice]

                    # img_slice[img_slice != 0] = np.clip(img_slice[img_slice != 0], instance.low_percentile, instance.high_percentile)
                    train[counter] = img_slice

                sequence[str((instance.sequence_type.lower(), instance.contrast))].append(counter)

                counter += 1

            gt = entry.open(open_func=self.open_func)
            if Path(entry.segmentation_path).suffix == ".h5":
                gt = gt['data']
            else:
                gt = gt.get_fdata()

            if self.dim == -1 or self.dim == len(entry.shape):
                gt = gt[:, :, image_index]
            elif self.dim == -2 or self.dim == (len(entry.shape) - 1):
                gt = gt[:, image_index]
            else:
                gt = gt[image_index]
            gt = np.expand_dims(gt, axis=0)

        out = {'image': train, 'mask': gt}

        if self.transforms is not None:
            out = self.transforms(out)
        
        train = torch.from_numpy(out['image'])
        gt = torch.from_numpy(out['mask'])

        # Order the sequences
        if self.sequence_order is not None:
            # Assumes image input
            output = torch.zeros(((2*self.slices + 1)*len(self.sequence_order), train.shape[-2], train.shape[-1]))

            for i, (item, order) in enumerate(self.sequence_order.items()):
                if str(order) in sequence.keys():
                    mean, std = stats[str(order)]
                    mean = mean if mean is not None else 0
                    std = std if std is not None else 1
                    for j, indexx in enumerate(sequence[str(order)]):
                        inp_img = train[indexx]
                        inp_img[inp_img != 0] = (inp_img[inp_img != 0] - mean)/std
                        output[(2*self.slices + 1)*item + j] = inp_img

            if self.input_level_dropout is not None:
                output = self.input_level_dropout({'image': output})['image']
            train = output

        return train, gt

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
