from typing import Union, Dict, List, Optional
from copy import deepcopy
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from ...logger import get_logger


class DatasetInstance(object):
    """
    A class used to store information about the different training objects
    This class works like a regular dict
    """

    def __init__(self,
                 image_path: Union[str, Path, List[Union[str, Path]]] = None,
                 sequence_type: Optional[str] = None,
                 field_strength: float = None,
                 contrast: bool = None,
                 shape: tuple = None,
                 mean: float = None,
                 std: float = None,
                 ):
        """
        Args:
            image_path (str, Path, list): The path where the data is stored
            sequence_type (str): The sequence type for MRI
            field_strength (float): Field strength of the scan
            contrast (bool): contrast in this scan
            shape (tuple): The shape of the data
            mean (float): mean value for this scan
            std (float): std for this scan
        """

        self.logger = get_logger(name=__name__)

        # Check image path
        if isinstance(image_path, (Path, str)):
            self.image_path = str(image_path)
            if not Path(image_path).is_file():
                self.logger.info('The path: ' + str(image_path))
                self.logger.info('Is not an existing file, are you sure this is the correct path?')
        else:
            self.image_path = image_path

        self.sequence_type = sequence_type
        self.field_strength = field_strength

        if not isinstance(contrast, bool) and contrast is not None:
            raise TypeError('The variable pre_contrast ', contrast, ' need to be boolean')

        self.contrast = contrast
        self.shape = shape
        self.mean = mean
        self.std = std

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def open(self, open_func=None):
        """
        Open the file
        Args:
            open_func (the function to open the file)
        returns:
            the opened file
        """
        if open_func is not None:
            image = open_func(self.image_path)
        else:
            suffix = Path(self.image_path).suffix
            if suffix == '.h5':
                image = self.open_hdf5(self.image_path)
            elif suffix in ['.nii', '.gz']:
                image = self.open_nifti(self.image_path)
            elif suffix in ['.npy', '.npz']:
                image = self.open_numpy(self.image_path)
            else:
                raise TypeError('cannot open file: ', self.image_path)

        return image

    def open_hdf5(self, image_path):
        return h5py.File(image_path, 'r')

    def open_nifti(self, image_path):
        return nib.load(image_path)

    def open_numpy(self, image_path):
        if Path(image_path).suffix in "'.npz":
            images = np.load(image_path)
            return images[list(images.keys())[0]]
        return np.load(image_path)

    def add_shape(self, open_func=None, shape=None, keyword='data'):
        """
        Add shape to entry
        Args:
            open_func (callable): function for opening file
            shape (tuple): shape of file
            keyword (str): potential keyword for opening file
        """
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            img = self.open(open_func=open_func)
            try:
                shape = img.shape
            except:
                shape = img[keyword].shape

            self.shape = shape
    
    def add_statistics(self, keyword='data'):
        # Add image statistics such as mean and std
        suffix = Path(self.image_path).suffix
        suffix = Path(self.image_path).suffix
        if suffix == '.h5':
            image = self.open_hdf5(self.image_path)[keyword][()]
        elif suffix in ['.nii', '.gz']:
            image = self.open_nifti(self.image_path).get_fdata()
        elif suffix in ['.npy', '.npz']:
            image = self.open_numpy(self.image_path)
        else:
            raise TypeError('cannot open file: ', self.image_path)
        image = np.float64(image)
        
        self.mean = np.mean(image[image != 0])
        self.std = np.std(image[image != 0])

    def keys(self):
        """
        dict keys of class
        """
        return self.to_dict().keys()

    def to_dict(self) -> dict:
        """
        returns:
            dict format of this class
        """
        return {'image_path': self.image_path,
                'sequence_type': self.sequence_type,
                'field_strength': self.field_strength,
                'contrast': self.contrast,
                'shape': self.shape,
                'mean': str(self.mean) if self.mean is not None else self.mean,
                'std': str(self.std) if self.std is not None else self.std,
                }

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.image_path = in_dict['image_path']
            self.sequence_type = in_dict['sequence_type']
            self.field_strength = in_dict['field_strength']
            self.contrast = in_dict['contrast']
            self.shape = in_dict['shape']
            self.mean = in_dict['mean']
            if isinstance(self.mean, str):
                self.mean = float(self.mean)
            self.std = in_dict['std']
            if isinstance(self.std, str):
                self.std = float(self.std)

        return self
