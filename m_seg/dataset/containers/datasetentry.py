from pathlib import Path
from typing import Union, Dict, List, Optional
import h5py
from copy import deepcopy

import numpy as np
import nibabel as nib

from ...logger import get_logger
from .datasetinstance import DatasetInstance



class DatasetEntry(object):
    """
    A class used to store information about the different training objects
    This class works like a regular dict
    """

    def __init__(self,
                 instances: List[DatasetInstance] = None,
                 segmentation_path: Union[str, Path] = None,
                 datasetname: str = None,
                 dataset_type: str = None,
                 shape: tuple = None,
                 label: Optional[int] = None,
                 ):
        """An entry for a given patient, contains information about where the sequences, mask, and/or labels

        Args:
            instances (List[DatasetInstance], optional): A list of sequence instances. Defaults to None.
            segmentation_path (Union[str, Path], optional): path where the segmentation is stored. Defaults to None.
            datasetname (str, optional): name of the dataset. Defaults to None.
            dataset_type (str, optional): what kind of dataset is this. Defaults to None.
            shape (tuple, optional): shape of the segmentation. Defaults to None.
            label (Optional[int], optional): label of the entry, only applicable for classification. Defaults to None.
        """

        self.logger = get_logger(name=__name__)

        self.instances = instances if instances is not None else list()

        # Check segmentation path
        if isinstance(segmentation_path, (Path, str)):
            self.segmentation_path = str(segmentation_path)
            if not Path(segmentation_path).is_file():
                self.logger.info('The path: ' + str(segmentation_path))
                self.logger.info('Is not an existing file, are you sure this is the correct path?')
        else:
            self.segmentation_path = segmentation_path

        self.datasetname = datasetname
        self.dataset_type = dataset_type

        self.label = label

        self.shape = shape
        self.score = dict()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.instances[key]
        return self.to_dict()[key]

    def __delitem__(self, index):
        del self.instances[index]

    def __str__(self):
        return str(self.to_dict())

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        return self.__str__()

    def order_instances(self):
        self.instances.sort(key=lambda x: x.sequence_type.lower() + '_' + str(int(x.contrast)))

    def add_instance(self, entry: DatasetInstance):
        """
        Append DataseEntry
        Args:
            info (DatasetEntry): The DatasetEntry to be appended
        """
        self.instances.append(deepcopy(entry))

    def open(self, open_func=None):
        """
        Open the file
        Args:
            open_func (the function to open the file)
        returns:
            the opened file
        """
        if open_func is not None:
            image = open_func(self.segmentation_path)
        else:
            suffix = Path(self.segmentation_path).suffix
            if suffix == '.h5':
                image = self.open_hdf5(self.segmentation_path)
            elif suffix in ['.nii', '.gz']:
                image = self.open_nifti(self.segmentation_path)
            elif suffix in ['.npy', '.npz']:
                image = self.open_numpy(self.segmentation_path)
            else:
                raise TypeError('cannot open file: ', self.segmentation_path)

        return image

    def open_hdf5(self, segmentation_path):
        return h5py.File(segmentation_path, 'r')

    def open_nifti(self, segmentation_path):
        return nib.load(segmentation_path)

    def open_numpy(self, segmentation_path):
        if Path(segmentation_path).suffix in "'.npz":
            images = np.load(segmentation_path)
            return images[list(images.keys())[0]]
        return np.load(segmentation_path)

    def add_score(self, score: Dict[str, float]):
        """
        Add segmentation score to entry for a given slice in volume, can be used for segmentation but also reconstruction
        Args:
            img_slice (int): The slice the score is for (-1 is the entire volume)
            score (Dict[str, float]): Dict of metrics with score
        """
        if self.score:
            self.logger.info('there already exists score for this entry, they are overwritten')
        self.score = deepcopy(score)

    def add_shape(self, open_func=None, shape=None, keyword='data'):
        """
        Add shape to entry
        Args:
            open_func (callable): function for opening file
            shape (tuple): shape of file
            keyword (str): potential keyword for opening file
        """
        if self.segmentation_path is None:
            return
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            img = self.open(open_func=open_func)
            try:
                shape = img.shape
            except:
                shape = img[keyword].shape

            self.shape = shape

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
        entry_dict = {
            'segmentation_path': self.segmentation_path,
            'datasetname': self.datasetname,
            'dataset_type': self.dataset_type,
            'shape': self.shape,
            'score': self.score,
            'label': self.label,
            }

        entry_dict['instances'] = [instance.to_dict() for instance in self.instances]
        return entry_dict

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.segmentation_path = in_dict['segmentation_path']
            self.datasetname = in_dict['datasetname']
            self.dataset_type = in_dict['dataset_type']
            self.shape = in_dict['shape']
            self.score = in_dict['score']
            self.label = in_dict['label']

            for instance in in_dict['instances']:
                self.instances.append(DatasetInstance().from_dict(instance))

        return self
