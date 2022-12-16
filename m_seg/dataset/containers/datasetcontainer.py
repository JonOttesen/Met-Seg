import json
import random
import contextlib
import os
import glob

from pathlib import Path
from typing import Union, List, Dict
from copy import deepcopy

from tqdm import tqdm

from .datasetentry import DatasetEntry
from .datasetinstance import DatasetInstance
from .datasetinfo import DatasetInfo
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


class DatasetContainer(object):
    """
    Method to save and store the information about the dataset
    This includes the shape, sequence type, locations, post/pre contrast etc.
    This means that all files can be stored at the same place or separately,
    it is cleaner to handle than just lumping files together in folders.

    And it is iterable and savable
    """

    def __init__(self,
                 info: List[DatasetInfo] = None,
                 entries: List[DatasetEntry] = None):
        """
        Args:
            info (list(DatasetInfo)): list of information about the dataset used
            entries (list(DatasetEntries)): list of the entries, i.e., the files of the dataset
        """

        self.logger = get_logger(name=__name__)
        self.info = info if info is not None else list()
        self.entries = entries if entries is not None else list()

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)

    def __delitem__(self, index):
        del self.entries[index]

    def __str__(self):
        return str(self.to_dict())

    def order(self):
        self.order_entries()
        self.order_instances()

    def order_entries(self):
        self.entries.sort(key=lambda x: str(x.segmentation_path).lower())

    def order_instances(self):
        for entry in self.entries:
            entry.order_instances()

    def split(self, seed: int, split: float = 0.5):
        """
        Randomly split the dataset
        Args:
            seed (int): The seed used for the random split
            split (float): The split fraction
        returns:
            The two split dataset containers
        """
        dataset = deepcopy(self)
        dataset.shuffle(seed=seed)

        split_1 = DatasetContainer()
        split_2 = DatasetContainer()
        for info in self.info:
            split_1.add_info(deepcopy(info))
            split_2.add_info(deepcopy(info))

        split_number = split*len(dataset)

        for i, entry in enumerate(dataset):
            i += 1
            if i <= split_number:
                split_1.add_entry(deepcopy(entry))
            else:
                split_2.add_entry(deepcopy(entry))
        return split_1, split_2

    def k_fold(self, folds: int = 5, fold: int = 1, seed: int = 42):

        dataset = deepcopy(self)
        dataset.shuffle(seed=seed)

        valid = DatasetContainer()
        train = DatasetContainer()
        for info in self.info:
            valid.add_info(deepcopy(info))
            train.add_info(deepcopy(info))

        split = int(len(dataset)/folds)

        splits = list(range((fold - 1)*split, fold*split))

        for i, entry in enumerate(dataset):
            if i in splits:
                valid.add_entry(deepcopy(entry))
            else:
                train.add_entry(deepcopy(entry))
        return train, valid

    def shuffle(self, seed=None):
        """
        Shuffles the entries, used for random training
        Args:
            seed (int): The seed used for the random shuffle
        """
        with temp_seed(seed):
            random.shuffle(self.entries)

    def info_dict(self):
        """
        returns:
            dict version of the list of DatasetInfo
        """
        info_dict = dict()
        for inf in self.info:
            info_dict[inf.datasetname] = inf.to_dict()

        return info_dict

    def copy(self):
        return deepcopy(self.entries)

    def add_info(self, info: DatasetInfo):
        """
        Append DatasetInfo
        Args:
            info (DatasetInfo): The DatasetInfo to be appended
        """
        self.info.append(deepcopy(info))

    def add_entry(self, entry: DatasetEntry):
        """
        Append DataseEntry
        Args:
            info (DatasetEntry): The DatasetEntry to be appended
        """
        self.entries.append(deepcopy(entry))

    def add_shapes(self, open_func=None, shape=None, keyword='data'):
        """
        Fetches the shapes of the images for each entry
        Args:
            open_func (callable): The function used to open the files
            shape (tuple): The shape of the entry, this can be image size
            keyword: Potential key used for hdf5 files
        """
        for entry in tqdm(self):
            entry.add_shape(open_func=open_func, shape=shape, keyword=keyword)
            for instance in entry.instances:
                instance.add_shape(open_func=open_func, shape=shape, keyword=keyword)

    def shapes_given(self):
        """
        Checks if all entries have shapes in the DatasetContainer
        """
        for entry in self:
            if entry.shape is None:
                return False
            else:
                return True

    def shapes(self):
        return list(set([tuple(entry['shape']) for entry in self]))

    def sequences(self):
        return list(set([entry['sequence_type'] for entry in self]))

    def keys(self):
        return self.to_dict().keys()

    def to_dict(self):
        """
        returns:
            dict version of DatasetContainer
        """
        container_dict = dict()
        container_dict['info'] = [inf.to_dict() for inf in self.info]
        container_dict['entries'] = [entry.to_dict() for entry in self.entries]
        return container_dict

    def from_dict(self, in_dict):
        """
        Appends data into DatasetContainer from dict
        Args:
            in_dict (dict): Dict to append data from, meant to be used to recover when loading from file
        """
        for inf in in_dict['info']:
            self.info.append(DatasetInfo().from_dict(inf))
        for entry in in_dict['entries']:
            self.entries.append(DatasetEntry().from_dict(entry))

    def to_json(self, path: Union[str, Path]):
        """
        Save DatasetContainer as json file
        Args:
            path (str, Path): path where DatasetContainer is saved
        """
        path = Path(path)
        suffix = path.suffix
        if suffix != '.json':
            raise NameError('The path must have suffix .json not, ', suffix)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(self.to_dict(), outfile, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        """
        Load DatasetContainer from file
        Args:
            path (str, Path): Path to load from
        returns:
            The DatasetContainer from file
        """
        with open(path) as json_file:
            data = json.load(json_file)
        new_container = cls()
        new_container.from_dict(data)
        return new_container

    def folder_directory(
        self,
        path: Union[str, Path],
        seg_name: str,
        datasetname: str,
        dataset_type: str,
        source: str = 'Data source',
        dataset_description: str = 'A small description of the dataset',
        ):

        info = DatasetInfo(
            datasetname=datasetname,
            dataset_type=dataset_type,
            source=source,
            dataset_description=dataset_description
            )
        self.add_info(info=info)

        subdirs = os.scandir(str(path))

        for folder in subdirs:
            file = glob.glob(str(Path(path) / Path(folder)) + "/*6seq.npy")[0]

            entry = DatasetEntry(
                datasetname=datasetname,
                dataset_type=dataset_type,
                )
            entry.segmentation_path=str(file)

            self.add_entry(entry=entry)

        # self.add_shapes()

        return self

    def from_folder(
        self,
        path: Union[str, Path],
        datasetname: str,
        dataset_type: str,
        to_seq: Dict[str, List[str]],
        check_contrast: List[str],
        source: str = 'Data source',
        dataset_description: str = 'A small description of the dataset',
        ):
        """
        Creates a dataset container from a folder filled with folders filled with data
        Assumes the end folder only contains datafiles, all folders will be seperated into different
        entries.
        Args:
            path (str, Path): Path to fastMRI data
            datasetname (str): Name of dataset
            dataset_type (str): The type of dataset this is
            source (str): The source of the dataset (fastMRI)
            dataset_description (str): description of dataset
        returns:
            DatasetContainer filled with data paths
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError('path argument is {}, expected type is pathlib.Path or str'.format(type(path)))

        if not isinstance(to_seq, dict):
            raise TypeError('to_seq argument is {}, expected type is dict'.format(type(to_seq)))

        path = path.absolute()

        info = DatasetInfo(
            datasetname=datasetname,
            dataset_type=dataset_type,
            source=source,
            dataset_description=dataset_description
            )

        self.add_info(info=info)

        walker = os.walk(path)

        for walk in walker:

            # Check if there are no sub-directories
            if not walk[1]:
                folder = Path(walk[0])

                entry = DatasetEntry(
                    datasetname=datasetname,
                    dataset_type=dataset_type,
                    )

                for file in walk[-1]:
                    # Check if the file is of correct type
                    if Path(file).suffix not in ['.nii', '.gz', '.npz', '.npy', '.h5']:
                        continue

                    file_path = folder / Path(file)
                    filename = Path(file).name.lower()  # Lower case filename

                    for key, items in to_seq.items():
                        items = items if isinstance(items, list) else list(items)

                        if any(item.lower() in filename for item in items):
                            sequence = key
                            break
                        else:
                            sequence = ""

                    if sequence.lower() == "seg":
                        entry.segmentation_path=str(file_path)
                    else:
                        contrast = any(item.lower() in filename for item in check_contrast)
                        entry.add_instance(
                            DatasetInstance(
                                image_path=str(file_path),
                                sequence_type=sequence,
                                field_strength=None,
                                contrast=contrast,
                                shape=None,
                                ))

                self.add_entry(entry=entry)

        self.add_shapes()

        return self

    def ELITE(
        self,
        path: Union[str, Path],
        datasetname: str,
        dataset_type: str,
        source: str = 'ELITE',
        dataset_description: str = 'ELITE data',
        sequence_statistics: bool = False,
        ):
        to_seq = {
            "seg": ["seg"],
            "BRAVO": ["bravo"],
            "FLAIR": ["FLAIR"],
            "T1": ["T1"],
            "T2": ["T2"],
        }
        check_contrast = ["ce", "gd", "_c.", "_c_"]

        container = self.from_folder(
            path=path,
            datasetname=datasetname,
            dataset_type=dataset_type,
            to_seq=to_seq,
            check_contrast=check_contrast,
            source=source,
            dataset_description=dataset_description,
        )
        container.order()
        if sequence_statistics:
            for entry in tqdm(container):
                for instance in entry:
                    instance.add_statistics()
                    
        return container

    def BraTS(
        self,
        path: Union[str, Path],
        datasetname: str,
        dataset_type: str,
        source: str = 'BraTS',
        dataset_description: str = 'Data for BraTS challenge',
        sequence_statistics: bool = False,
        ):
        to_seq = {
            "seg": ["seg"],
            "T1": ["t1"],
            "T2": ["t2"],
            "FLAIR": ["flair"]
        }
        check_contrast = ["ce"]
        container = self.from_folder(
            path=path,
            datasetname=datasetname,
            dataset_type=dataset_type,
            to_seq=to_seq,
            check_contrast=check_contrast,
            source=source,
            dataset_description=dataset_description,
        )
        container.order()
        if sequence_statistics:
            for entry in tqdm(container):
                for instance in entry:
                    instance.add_statistics()

        return container
