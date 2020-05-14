import h5py as h5
from pathlib import Path
from pydicom import dcmread
import numpy as np


class TagError(KeyError):
    pass


class Dataset:
    def __init__ (self, file='dataset.hdf5'):
        self.__file = h5.File(file)

        if "counter" not in self.__file.attrs:
            self.__file.attrs['counter'] = 0

        self.__update_studylist()

    def __del__(self):
        self.__file.close()

    def __getitem__(self, key):
        if type(key) == slice or type(key) == int:
            return self.__studylist.__getitem__(key)
        elif type(key) == str:
            return self.__getattr__(self, key)

    def __getattr__(self, tag):
        return self.__studylist.__getattr__(tag)

    def __update_studylist(self):
        self.__studylist = StudyList([item for item in self.__file.values() if type(item) if h5.Dataset])

    def add_dicom_study(self, directory, file_format="*", tags=None):
            directory = Path(directory)
            slices = [dcmread(str(f)) for f in directory.glob(file_format) if f.is_file()]
            slices.sort(key=lambda x: x.SliceLocation)

            image = np.array([slice_.pixel_array + slice_.RescaleIntercept for slice_ in slices], dtype=np.int16)
            scale = float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[0].SliceThickness)

            self.__file.attrs['counter'] += 1
            self.__file.create_dataset(name=f"Study{self.__file.attrs['counter']}", data=image)

            self.__update_studylist()


class StudyList(list):
    def __init__(self, *args, tags=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.__tags = tags

    def __str__(self):
        return f"<StudyList> tags:{self.__tags} length:{len(self)}"

    def __getattr__(self, tag):
        if tag in self.__tags:
            raise TagError("Tag {tag} already present")
        return StudyList([dataset for dataset in self if tag in dataset.attrs['tags']], self.__tags + [tag])


class Study:
    def __init__(self, dataset):
        self.__dataset = dataset
        self.__space = dataset.attrs["space"]
        self.reset()

    def __getitem__(self, key):
        if type(key) == slice or type(key) == int:
            return self.__dataset[key]
        else:
            super().__getitem__(key)
    
    @property
    def space(self):
        return self.__space

    def reset(self):
        self.tags = list(self.__dataset.attrs['tags'])

    def save(self):
        self.__dataset.attrs['tags'] = self.tags