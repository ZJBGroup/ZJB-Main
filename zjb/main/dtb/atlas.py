import pickle

import numpy as np
from traits.api import Array, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..data.space import Space
from ..trait_types import FloatVector, StrVector


class Atlas(Data):
    name = Str()

    labels = StrVector

    areas = FloatVector

    def save_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls


class RegionSpace(Space):
    atlas = Instance(Atlas, required=True)
