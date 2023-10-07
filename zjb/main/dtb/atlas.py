import numpy as np
from traits.api import Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..data.space import Space
from ..trait_types import StrVector


class Atlas(Data):
    name = Str()

    labels = StrVector

    @classmethod
    def from_file(cls, atlas_file_path, atlas_name):
        result = Atlas()
        # 从文件路径中读取脑区列表的.npy文件
        result.labels= np.load(atlas_file_path)

        result.name = atlas_name

        return result


class RegionSpace(Space):
    atlas = Instance(Atlas, required=True)
