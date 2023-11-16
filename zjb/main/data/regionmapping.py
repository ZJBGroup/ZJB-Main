from typing import TYPE_CHECKING

import numpy as np
from traits.api import Array

from zjb._traits.types import Instance
from zjb.dos.data import Data
from zjb.main.data.space import SurfaceSpace, VolumeSpace
from zjb.main.dtb.atlas import Atlas
from zjb.main.trait_types import RequiredIntVector

if TYPE_CHECKING:
    from nibabel.gifti.gifti import GiftiImage


class SurfaceRegionMapping(Data):
    space = Instance(SurfaceSpace, required=True)

    atlas = Instance(Atlas, required=True)

    data = RequiredIntVector

    @classmethod
    def from_label_gii(
        cls,
        left_label: "GiftiImage | str",
        right_label: "GiftiImage | str",
        space: "SurfaceSpace | str",
        atlas: "Atlas | str",
    ):
        from nibabel.gifti.gifti import GiftiImage

        if not isinstance(left_label, GiftiImage):
            left_label = GiftiImage.from_filename(left_label)
        if not isinstance(right_label, GiftiImage):
            right_label = GiftiImage.from_filename(right_label)

        if not isinstance(space, SurfaceSpace):
            space = SurfaceSpace.from_gii(space, left_label, right_label)
        if not isinstance(atlas, Atlas):
            atlas = Atlas.from_label_gii(atlas, left_label)
        data = np.concatenate([left_label.darrays[0].data, right_label.darrays[0].data])

        return cls(space=space, atlas=atlas, data=data)

    @classmethod
    def from_txt(cls, space: "SurfaceSpace | str", atlas: "Atlas | str", path: str):
        f = open(path)
        lines = f.readlines()
        rows = len(lines)
        cols = len(lines[0].split())
        mat = np.zeros((rows, cols), dtype=float)

        # 遍历每一行的数据，并将其转换为浮点型，然后存储到矩阵中
        for i in range(rows):
            data = lines[i].strip().split()  # 去掉每行的换行符，并用空格分割数据
            data = [float(x) for x in data]  # 将数据转换为浮点型
            mat[i, :] = data  # 将数据赋值给矩阵的第i行
        mat = mat.squeeze()

        return cls(space=space, atlas=atlas, data=mat)


class VolumeRegionMapping(Data):
    volume = Instance(VolumeSpace, required=True)

    atlas = Instance(Atlas, required=True)

    data = Array(dtype=int, shape=(None, None, None), required=True)
