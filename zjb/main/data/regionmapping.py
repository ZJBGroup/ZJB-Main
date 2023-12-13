from typing import TYPE_CHECKING

import numpy as np
from traits.api import Array

from zjb._traits.types import Instance
from zjb.main.data.space import SurfaceSpace, VolumeSpace
from zjb.main.dtb.atlas import Atlas
from zjb.main.trait_types import RequiredIntVector

from .base import ArrayData

if TYPE_CHECKING:
    from nibabel.gifti.gifti import GiftiImage


class SurfaceRegionMapping(ArrayData):
    """
    表面区域映射类。该类用于创建和处理表面区域映射。

    Attributes
    ----------
    space : SurfaceSpace
        表面空间实例，表示数据所在的空间。
    atlas : Atlas
        图谱实例，用于区域映射。
    data : array[int] shape (n_nodes)
        整型向量表示的映射数据。
    """

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
        """
        从左右脑标签的gii文件构造 SurfaceRegionMapping 实例。

        Parameters
        ----------
        left_label : GiftiImage 或 str
            左脑的标签文件，可以是 GiftiImage 实例或文件路径。
        right_label : GiftiImage 或 str
            右脑的标签文件，可以是 GiftiImage 实例或文件路径。
        space : SurfaceSpace 或 str
            表面空间实例或文件路径。
        atlas : Atlas 或 str
            图谱实例或文件路径。

        Returns
        -------
        SurfaceRegionMapping
            由指定参数构造的 SurfaceRegionMapping 实例。
        """
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
    def from_txt(cls, space: "SurfaceSpace", atlas: "Atlas", path: str):
        """

        Parameters
        ----------
        space : SurfaceSpace
            表面空间实例。
        atlas : Atlas
            图谱实例。
        path : str
            包含映射数据的文本文件路径。

        Returns
        -------
         SurfaceRegionMapping
            由指定参数构造的 SurfaceRegionMapping 实例。
        """
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


class VolumeRegionMapping(ArrayData):
    """
    体素空间映射类。

    该类用于表示体素空间中的区域映射数据。

    Attributes
    ----------
    volume : VolumeSpace
        体素空间的实例。
    atlas : Atlas
        图谱的实例。
    data : array[int], shape (nx, ny, nz)
        包含映射数据的 NumPy 数组。
    """

    volume = Instance(VolumeSpace, required=True)

    atlas = Instance(Atlas, required=True)

    data = Array(dtype=int, shape=(None, None, None), required=True)
