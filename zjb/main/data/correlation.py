import numpy as np
from traits.api import ArrayOrNone

from zjb._traits.types import Instance

from ..dtb.atlas import RegionSpace
from .base import ArrayData
from .space import Space


class SpaceCorrelation(ArrayData):
    """空间相关, 特定空间内任意两个点之间的相关程度

    空间相关是一个ArrayLike, 可以使用np.asarray或np.array获取空间相关的数据

    Attrs
    =====
    space: Space
        空间, 定义相关所在的空间
    data: array, shape (*space.shape,) + (*space.shape,)
        相关数组, 空间中任意两个点的相关程度
    """

    space = Instance(Space)


class Connectivity(SpaceCorrelation):
    """连通性, 特定空间中任意两个点之间的连接强度"""


class RegionalConnectivity(Connectivity):
    """脑区连通性, 特指脑区空间的连通性

    Attrs
    =====
    space: RegionSpace
        脑区空间, 定义连通性所在的空间
    """

    space = Instance(RegionSpace)

    @classmethod
    def from_npy(cls, fp: str, space: RegionSpace):
        """从.npy文件导入"""
        matrix = np.load(fp)
        return cls(space=space, data=matrix)


class StructuralConnectivity(RegionalConnectivity):
    """结构连通性, 通过对DTI进行纤维束追踪得到脑区之间的白质连接强度,
    还可选地包含脑区间的纤维束长度信息

    Attrs
    =====
    tract_lengths: np.ndarray[space_shape+space_shape, dtype[float]] | None
        纤维素长度数组数组, 任意两个脑区之间的纤维束长度[mm]
    """

    tract_lengths = ArrayOrNone()

    def delays(self, speed: float = 3.0, dt: "float | None" = None):
        """根据纤维束长度计算延迟矩阵

        Parameters
        ----------
        speed : float, optional
            电导速度[mm/ms], by default 3.0
        dt : float | None, optional
            时间步长[ms], 为None表示不进行离散化, by default None

        Returns
        -------
        np.ndarray[space_shape+space_shape, dtype[float | int]] | None
            _description_
        """
        tract_lengths = self.tract_lengths
        if not tract_lengths:
            return None
        delays = tract_lengths / speed
        if not dt:
            return delays
        return np.rint(delays / dt).astype(int)


class FunctionalConnectivity(RegionalConnectivity):
    """功能连通性, 对各个脑区的脑活动计算相关可以得到脑区之间的功能连通性,
    最常见的是通过对BOLD信号计算Pearson相关得到"""
