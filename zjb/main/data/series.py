import pickle
from enum import Enum as ZJBEnum

import numpy as np
from mne.io import RawArray
from traits.api import Enum, Float, Int, Property

from zjb._traits.types import Instance
from .base import ArrayData
from .space import SurfaceSpace
from ..data.space import Space
from ..dtb.atlas import RegionSpace


class TimeUnit(ZJBEnum):
    """
    时间单位枚举类。

    这个枚举类定义了不同的时间单位。

    Attributes
    ----------
    UNKNOWN : str
        未知时间单位。
    MILLISECOND : str
        毫秒。
    SECOND : str
        秒。
    """

    UNKNOWN = "unknown"
    MILLISECOND = "ms"
    SECOND = "s"


class SpaceSeries(ArrayData):
    """
    空间序列类。

    该类用于表示与空间相关的数据序列。

    Attributes
    ----------
    space : Space
        空间实例。
    data : array
        包含空间序列数据的 NumPy 数组。
    """

    space = Instance(Space)


class TimeSeries(SpaceSeries):
    """
    时间序列类。

    该类继承自 SpaceSeries，用于表示随时间变化的空间数据序列。

    Attributes
    ----------
    time_dim : int
        数据中代表时间的维度。
    sample_unit : TimeUnit
        采样的时间单位。
    sample_period : float
        采样间隔时间。
    start_time : float
        序列的开始时间。
    time : Property
        根据开始时间、采样间隔和时间维度计算的时间数组。
    """

    time_dim = Int(0)  # 时间的维度

    sample_unit = Enum(TimeUnit)  # 采样单位

    sample_period = Float(1)  # 采样间隔

    start_time = Float(0.0)  # 开始时间

    time = Property()

    def save_file(self, file_path):
        """
        将 TimeSeries 实例保存到pickle文件。

        Parameters
        ----------
        file_path : str
            要保存文件的路径。
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        """
        从pickle文件加载 TimeSeries 实例。

        Parameters
        ----------
        file_path : str
            要加载的文件路径。

        Returns
        -------
        TimeSeries
            从文件中加载的 TimeSeries 实例。
        """
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls

    def _get_time(self):
        """
        计算并返回时间数组。

        Returns
        -------
        np.ndarray
            时间序列的 NumPy 数组。
        """
        time = self.start_time + (
            np.arange(self.data.shape[self.time_dim]) * self.sample_period
        )
        return time


class RegionalTimeSeries(TimeSeries):
    """
    脑区时间序列类。

    该类继承自 TimeSeries，用于表示随时间变化的脑区数据序列。

    Attributes
    ----------
    space : RegionSpace
        表示脑区的空间实例。
    """

    space = Instance(RegionSpace)


class VertexalTimeSeries(TimeSeries):
    """
    顶点时间序列类。

    该类继承自 TimeSeries，用于表示随时间变化的顶点数据序列。

    Attributes
    ----------
    space : SurfaceSpace
        表示皮层的空间实例。
    """

    space = Instance(SurfaceSpace)


class MNEsimSeries(ArrayData):
    """MNE仿真的数据"""

    rawarray = Instance(RawArray)
