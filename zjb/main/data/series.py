import pickle

import numpy as np
from traits.api import Array, Enum, Float, Int, Property, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data
from zjb.main.data.space import Space
from zjb.main.dtb.atlas import Atlas, RegionSpace


class TimeUnit(Enum):
    """时间单位"""

    UNKNOWN = "unknown"
    MILLISECOND = "ms"
    SECOND = "s"


class SpaceSeries(Data):
    space = Instance(Space)

    data = Array()


class TimeSeries(SpaceSeries):
    time_dim = Int(0)  # 时间的维度

    sample_unit = Enum(TimeUnit.MILLISECOND)  # 采样单位

    sample_period = Float(1)  # 采样间隔

    start_time = Float(0.0)  # 开始时间

    time = Property()

    def save_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls

    def _get_time(self):
        time = self.start_time + (np.arange(self.data.shape[self.time_dim]) * self.sample_period)
        return time


class RegionalTimeSeries(TimeSeries):
    """脑区时间序列"""

    space = Instance(RegionSpace)

    atlas = Instance(Atlas)
