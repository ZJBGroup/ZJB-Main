import pickle

import numpy as np
from traits.api import Array, Float, Int, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data
from zjb.main.data.space import Space


class SpaceSeries(Data):
    space = Instance(Space)

    data = Array()


class TimeSeries(SpaceSeries):
    time_dim = Int(0)  # 时间的维度

    sample_unit = Str("ms")  # 采样单位

    sample_period = Float(1)  # 采样间隔

    start_time = Float(0.0)  # 开始时间

    def save_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls
