import numpy as np
from traits.api import Array, Float, Int, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data
from zjb.main.data.space import Space


class SpaceSeries(Data):
    space = Instance(Space)

    data = Array()


class TimeSeries(SpaceSeries):
    time_dim = Int()  # 时间的维度

    sample_unit = Str()  # 采样单位

    sample_period = Float(0)  # 采样间隔

    start_time = Float(0)  # 开始时间

    @classmethod
    def from_file(
        cls,
        timeseries_file_path,
        time_dim=0,
        sample_unit="ms",
        sample_period=1,
        start_time=0,
    ):
        result = TimeSeries()
        # 从文件路径中读取.npy文件
        result.data = np.load(timeseries_file_path)

        # 获取数据的维度
        data_dim = result.data.ndim
        if data_dim > 2:
            raise ValueError(
                "Time series data can only receive data less than 3 dimensions"
            )
        elif data_dim == 1:
            result.time_dim = 0  # 默认一维数据是单脑区的时间序列，但也不能完全排除多个脑区在某一时间点上的时间序列的情况
        result.time_dim = time_dim
        result.sample_unit = sample_unit
        result.sample_period = sample_period
        result.start_time = start_time

        return result

    def save_file(self, file_path, meta_file=True):
        np.save(file_path, self.data)  # 保存时间序列数据到.npy文件

        # 保存时间序列的元数据到同目录下的.txt文件
        if meta_file == True:
            meta_file_path = file_path.replace(".npy", ".txt")
            with open(meta_file_path, "w") as f:
                f.write(f"time_dim: {self.time_dim}\n")
                f.write(f"sample_unit: {self.sample_unit}\n")
                f.write(f"sample_period: {self.sample_period}\n")
                f.write(f"start_time: {self.start_time}\n")
