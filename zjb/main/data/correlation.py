import numpy as np
from traits.api import Array, Bool, Float, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data
from .space import Space


class SpaceCorrelation(Data):
    space = Instance(Space)

    weights = Array()


class StructuralConnectivity(SpaceCorrelation):
    region_labels = Array(required=False)

    undirected = Bool(required=False)

    tract_lengths = Array(required=False)

    speed = Float(3.0)

    delays = Array(required=False)  # 由tract_lengths与speed更新

    areas = Array(required=False)  # 区域体积，用以计算单位体积下的连通性

    average_weights = Array(required=False)  # 单位体积下的平均连通性，若不提供，则由weights与areas自动进行计算

    @classmethod
    def from_file(
        cls,
        weights_file_path,
        region_labels=0,
        undirected=True,
        tract_lengths=np.array([0]),
        speed=3.0,
        delays=np.array([0]),
        areas=np.array([0]),
        average_weights=np.array([0]),
    ):
        result = StructuralConnectivity()
        result.weights = np.load(weights_file_path)

        if region_labels != 0:
            result.region_labels = region_labels

        result.undirected = undirected

        # 若无指定延迟矩阵，则通过tract_lengths和speed进行计算
        result.tract_lengths = tract_lengths
        result.speed = speed
        if delays != np.array([0]):
            result.delays = delays
        else:
            result.delays = result.tract_lengths / result.speed

        # 若无指定平均连接密度，则通过脑区体积和权重进行计算
        result.areas = areas
        if average_weights != np.array([0]):
            result.average_weights = average_weights
        else:
            result.average_weights = result.weights / result.areas[:, np.newaxis]

        return result

    @property
    def speed(self):
        """返回speed属性"""
        return self._speed

    @property
    def tract_lengths(self):
        """返回tract_lengths属性"""
        return self._tract_lengths

    @property
    def areas(self):
        """返回areas属性"""
        return self._areas

    @property
    def weights(self):
        """返回weights属性"""
        return self._weights

    @speed.setter
    def speed(self, value):
        """当speed发生变化时，延迟矩阵发生变化"""

        self._speed = value

        self.delays = self.tract_lengths / self.speed

    @tract_lengths.setter
    def tract_lengths(self, value):
        """当纤维束长度发生变化时，延迟矩阵发生变化"""

        self._tract_lengths = value

        self.delays = self.tract_lengths / self.speed

    @areas.setter
    def areas(self, value):
        """当areas发生变化时，连通性密度发生变化"""

        self._areas = value

        self.average_weights = self.weights / self.areas

    @weights.setter
    def weights(self, value):
        """当areas发生变化时，连通性密度发生变化"""

        self._weights = value

        self.average_weights = self.weights / self.areas

    def save_file(self, file_path, save_average_weights=True, meta_file=True):
        np.save(file_path, self.weights)  # 保存连接矩阵数据到.npy文件

        # 保存连接矩阵的元数据到同目录下的.txt文件
        if meta_file == True:
            meta_file_path = file_path.replace(".npy", ".txt")
            with open(meta_file_path, "w") as f:
                f.write(f"region_labels: {self.region_labels}\n")
                f.write(f"undirected: {self.undirected}\n")
                f.write(f"tract_lengths: {self.tract_lengths}\n")
                f.write(f"speed: {self.speed}\n")
                f.write(f"delays: {self.delays}\n")
                f.write(f"areas: {self.areas}\n")

        #  保存连接密度矩阵
        if save_average_weights == True:
            average_weights_file_path = file_path.replace(
                ".npy", "_average_weights.npy"
            )
            np.save(average_weights_file_path, self.average_weights)


class FunctionalConnectivity(SpaceCorrelation):
    region_labels = Array(required=False)

    undirected = Bool(required=False)

    areas = Array(required=False)  # 区域体积，用以计算单位体积下的连通性

    average_weights = Array(required=False)  # 单位体积下的平均连通性，若不提供，则由weights与areas自动进行计算

    state = Str()

    @classmethod
    def from_file(
        cls,
        weights_file_path,
        region_labels=0,
        undirected=True,
        areas=np.array([0]),
        average_weights=np.array([0]),
    ):
        result = StructuralConnectivity()
        result.weights = np.load(weights_file_path)

        if region_labels != 0:
            result.region_labels = region_labels

        result.undirected = undirected

        # 若无指定平均连接密度，则通过脑区体积和权重进行计算
        result.areas = areas
        if average_weights != np.array([0]):
            result.average_weights = average_weights
        else:
            result.average_weights = result.weights / result.areas[:, np.newaxis]

        return result

    @property
    def areas(self):
        """返回areas属性"""
        return self._areas

    @property
    def weights(self):
        """返回weights属性"""
        return self._weights

    @areas.setter
    def areas(self, value):
        """当areas发生变化时，连通性密度发生变化"""

        self._areas = value

        self.average_weights = self.weights / self.areas

    @weights.setter
    def weights(self, value):
        """当areas发生变化时，连通性密度发生变化"""

        self._weights = value

        self.average_weights = self.weights / self.areas

    def save_file(self, file_path, save_average_weights=True, meta_file=True):
        np.save(file_path, self.weights)  # 保存连接矩阵数据到.npy文件

        # 保存连接矩阵的元数据到同目录下的.txt文件
        if meta_file == True:
            meta_file_path = file_path.replace(".npy", ".txt")
            with open(meta_file_path, "w") as f:
                f.write(f"region_labels: {self.region_labels}\n")
                f.write(f"undirected: {self.undirected}\n")
                f.write(f"areas: {self.areas}\n")
                f.write(f"state: {self.state}\n")

        #  保存连接密度矩阵
        if save_average_weights == True:
            average_weights_file_path = file_path.replace(
                ".npy", "_average_weights.npy"
            )
            np.save(average_weights_file_path, self.average_weights)
