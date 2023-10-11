import pickle

import numpy as np
from traits.api import Array, Float, Int, List, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data


class Space(Data):
    """Space用于表示数据所在的空间

    在同一Space中的数据在空间维度具有相同的形状,
    相同的空间维度索引指向相同的空间位置
    """

    name = Str()

    shape = List(Int)


class SurfaceSpace(Space):
    pass


class VolumeSpace(Space):
    pass


class ChannelSpace(Space):
    pass


class Surface(Data):
    space = Instance(SurfaceSpace)

    vertices = Array(dtype=float, shape=(None, 3))

    faces = Array(dtype=int, shape=(None, 3))

    def save_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls

    @classmethod
    def from_npy(cls, vertices_file_path, faces_file_path):
        result = cls()
        result.vertices = np.load(vertices_file_path)
        result.faces = np.load(faces_file_path)
        return result

    def surface_plot(self, show=False):
        import pyqtgraph as pg

        from zjb.main.visualization.surface_space import SurfaceViewWidget

        pg.mkQApp()
        surface = SurfaceViewWidget(self)
        if show:
            surface.setCameraParams(elevation=90, azimuth=-90, distance=50)
            surface.show()
            surface.setWindowTitle("SurfacePlot")
            pg.exec()
        return surface


class Volume(Data):
    space = Instance(VolumeSpace)
