import pickle
from typing import TYPE_CHECKING

import numpy as np
from traits.api import Array, Int, List, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

if TYPE_CHECKING:
    from nibabel.gifti.gifti import GiftiImage


class Space(Data):
    """Space用于表示数据所在的空间

    在同一Space中的数据在空间维度具有相同的形状,
    相同的空间维度索引指向相同的空间位置
    """

    name = Str()

    shape = List(Int)


class SurfaceSpace(Space):
    @classmethod
    def from_gii(cls, name: str, left: "GiftiImage | str", right: "GiftiImage | str"):
        from nibabel.gifti.gifti import GiftiImage

        if not isinstance(left, GiftiImage):
            left = GiftiImage.from_filename(left)
        if not isinstance(right, GiftiImage):
            right = GiftiImage.from_filename(right)
        shape = left.darrays[0].data.shape[0] + right.darrays[0].data.shape[0]
        return cls(name=name, shape=[shape])


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
    def from_surface_gii(
        cls,
        space: "SurfaceSpace | str",
        left_surf: "GiftiImage | str",
        right_surf: "GiftiImage | str",
    ):
        from nibabel.gifti.gifti import GiftiImage

        if not isinstance(left_surf, GiftiImage):
            left_surf = GiftiImage.from_filename(left_surf)
        if not isinstance(right_surf, GiftiImage):
            right_surf = GiftiImage.from_filename(right_surf)

        if not isinstance(space, SurfaceSpace):
            space = SurfaceSpace.from_gii(space, left_surf, right_surf)
        vertices = np.concatenate(
            [left_surf.darrays[0].data, right_surf.darrays[0].data]
        )
        faces = np.concatenate([left_surf.darrays[1].data, right_surf.darrays[1].data + left_surf.darrays[0].data.shape[0]])

        return cls(space=space, vertices=vertices, faces=faces)

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
