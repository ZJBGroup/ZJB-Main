import pickle
from typing import TYPE_CHECKING

import numpy as np
from traits.api import Array, Int, List, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

if TYPE_CHECKING:
    from nibabel.gifti.gifti import GiftiImage


class Space(Data):
    """
    空间类
    Space用于表示数据所在的空间

    在同一Space中的数据在空间维度具有相同的形状,
    相同的空间维度索引指向相同的空间位置

    Attributes
    ----------
    name : str
        空间的名称。
    shape : List[int]
        空间的形状，用整数列表表示。
    """

    name = Str()

    shape = List(Int)


class SurfaceSpace(Space):
    """
    表面空间类。

    继承自 Space 类，用于表示表面型的空间数据。
    """

    @classmethod
    def from_gii(cls, name: str, left: "GiftiImage | str", right: "GiftiImage | str"):
        """
        从 GiftiImage 文件或路径创建 SurfaceSpace 实例。

        Parameters
        ----------
        name : str
            表面空间的名称。
        left : GiftiImage 或 str
            左半脑的 GiftiImage 实例或文件路径。
        right : GiftiImage 或 str
            右半脑的 GiftiImage 实例或文件路径。

        Returns
        -------
        SurfaceSpace
            创建的 SurfaceSpace 实例。
        """
        from nibabel.gifti.gifti import GiftiImage

        if not isinstance(left, GiftiImage):
            left = GiftiImage.from_filename(left)
        if not isinstance(right, GiftiImage):
            right = GiftiImage.from_filename(right)
        shape = left.darrays[0].data.shape[0] + right.darrays[0].data.shape[0]
        return cls(name=name, shape=[shape])


class VolumeSpace(Space):
    """
    体积空间类。

    继承自 Space 类，用于表示体积型的空间数据。
    """
    pass


class ChannelSpace(Space):
    """
   通道空间类。

   继承自 Space 类，用于表示通道型的空间数据。
   """
    pass


class Surface(Data):
    """
    表面类。

    用于表示表面数据，包括顶点和面。

    Attributes
    ----------
    space : SurfaceSpace
        表面数据所在的空间实例。
    vertices : Array
        表面的顶点数据，浮点型二维数组。
    faces : Array
        表面的面数据，整型二维数组。
    """
    space = Instance(SurfaceSpace)

    vertices = Array(dtype=float, shape=(None, 3))

    faces = Array(dtype=int, shape=(None, 3))

    def save_file(self, file_path):
        """ 将 Surface 实例保存到文件。"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        """从文件加载 Surface 实例。"""
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
        """从 GiftiImage 创建 Surface 实例。"""
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
        """从 NumPy 文件创建 Surface 实例。"""
        result = cls()
        result.vertices = np.load(vertices_file_path)
        result.faces = np.load(faces_file_path)
        return result

    @classmethod
    def from_txt(cls, space: "SurfaceSpace | str", vertices_file_path: str, faces_file_path: str):
        """从 txt 文件创建 Surface 实例。"""
        f = open(vertices_file_path)
        lines = f.readlines()
        rows = len(lines)
        cols = len(lines[0].split())
        vertices = np.zeros((rows, cols), dtype=float)

        # 遍历每一行的数据，并将其转换为浮点型，然后存储到矩阵中
        for i in range(rows):
            data = lines[i].strip().split()
            data = [float(x) for x in data]
            vertices[i, :] = data

        f = open(faces_file_path)
        lines = f.readlines()
        rows = len(lines)
        cols = len(lines[0].split())
        faces = np.zeros((rows, cols), dtype=float)

        for i in range(rows):
            data = lines[i].strip().split()
            data = [float(x) for x in data]
            faces[i, :] = data

        return cls(space=space, vertices=vertices, faces=faces)

    def surface_plot(self, show=False):
        """
        展示表面的三维图像。

        Parameters
        ----------
        show : bool, 可选
            是否立即显示图像。默认为 False，即不立即显示。
        Returns
        -------
        SurfaceViewWidget
            创建的表面视图小部件实例。
        """
        import pyqtgraph as pg

        from zjb.main.visualization.surface_space import SurfaceViewWidget

        pg.mkQApp()
        surface = SurfaceViewWidget()
        surface.setSurface(self)
        if show:
            surface.setCameraParams(elevation=90, azimuth=-90, distance=50)
            surface.show()
            surface.setWindowTitle("SurfacePlot")
            pg.exec()
        return surface


class Volume(Data):
    """
    体素类。

    用于表示体素数据。

    Attributes
    ----------
    space : VolumeSpace
        体素数据所在的空间实例。
    """
    space = Instance(VolumeSpace)
