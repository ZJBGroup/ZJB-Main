import csv
import pickle
from typing import TYPE_CHECKING

from traits.api import Dict, Property, Str

from zjb._traits.types import Instance, TraitAny, TypedInstance
from zjb.dos.data import Data

from ..data.space import Space
from ..trait_types import FloatVector, RequiredStrVector

if TYPE_CHECKING:
    from nibabel.gifti.gifti import GiftiImage


class Atlas(Data):
    """
    脑图谱类。

    用于表示大脑区域的图谱信息。

    Attributes
    ----------
    name : Str
        图谱的名称。
    labels : RequiredStrVector
        图谱中每个脑区的标签。
    areas : FloatVector
        图谱中每个脑区的面积。
    subregions : Dict
        图谱中各脑区的子区域。
    number_of_regions : Property
        图谱中脑区的数量。
    space : TypedInstance
        图谱所在的空间实例。
    """
    name = Str()

    labels = RequiredStrVector

    areas = FloatVector

    subregions = Dict(Str, TraitAny)

    number_of_regions = Property()  # 脑区数量

    space = TypedInstance["RegionSpace"](
        "RegionSpace", allow_none=False, module=__name__
    )

    def _get_number_of_regions(self):
        """ 获取图谱中脑区的数量。"""
        return self.labels.shape[0]

    def _space_default(self):
        """设置图谱默认空间"""
        return RegionSpace(atlas=self)

    def save_file(self, file_path):
        """将 Atlas 实例保存到pickle文件。"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        """从pickle文件加载 Atlas 实例。"""
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls

    @classmethod
    def from_lut(cls, name: str, lut: str):
        """从lut文件创建 Atlas 实例。"""
        labels: list[str] = []
        with open(lut) as f:
            for line in csv.reader(f, delimiter=" ", skipinitialspace=True):
                if not line:
                    continue
                if line[0][0] == "#":
                    continue
                labels.append(line[1])
        # labels[0]应当是'Unknown', 需要删除
        labels = labels[1:]
        return cls(name=name, labels=labels)

    @classmethod
    def from_label_gii(
        cls,
        name: str,
        label: "GiftiImage | str",
    ):
        """从 GiftiImage 创建 Atlas 实例。"""
        from nibabel.gifti.gifti import GiftiImage

        if not isinstance(label, GiftiImage):
            label = GiftiImage.from_filename(label)

        labels: list[str] = list(label.labeltable.get_labels_as_dict().values())
        labels = labels[1:]

        return cls(name=name, labels=labels)

    def atlas_surface_plot(self, surface, surface_region_mapping, show=False):
        """
        展示图谱的脑表面图像。

        Parameters
        ----------
        surface : Surface
            表面数据对象，用于图像展示。
            surface_region_mapping : SurfaceRegionMapping
            表面区域映射对象，用于在图谱上映射区域。
        show : bool, 可选
            是否立即显示图像。默认为 False，即不立即显示。

        Returns
        -------
        AtlasSurfaceViewWidget
            创建的图谱表面视图小部件实例。
        """
        import pyqtgraph as pg

        from zjb.main.visualization.surface_space import AtlasSurfaceViewWidget

        pg.mkQApp()
        atlas_surface = AtlasSurfaceViewWidget()
        atlas_surface.setAtlas(self, surface, surface_region_mapping)

        if show:
            atlas_surface.setCameraParams(elevation=90, azimuth=-90, distance=50)
            atlas_surface.show()
            atlas_surface.setWindowTitle("AtlasSurfacePlot")
            pg.exec()
        return atlas_surface

    def atlas_volume_plot(self):
        pass


class RegionSpace(Space):
    atlas = Instance(Atlas, required=True)

    @classmethod
    def from_atlas(cls, atlas: Atlas):
        """
        从 Atlas 实例创建 RegionSpace 实例。

        Parameters
        ----------
        atlas : Atlas
            图谱实例。

        Returns
        -------
        RegionSpace
            创建的 RegionSpace 实例。
        """
        return atlas.space
