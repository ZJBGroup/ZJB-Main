import csv
import pickle
import re
from typing import TYPE_CHECKING

import numpy as np
from traits.api import Array, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..data.space import Space
from ..trait_types import FloatVector, RequiredStrVector

if TYPE_CHECKING:
    from nibabel.gifti.gifti import GiftiImage


class Atlas(Data):
    name = Str()

    labels = RequiredStrVector

    areas = FloatVector

    def save_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls

    @classmethod
    def from_lut(cls, name: str, lut: str):
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
        from nibabel.gifti.gifti import GiftiImage

        if not isinstance(label, GiftiImage):
            label = GiftiImage.from_filename(label)

        labels: list[str] = list(label.labeltable.get_labels_as_dict().values())
        labels = labels[1:]

        return cls(name=name, labels=labels)

    def atlas_surface_plot(self, surface, surface_region_mapping, show=False):
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
