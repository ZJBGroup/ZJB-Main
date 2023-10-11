import pickle

import numpy as np
from traits.api import Array, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..data.space import Space
from ..trait_types import FloatVector, StrVector


class Atlas(Data):
    name = Str()

    labels = StrVector

    areas = FloatVector

    def save_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls

    def atlas_surface_plot(self, surface, surface_region_mapping, show=False):
        import pyqtgraph as pg

        from zjb.main.visualization.surface_space import AtlasSurfaceViewWidget

        pg.mkQApp()
        atlas_surface = AtlasSurfaceViewWidget(self, surface, surface_region_mapping)

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
