from traits.api import Str

from zjb._traits.types import Instance
from zjb.dos.data import Data
from zjb.main.data.space import Surface, Volume
from zjb.main.dtb.atlas import Atlas
from zjb.main.trait_types import IntVector


class SurfaceRegionMapping(Data):
    surface = Instance(Surface)

    atlas_name = Str()

    data = IntVector


class VolumeRegionMapping(Data):
    volume = Instance(Volume)

    atlas_name = Str()

    data = IntVector
