from traits.api import Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..data.space import Space
from ..trait_types import StrVector


class Atlas(Data):
    name = Str()

    labels = StrVector


class RegionSpace(Space):
    atlas = Instance(Atlas, required=True)
