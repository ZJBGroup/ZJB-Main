from traits.api import Array

from zjb._traits.types import Instance
from zjb.dos.data import Data

from .space import Space


class SpaceCorrelation(Data):
    space = Instance(Space)

    data = Array()