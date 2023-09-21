from traits.api import Array

from zjb._traits.types import Instance
from zjb.dos.data import Data

from .space import Space


class SpaceSeries(Data):
    space = Instance(Space)

    data = Array()


class TimeSeries(SpaceSeries):
    pass
