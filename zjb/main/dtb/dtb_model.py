from traits.api import Any, Dict, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from .atlas import Atlas
from .dynamics_model import DynamicsModel


class DTBModel(Data):
    name = Str()

    atlas = Instance(Atlas, required=True)

    dynamics = Instance(DynamicsModel, required=True)

    parameters = Dict(Str, Any)
