from traits.api import Any, Dict, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from .atlas import Atlas
from .dynamic_model import DynamicModel


class DTBModel(Data):
    name = Str()

    atlas = Instance(Atlas, required=True)

    dynamic = Instance(DynamicModel, required=True)

    parameters = Dict(Str, Any)
