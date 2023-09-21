from traits.api import Any, Dict, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from .dtb_model import DTBModel
from .subject import Subject


class DTB(Data):
    name = Str()

    subject = Instance(Subject, required=True)

    model = Instance(DTBModel, required=True)

    parameters = Dict(Str, Any)

    data = Dict(Str, Any)
