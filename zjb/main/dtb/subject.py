from traits.api import Any, Dict, Str

from zjb.dos.data import Data


class Subject(Data):
    name = Str()

    data = Dict(Str, Any)
