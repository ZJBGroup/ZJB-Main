from traits.api import Any, Dict, Str

from zjb.dos.data import Data


class Subject(Data):
    name = Str()

    data = Dict(Str, Any)

    def unbind(self):
        if not self._manager:
            return
        for data in self.data.values():
            if isinstance(data, Data):
                data.unbind()
        self._manager.unbind(self)
