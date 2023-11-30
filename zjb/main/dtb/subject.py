from traits.api import Any, Dict, Str

from zjb.dos.data import Data


class Subject(Data):
    """
    Attributes
    ----------
    name : Str
        被试的名称。
    data : Dict(Str, Any)
        存储与被试相关的数据，其中键为字符串类型，值为任意类型。
    """
    name = Str()

    data = Dict(Str, Any)

    def unbind(self):
        """解除被试与其数据的绑定关系。"""
        if not self._manager:
            return
        for data in self.data.values():
            if isinstance(data, Data):
                data.unbind()
        self._manager.unbind(self)
