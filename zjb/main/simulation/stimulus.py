"""
仿真刺激模块, 用于生成具有不同时空模式的刺激, 可以作为仿真器的参数值
"""

from typing import TYPE_CHECKING, Any

from traits.api import Expression, Float, Property, Union, property_depends_on

from ..trait_types import FloatVector
from .simulator import ExprParameter

if TYPE_CHECKING:
    from numpy import float_
    from numpy.typing import NDArray


def _delegate_parameters_property(name: str, trait: Any = None, **metadata: Any):
    def fget(object: ExprParameter, name: str):
        return object.parameters[name]

    def fset(object: ExprParameter, name: str, value: Any):
        object.parameters[name] = value

    return Property(fget=fget, fset=fset, trait=trait, **metadata)


class Stimulus(ExprParameter):
    """刺激是用于激活或抑制大脑的电流, 可以由植入大脑或放置于头皮的电极直接提供,
    或通过向头部施加磁场来产生感应电流(TMS)

    刺激具有特定的时间模式, 并且会按照特定的空间分布作用到各个脑区:
        - 时间模式是关于时间的函数(表达式);
        - 空间分布可以是数值, 刺激会按照同样的强度作用到所有脑区;
        - 空间分布也可以是长度为脑区数的向量, 该向量对应刺激作用到对应脑区的相对强度

    Attributes
    ----------
    space: float | array[float], shape (n_regions)
        刺激的空间分布, by default 1
    time: str
        刺激的时间模式表达式, 应当是当前仿真时间`__ct`的函数
    """

    expr = Property()

    dependencies = ["__ct"]

    space: "float | NDArray[float_]" = _delegate_parameters_property(
        "space", Union(Float, FloatVector)
    )

    time = Expression("")

    parameters = {"space": 1}

    @property_depends_on("time")
    def _get_expr(self):
        return f"space * ({self.time})" if self.time else "space"


class PulseStimulus(Stimulus):
    """脉冲刺激

    Attributes
    ----------
    amp: float
        脉冲强度, by default 1
    start: float
        脉冲起始时刻, by default 1
    width: float
        脉冲宽度, by default 1
    """

    time = "np.where((__ct >= start) & (__ct < start+width), amp, 0)"

    parameters = {"space": 1, "amp": 1, "start": 1, "width": 1}

    amp: float = _delegate_parameters_property("amp", Float())

    start: float = _delegate_parameters_property("start", Float())

    width: float = _delegate_parameters_property("width", Float())


class SinusoidStimulus(Stimulus):
    """正弦刺激, 形如 :math:`amp * \\sin(2 pi * freq * t + phase) + offset`

    Attributes
    ----------
    amp: float
        刺激强度, by default 1
    freq: float
        刺激频率, by default 1
    phase: float
        刺激相位, by default 0
    offset: float
        直流偏置强度, by default 0
    """

    time = "amp * np.sin(2 * np.pi * freq * __ct + phase) + offset"

    parameters = {"space": 1, "amp": 1, "freq": 1, "phase": 0, "offset": 0}

    amp: float = _delegate_parameters_property("amp", Float)

    freq: float = _delegate_parameters_property("freq", Float)

    phase: float = _delegate_parameters_property("phase", Float)

    offset: float = _delegate_parameters_property("offset", Float)
