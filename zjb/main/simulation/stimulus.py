"""
仿真刺激模块, 用于生成具有不同时空模式的刺激, 可以作为仿真器的参数值
"""

from typing import TYPE_CHECKING, Callable

import numba as nb
import numpy as np
from traits.api import Array, Float, Int, Property, Union, property_depends_on

from ..trait_types import FloatVector
from .simulator import NumbaFuncParameter

if TYPE_CHECKING:
    from typing import Callable

    from numpy import float_
    from numpy.typing import NDArray


class Stimulus(NumbaFuncParameter):
    """刺激是用于激活或抑制大脑的电流, 可以由植入大脑或放置于头皮的电极直接提供,
    或通过向头部施加磁场来产生感应电流(TMS)

    刺激具有特定的时间模式, 并且会按照特定的空间分布作用到各个脑区:
        - 时间模式是关于时间的函数;
        - 空间分布可以是数值, 刺激会按照同样的强度作用到所有脑区;
        - 空间分布也可以是长度为脑区数的向量, 该向量对应刺激作用到对应脑区的相对强度

    Attributes
    ----------
    space: float | array[float], shape (n_regions)
        刺激的空间分布, by default 1
    numba_func: Callable[[float], float | array[float]], shape (n_regions)
        根据时间生成具有空间分布的刺激的numba函数
    """

    space: "float | NDArray[float_]" = Union(Float(1), FloatVector)  # type: ignore

    dependencies = ["__ct"]

    numba_func: "Callable[[float], float | NDArray[float_]]" = Property()

    def _get_numba_func(self):
        space = self.space
        time_func = self.make_time_func()

        @nb.njit(inline="always")
        def _numba_func(t: float):
            return space * time_func(t)

        return _numba_func

    def make_time_func(self) -> "Callable[[float], float]":
        """创建时间模式函数, 必须返回一个`numba.njit`编译的函数

        返回的函数接收一个参数t, 返回刺激强度
        """
        raise NotImplementedError


class PulseStimulus(Stimulus):
    """脉冲刺激, 继承自 :py:class:`Stimulus`

    Attributes
    ----------
    amp: float
        脉冲强度, by default 1
    start: float
        脉冲起始时刻, by default 1
    width: float
        脉冲宽度, by default 1
    """

    amp = Float(1)

    start = Float(1)

    width = Float(1)

    numba_func = Property()

    @property_depends_on(["space", "amp", "start", "width"])
    def _get_numba_func(self):
        space = self.space
        amp, start, width = self.amp, self.start, self.width

        @nb.njit(inline="always")
        def _numba_func(t: float):
            if start <= t < start + width:
                return space * amp
            return space * 0

        return _numba_func


class NCyclePulseStimulus(PulseStimulus):
    """N周期脉冲刺激, 继承自 :py:class:`PulseStimulus`

    Attributes
    ----------
    period: float
        脉冲周期, by default 2
    count: int
        脉冲周期数, 小于等于0表示无限周期, by default 0
    """

    period = Float(2)

    count = Int(0)

    numba_func = Property()

    @property_depends_on(["space", "amp", "start", "width", "phase", "period", "count"])
    def _get_numba_func(self):
        space = self.space
        amp, start, width, period, count = (
            self.amp,
            self.start,
            self.width,
            self.period,
            self.count,
        )

        @nb.njit(inline="always")
        def _numba_func(t: float):
            if t < start:
                return space * 0
            if count > 0 and t > start + period * count:
                return space * 0
            if (t - start) % period < width:
                return space * amp
            return space * 0

        return _numba_func


class CustomPulseStimulus(Stimulus):
    """自定义脉冲刺激, 由任意多个自定义脉冲序列组成

    Attributes
    ----------
    series: array[float], shape (n_pulse, 3)
        脉冲序列, 每一行由一个脉冲的(起始时刻, 终止时刻, 脉冲强度)组成,
        同一时刻存在的多个脉冲会相互叠加, 脉冲持续时间不包含终止时刻

    Examples
    --------
    >>> s = CustomPulseStimulus(series=[[1, 3, 1], [4, 5, 2], [0.5, 9.5, 0.1]])
    >>> f = s.numba_func
    >>> xs = np.linspace(0, 10, 11)
    >>> print([f(x) for x in xs])
    [0.0, 1.1, 1.1, 0.1, 2.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
    """

    series = Array(float, (None, 3))

    numba_func = Property()

    @property_depends_on(["space", "series"])
    def _get_numba_func(self):
        space = self.space
        series = self.series

        @nb.njit(inline="always")
        def _numba_func(t: float):
            amp = np.sum(series[(t >= series[:, 0]) & (t < series[:, 1]), 2])
            return space * amp

        return _numba_func


class SinusoidStimulus(Stimulus):
    """正弦刺激, 形如 :math:`amp * \\sin(2 \\pi * freq * t + phase) + offset`,
    继承自 :py:class:`Stimulus`

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

    amp = Float(1)

    freq = Float(1)

    phase = Float(0)

    offset = Float(0)

    numba_func = Property()

    @property_depends_on(["space", "amp", "freq", "phase", "offset"])
    def _get_numba_func(self):
        space = self.space
        amp, freq, phase, offset = self.amp, self.freq, self.phase, self.offset

        @nb.njit(inline="always")
        def _numba_func(t: float):
            return space * amp * np.sin(2 * np.pi * freq * t + phase) + offset

        return _numba_func


class GaussianStimulus(Stimulus):
    """高斯函数刺激, 形如 :math:`amp * \\exp(- \\left(x - \\mu\\right) ^ 2 /
    \\left(2 \\sigma^2\\right)) + offset`, 继承自 :py:class:`Stimulus`

    Attributes
    ----------
    amp: float
        刺激强度, by default 1
    mu: float
        高斯函数中心, by default 1
    sigma: float
        高斯函数标准差, by default 0.5
    offset: float
        直流偏置强度, by default 0
    """

    amp = Float(1)

    mu = Float(2)

    sigma = Float(0.5)

    offset = Float(0)

    numba_func = Property()

    @property_depends_on(["space", "amp", "mu", "sigma", "offset"])
    def _get_numba_func(self):
        space = self.space
        amp, mu, sigma, offset = self.amp, self.mu, self.sigma, self.offset

        @nb.njit(inline="always")
        def _numba_func(t: float):
            return space * amp * np.exp(-((t - mu) ** 2) / (2 * sigma**2)) + offset

        return _numba_func
