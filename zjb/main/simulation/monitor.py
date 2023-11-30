from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from mako.template import Template
from traits.api import ABCMetaHasTraits, Float, HasPrivateTraits, Int, Str

TEMPLATE = Template(
    filename=str(Path(__file__).parent / "_templates" / "monitors.mako")
)


class Monitor(HasPrivateTraits, metaclass=ABCMetaHasTraits):
    """
    监视器的抽象基类，用于定义监视器的基本接口。

    Attributes
    ----------
    expression : Str
        监视器中采样的表达式。
    """
    expression = Str(required=True)

    @abstractmethod
    def render_init(self, name: str, env: dict[str, Any]) -> str:
        ...

    @abstractmethod
    def render_sample(self, name: str, env: dict[str, Any]) -> str:
        ...


if TYPE_CHECKING:
    RENDER_FUNC_TYPE = Callable[[Monitor, str, dict[str, Any]], str]


class _TemplateMonitor(Monitor):
    """
    模板监视器类，提供基于Mako模板的监视器实现。

    Attributes
    ----------
    _template_name : str
        模板文件中定义的模板名称。
    """
    _template_name = ""

    def render_init(self, name: str, env: dict[str, Any]) -> str:
        """根据模板渲染初始化代码。"""
        return TEMPLATE.get_def(self._template_name + "_init").render(  # type: ignore
            name=name, monitor=self, env=env
        )  # type: ignore

    def render_sample(self, name: str, env: dict[str, Any]) -> str:
        """根据模板渲染采样代码。"""
        return TEMPLATE.get_def(self._template_name + "_sample").render(  # type: ignore
            name=name, monitor=self, env=env
        )  # type: ignore


class Raw(_TemplateMonitor):
    """原始监测器, 采样仿真的原始数据"""

    _template_name = "raw"


class SubSample(_TemplateMonitor):
    """下采样监测器, 间隔时间步采样"""

    _template_name = "sub_sample"

    # 采样间隔, 每仿真`sample_interval`个时间步采样一次
    sample_interval = Int(10)


class TemporalAverage(_TemplateMonitor):
    """时间平均监测器, 采样固定时间间隔的平均值"""

    _template_name = "temporal_average"

    # 采样间隔, 每仿真`sample_interval`个时间步采样一次
    sample_interval = Int(10)


class BOLD(_TemplateMonitor):
    """BOLD信号监测器, 使用血氧气球模型将采样数据转化为BOLD信号"""

    _template_name = "bold"

    taus = Float(0.65)

    tauf = Float(0.41)

    tauo = Float(0.98)

    alpha = Float(0.32)

    Eo = Float(0.4)

    TE = Float(0.04)

    vo = Float(0.04)

    sample_interval = Int(10000)


MONITOR_DICT = {
    "raw": Raw,
    "sub_sample": SubSample,
    "temporal_average": TemporalAverage,
    "bold": BOLD,
}
