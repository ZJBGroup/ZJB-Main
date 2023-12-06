from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from mako.template import Template
from traits.api import (
    Dict,
    Float,
    HasPrivateTraits,
    HasRequiredTraits,
    List,
    Str,
    Union,
)

from zjb._traits.types import Instance

from ..data.correlation import SpaceCorrelation
from ..dtb.dynamics_model import DynamicsModel, HasExpression
from ..trait_types import FloatVector
from .monitor import Monitor
from .solver import EulerSolver, Solver

TEMPLATE = Template(
    filename=str(Path(__file__).parent / "_templates" / "simulator.mako")
)


class ExprParameter(HasExpression):
    """表达式参数

    Attributes
    ----------
    expression: str
        Python表达式
    dependencies: dict[str, float | array[float]], shape (n_regions)
        表达式依赖的参数字典, 值为参数值
    """

    dependencies = Dict(Str, Union(Float, FloatVector))


class Simulator(HasPrivateTraits, HasRequiredTraits):
    """(脑网络模型)仿真器

    Attributes
    ----------
    model : Instance(DynamicsModel)
        动力学模型实例，用于仿真。
    states : dict[str, float | array[float]], shape (n_regions)
        仿真器当前的状态变量字典
    parameters : dict[str, ExprParameter | float | array[float]], shape (n_regions)
        仿真器当前的参数字典
    connectivity : array_like, shape (n_regions, n_regions)
        脑网络的连接矩阵
    solver : Solver
        求解器实例，用于数值求解仿真的微分方程。
    monitors : List(Instance(Monitor))
        监视器列表，用于在仿真过程中采样数据。
    t : Float
        仿真总时长。
    """

    model = Instance(DynamicsModel, required=True)

    states = Dict(Str, Union(Float, FloatVector))

    parameters = Dict(Str, Union(Instance(ExprParameter), Float, FloatVector))

    connectivity = Instance(SpaceCorrelation, required=True)

    solver = Instance(Solver, EulerSolver)

    monitors = List(Instance(Monitor), required=True)

    t = Float(1000)

    def build(self):
        """
        构建仿真所需的代码和环境。这个方法将根据动力学模型的状态变量、参数和解算器设置来准备仿真的执行环境。
        """
        for name in self.model.state_variables:
            self.states.setdefault(name, 0)
        for name, parameter in self.model.parameters.items():
            self.parameters.setdefault(name, parameter)

        self._args = {
            "__t": self.t,
            "__dt": self.solver.dt,
            "__C": self.connectivity.data,
        } | self.states

        self._env: dict[str, Any] = self._update_env_from_parameter({})
        self._code = TEMPLATE.render(  # type: ignore
            simulator=self,
            solver=self.solver,
            monitors=self.monitors,
            model=self.model,
            env=self._env,
        )
        exec(self._code, self._env)
        self._simulator: "SimulatorFunction" = self._env["simulator"]

    def __call__(self):
        """
        执行仿真并返回结果。如果仿真环境尚未构建，首先调用 build 方法构建环境。

        Returns
        -------
        results
            仿真结果，通常包括各个状态变量和监视器采集的数据。
        """
        if self._simulator is None:  # type: ignore
            self.build()

        states, results = self._simulator(**self._args)
        for name, state in zip(self.states, states):
            self.states[name] = state

        return results

    def _update_env_from_parameter(self, env: "dict[str, Any]"):
        """根据仿真器参数更新编译所需的环境变量"""
        for name, para in self.parameters.items():
            if not isinstance(para, ExprParameter):
                env[name] = para
                continue
            for dname, dependency in para.dependencies.items():
                if dname in env:
                    raise KeyError(
                        f"Dependency {dname} of parameter {name} is duplicated"
                    )
                env[dname] = dependency
        return env


if TYPE_CHECKING:
    from numpy.typing import NDArray

    SimulatorFunction = Callable[
        ...,
        tuple[
            tuple[NDArray[Any], ...],
            tuple[NDArray[Any], ...],
        ],
    ]
