from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from mako.template import Template
from traits.api import Dict
from traits.api import Enum as TraitEnum
from traits.api import (
    Expression,
    Float,
    HasPrivateTraits,
    HasRequiredTraits,
    List,
    Property,
    Str,
    Union,
    property_depends_on,
)

from zjb._traits.types import Instance, TraitCallable

from ..dtb.dynamics_model import DynamicsModel
from ..trait_types import ArrayLike, FloatVector
from .monitor import Monitor
from .solver import EulerSolver, Solver

TEMPLATE = Template(
    filename=str(Path(__file__).parent / "_templates" / "simulator.mako")
)

_EXPR_TEMPLATE = Template(
    filename=str(Path(__file__).parent / "_templates" / "expr.mako")
)


class BUILTIN_PARAMETER(str, Enum):  # 使用(str, Enum)替换(StrEnum)以兼容Python3.11之前的版本
    """内建参数, 仿真器内部预先定义的一些参数, 可以在表达式中直接使用这些参数"""

    t = "__t"
    """仿真总时长"""
    dt = "__dt"
    """仿真时间步长, 由求解器定义"""
    C = "__C"
    """脑网络的连接矩阵"""
    C_1 = "__C_1"
    """脑网络连接矩阵的行求和, 等于`np.sum(__C, 1)`"""
    nt = "__nt"
    """仿真的总时间步数, 等于`int(__t / __dt)`"""
    nr = "__nr"
    """脑网络的节点数, 等于`__C.shape[0]`"""
    it = "__it"
    """仿真当前迭代的时间步, 整数, 从0到__nt-1"""
    ct = "__ct"
    """仿真当前迭代的时间, 等于`__dt * __it`"""


class NumbaFuncParameter(HasPrivateTraits, HasRequiredTraits):
    """Numba函数参数

    Attributes
    ----------
    dependencise: list[BUILTIN_PARAMETER]
        函数所依赖的内建参数, by default (,)
    numba_func: Callable[..., float | array[float]]
        `numba.njit`装饰的可执行函数
    """

    dependencies = List(TraitEnum(BUILTIN_PARAMETER))

    numba_func: "Callable[..., float | NDArray[float_]]" = TraitCallable(required=True)

    _arg_str: str = Property()

    @property_depends_on("dependencies")
    def _get__arg_str(self):
        return ",".join(self.dependencies)


class ExprParameter(NumbaFuncParameter):
    """表达式参数

    Attributes
    ----------
    expression: str
        Python表达式
    parameters: dict[str, float | array[float]], shape (n_regions)
        表达式内部的参数字典
    numba_func: Callable[..., float | array[float]], 只读
        numba装饰的可执行函数
    """

    expr = Expression()

    parameters = Dict(Str, Union(Float, FloatVector))

    numba_func: "Callable[..., float | NDArray[float_]]" = Property()

    @property_depends_on(["expression", "dependencies", "parameters"])
    def _get_numba_func(self):
        code = _EXPR_TEMPLATE.render(expr=self)
        env = self.parameters.copy()
        exec(code, env)
        return env["func"]


class Simulator(HasPrivateTraits, HasRequiredTraits):
    """(脑网络模型)仿真器

    Attributes
    ----------
    model : DynamicsModel
        动力学模型实例，用于仿真。
    states : dict[str, float | array[float]], shape (n_regions)
        仿真器当前的状态变量字典
    parameters : dict[str, ExprParameter | float | array[float]], shape (n_regions)
        仿真器当前的参数字典
    connectivity : array_like, shape (n_regions, n_regions)
        脑网络的连接矩阵
    solver : Solver
        求解器实例，用于数值求解仿真的微分方程。
    monitors : list[Monitor]
        监视器列表，用于在仿真过程中采样数据。
    t : float
        仿真总时长。
    """

    model = Instance(DynamicsModel, required=True)

    states = Dict(Str, Union(Float, FloatVector))

    parameters = Dict(Str, Union(Instance(NumbaFuncParameter), Float, FloatVector))

    connectivity = ArrayLike(required=True)

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
            "__C": np.asarray(self.connectivity, dtype=float),
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
            if isinstance(para, NumbaFuncParameter):
                env[f"_{name}_func"] = para.numba_func
            else:
                env[name] = para
        return env


if TYPE_CHECKING:
    from numpy import float_
    from numpy.typing import NDArray

    SimulatorFunction = Callable[
        ...,
        tuple[
            tuple[NDArray[float_], ...],
            tuple[NDArray[float_], ...],
        ],
    ]
