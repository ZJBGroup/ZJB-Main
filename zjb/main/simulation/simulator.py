from pathlib import Path
from typing import Any, Callable

import numpy as np
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
from ..dtb.dynamics_model import DynamicsModel
from ..trait_types import FloatVector
from .monitor import Monitor
from .solver import EulerSolver, Solver

TEMPLATE = Template(
    filename=str(Path(__file__).parent / "_templates" / "simulator.mako")
)


class Simulator(HasPrivateTraits, HasRequiredTraits):
    """
    Attributes
    ----------
    model : Instance(DynamicsModel)
        动力学模型实例，用于仿真。
    states : Dict(Str, Union(Float, FloatVector))
        仿真的状态变量。
    parameters : Dict(Str, Union(Float, FloatVector))
        动力学模型的参数。
    connectivity : Instance(SpaceCorrelation)
        空间相关性连接的实例，描述不同节点间的连接性。
    solver : Instance(Solver)
        求解器实例，用于数值求解仿真的微分方程。
    monitors : List(Instance(Monitor))
        监视器列表，用于在仿真过程中采样数据。
    t : Float
        仿真总时长。
    """
    model = Instance(DynamicsModel, required=True)

    states = Dict(Str, Union(Float, FloatVector))

    parameters = Dict(Str, Union(Float, FloatVector))

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

        self._args = (
            {
                "__t": self.t,
                "__dt": self.solver.dt,
                "__C": self.connectivity.data,
            }
            | self.states
            | self.parameters
        )

        self._env: dict[str, Any] = {}
        self._code = TEMPLATE.render(  # type: ignore
            solver=self.solver, monitors=self.monitors, model=self.model, env=self._env
        )
        exec(self._code, self._env)
        self._simulator: Callable[
            ...,
            tuple[
                tuple[np.ndarray[Any, Any], ...],
                tuple[np.ndarray[Any, Any], ...],
            ],
        ] = self._env["simulator"]

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
