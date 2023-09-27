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
    model = Instance(DynamicsModel, required=True)

    states = Dict(Str, Union(Float, FloatVector))

    parameters = Dict(Str, Union(Float, FloatVector))

    connectivity = Instance(SpaceCorrelation, required=True)

    solver = Instance(Solver, EulerSolver)

    monitors = List(Instance(Monitor), required=True)

    t = Float(1000)

    def build(self):
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
        if self._simulator is None:  # type: ignore
            self.build()

        states, results = self._simulator(**self._args)
        for name, state in zip(self.states, states):
            self.states[name] = state

        return results
