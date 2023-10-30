from abc import abstractmethod
from pathlib import Path
from typing import Any

from mako.template import Template
from traits.api import ABCMetaHasTraits, Dict, Float, HasPrivateTraits, Str

from ..dtb.dynamics_model import DynamicsModel

TEMPLATE = Template(filename=str(Path(__file__).parent / "_templates" / "solvers.mako"))


class Solver(HasPrivateTraits, metaclass=ABCMetaHasTraits):
    dt = Float(0.1)

    noises = Dict(Str, Float)

    @abstractmethod
    def render(self, model: DynamicsModel, env: dict[str, Any]) -> str:
        ...


class EulerSolver(Solver):
    def render(self, model: DynamicsModel, env: dict[str, Any]) -> str:
        return TEMPLATE.get_def("euler").render(solver=self, model=model, env=env)  # type: ignore


class HenuSolver(Solver):
    def render(self, model: DynamicsModel, env: dict[str, Any]) -> str:
        return TEMPLATE.get_def("henu").render(solver=self, model=model, env=env)  # type: ignore


SOLVER_DICT = {
    "Euler": EulerSolver,
    "Henu": HenuSolver,
}
