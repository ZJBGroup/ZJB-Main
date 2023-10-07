from abc import abstractmethod
from typing import Any

from traits.api import Dict, Float, List, Str, Union

from zjb._traits.types import Instance, OptionalInstance
from zjb.dos.data import Data

from ..data.correlation import SpaceCorrelation
from ..simulation.monitor import Monitor
from ..simulation.solver import EulerSolver, Solver
from ..trait_types import FloatVector
from .atlas import Atlas
from .dynamics_model import DynamicsModel


class DynamicParameter:
    @abstractmethod
    def __call__(
        self,
        model: "DTBModel",
        connectivity: SpaceCorrelation,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class DTBModel(Data):
    name = Str()

    atlas = Instance(Atlas, required=True)

    dynamics = Instance(DynamicsModel, required=True)

    states = Dict(Str, Union(Float, FloatVector))

    parameters = Dict(Str, Union(Float, FloatVector, Str))

    dynamic_parameters = OptionalInstance(DynamicParameter)

    solver = Instance(Solver, EulerSolver)

    monitors = List(Instance(Monitor), required=True)

    t = Float(1000)
