from traits.api import Dict, Float, List, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..simulation.monitor import Monitor
from ..simulation.solver import EulerSolver, Solver
from ..trait_types import TraitAny
from .atlas import Atlas
from .dynamics_model import DynamicsModel


class DTBModel(Data):
    name = Str()

    atlas = Instance(Atlas, required=True)

    dynamics = Instance(DynamicsModel, required=True)

    states = Dict(Str, TraitAny)

    parameters = Dict(Str, TraitAny)

    solver = Instance(Solver, EulerSolver)

    monitors = List(Instance(Monitor), required=True)

    t = Float(1000)
