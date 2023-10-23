from traits.api import Dict, Float, Str, Union
from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..data.correlation import SpaceCorrelation
from ..simulation.simulator import Simulator
from ..trait_types import FloatVector, TraitAny
from .dtb_model import DTBModel
from .subject import Subject


class DTB(Data):
    name = Str()

    subject = Instance(Subject, required=True)

    model = Instance(DTBModel, required=True)

    parameters = Dict(Str, Union(Float, FloatVector, Str))

    connectivity = Instance(SpaceCorrelation, required=True)

    data = Dict(Str, TraitAny)

    def simulate(
        self,
        t: "float | None" = None,
        store_key: "str | None" = None,
    ):
        model = self.model
        if not t:
            t = model.t
        parameters = model.parameters | self.parameters
        for name, parameter in parameters.items():
            if isinstance(parameter, str):
                parameters[name] = self.subject.data[parameter]
        if model.dynamic_parameters:
            parameters |= model.dynamic_parameters(model, self.connectivity, parameters)
        simulator = Simulator(
            model=model.dynamics,
            states=model.states,
            parameters=parameters,
            connectivity=self.connectivity,
            solver=model.solver,
            monitors=model.monitors,
            t=t,
        )
        result = simulator()
        if store_key:
            with self:
                self.data |= {store_key: result}
        return result
