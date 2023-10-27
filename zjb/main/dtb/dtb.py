import itertools
from typing import Any, Iterable

from traits.api import Dict, Float, List, Str, Union

from zjb._traits.types import Instance, TraitAny
from zjb.doj import Job, generator_job_wrap
from zjb.dos.data import Data

from ..data.correlation import SpaceCorrelation
from ..data.series import RegionalTimeSeries
from ..simulation.simulator import Simulator
from ..trait_types import FloatVector
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
        dynamic_parameters: "dict[str, Any] | None" = None,
    ):
        model = self.model
        if not t:
            t = model.t
        parameters = model.parameters | self.parameters
        for name, parameter in parameters.items():
            if isinstance(parameter, str):
                parameters[name] = self.subject.data[parameter]
        if not dynamic_parameters:
            dynamic_parameters = {}
        if model.dynamic_parameters:
            dynamic_parameters |= model.dynamic_parameters(
                model, self.connectivity, parameters
            )
        parameters |= dynamic_parameters
        simulator = Simulator(
            model=model.dynamics,
            states=model.states,
            parameters=parameters,
            connectivity=self.connectivity,
            solver=model.solver,
            monitors=model.monitors,
            t=t,
        )
        data = simulator()
        result = SimulationResult(
            data=[RegionalTimeSeries(space=model.atlas.space, data=d) for d in data],
            parameters=dynamic_parameters,
        )
        if store_key:
            with self:
                self.data |= {store_key: result}
        return result

    def _pse_result(
        self,
        parameters: dict[str, Iterable[Any]],
        jobs: list[Job[Any, "SimulationResult"]],
        store_key: "str | None" = None,
    ):
        result = PSEResult(
            data=[job.out for job in jobs],
            parameters=parameters,
        )
        if store_key:
            with self:
                self.data |= {store_key: result}
        return result

    @generator_job_wrap
    def pse(
        self,
        parameters: dict[str, Iterable[Any]],
        t: "float | None" = None,
        store_key: "str | None" = None,
    ):
        jobs: list[Job[Any, "SimulationResult"]] = []
        for para_tuple in itertools.product(*parameters.values()):
            para_dict = {name: para for name, para in zip(parameters, para_tuple)}
            job = Job(
                DTB.simulate,
                self,
                t=t,
                store_key=store_key,
                dynamic_parameters=para_dict,
            )
            jobs.append(job)
            yield job
        return Job(DTB._pse_result, self, parameters, jobs)


class SimulationResult(Data):
    data = List(Instance(RegionalTimeSeries))

    parameters = Dict(Str, Union(Float, FloatVector))


class PSEResult(Data):
    data = List(Instance(SimulationResult))

    parameters = Dict(Str, List(Union(Float, FloatVector)))
