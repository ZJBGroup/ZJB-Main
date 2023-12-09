import itertools
import os
from typing import Any, Iterable

import numpy as np
from traits.api import Array, Dict, Float, Int, List, Str, Union

from zjb._traits.types import Instance, TraitAny
from zjb.doj import Job, generator_job_wrap
from zjb.dos.data import Data

from ..data.correlation import SpaceCorrelation
from ..data.series import RegionalTimeSeries
from ..simulation.simulator import NumbaFuncParameter, Simulator
from ..trait_types import FloatVector
from .dtb_model import DTBModel
from .subject import Subject


class DTB(Data):
    """
    DTB 类，代表数字孪生脑。

    Attributes
    ----------
    name : Str
        数字孪生脑的名称。
    subject : Subject
        与数字孪生脑关联的实验对象。
    model : DTBModel
        使用的数字孪生脑模型。
    parameters : Dict
        模型的参数。
    connectivity : SpaceCorrelation
        空间相关性实例。
    data : Dict
        存储仿真或分析的结果数据。
    """

    name = Str()

    subject = Instance(Subject, required=True)

    model = Instance(DTBModel, required=True)

    parameters = Dict(Str, Union(Instance(NumbaFuncParameter), Float, FloatVector, Str))

    connectivity = Instance(SpaceCorrelation, required=True)

    data = Dict(Str, TraitAny)

    def unbind(self):
        """将DTB从数据管理器中解绑"""
        if not self._manager:
            return
        for data in self.data.values():
            if isinstance(data, Data):
                data.unbind()
        self._manager.unbind(self)

    def simulate(
        self,
        t: "float | None" = None,
        store_key: "str | None" = None,
        dynamic_parameters: "dict[str, Any] | None" = None,
    ):
        """
        进行数字孪生脑仿真

        Parameters
        ----------
        t : float | None, 可选
            模拟时间。如果未指定，则使用模型中的默认时间。
        store_key : str | None, 可选
            用于存储模拟结果的键。如果指定，则结果会被存储在 DTB 实例的 data 属性中。
        dynamic_parameters : dict[str, Any] | None, 可选
            动态参数，这些参数在模拟过程中可能会发生变化。

        Returns
        -------
        SimulationResult | None
            如果未指定 store_key，返回模拟结果的 SimulationResult 实例；如果指定了 store_key，则不返回任何内容。
        """
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
            return None
        return result

    def _pse_result(
        self,
        parameters: dict[str, Iterable[Any]],
        jobs: list[Job[Any, "SimulationResult"]],
        store_key: "str | None" = None,
    ):
        """
        处理参数空间探索（PSE）的结果。

        Parameters
        ----------
        parameters : dict[str, Iterable[Any]]
            用于 PSE 的参数。
        jobs : list[Job[Any, "SimulationResult"]]
            PSE 过程中创建的作业列表。
        store_key : str | None, 可选
            用于存储 PSE 结果的键。

        Returns
        -------

        """
        result = PSEResult(
            data=[job.out for job in jobs],
            parameters=parameters,
        )
        if store_key:
            with self:
                self.data |= {store_key: result}
            return None
        return result

    @generator_job_wrap
    def pse(
        self,
        parameters: dict[str, Iterable[Any]],
        t: "float | None" = None,
        store_key: "str | None" = None,
    ):
        """
        执行参数空间探索（PSE）。

        Parameters
        ----------
        parameters : dict[str, Iterable[Any]]
            参数空间探索的的参数。
        t : float | None, 可选
            模拟时间。如果未指定，则使用模型中的默认时间。
        store_key : str | None, 可选
            用于存储 PSE 结果的键。

        Returns
        -------
        Job
            执行 PSE 的作业实例。
        """
        jobs: list[Job[Any, "SimulationResult"]] = []
        for para_tuple in itertools.product(*parameters.values()):
            para_dict = {name: para for name, para in zip(parameters, para_tuple)}
            job = Job(
                DTB.simulate,
                self,
                t=t,
                dynamic_parameters=para_dict,
            )
            jobs.append(job)
            yield job
        return Job(DTB._pse_result, self, parameters, jobs, store_key=store_key)


class SimulationResult(Data):
    """
    仿真结果类。

    存储 DTB 仿真的输出结果。

    Attributes
    ---------
    data : List[Instance(RegionalTimeSeries)]
        仿真结果，每个元素为脑区时间序列。
    parameters : Dict
        仿真过程中使用的参数。
    """

    data = List(Instance(RegionalTimeSeries))

    parameters = Dict(Str, Union(Float, FloatVector))

    def unbind(self):
        """解绑仿真结果中的数据"""
        if not self._manager:
            return
        for data in self.data:
            data.unbind()
        self._manager.unbind(self)


class PSEResult(Data):
    """
    参数空间探索（PSE）结果类。

    存储参数空间探索的输出结果。

    Attributes
    ---------
    data : List[Instance(SimulationResult)]
        PSE的每一组仿真结果。
    parameters : Dict
        PSE中使用的参数。
    """

    data = List(Instance(SimulationResult))

    parameters = Dict(Str, List(Union(Float, FloatVector)))

    def unbind(self):
        """解绑 PSE 结果中的数据。"""
        if not self._manager:
            return
        for data in self.data:
            data.unbind()
        self._manager.unbind(self)


class AnalysisResult(Data):
    """
    分析结果类。

    用于存储各种数据分析的结果。

    Attributes
    ---------
    name : Str
        分析的名称。
    origin : List
        分析的原始数据源。
    data : TraitAny
        分析结果的数据。
    parameters : Dict
        分析过程中使用的分析方法的参数。
    """

    name = Str()

    origin = List()  # 由一个或多个源数据（共同)进行分析

    data = TraitAny()  # 分析结果的数据

    parameters = Dict(Str, Union(Float, FloatVector, Str))  # 分析方法及所使用的参数

    def unbind(self):
        """解绑分析结果中的数据。"""
        if not self._manager:
            return
        for data in self.data:
            data.unbind()
        self._manager.unbind(self)
