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
    """动态参数类，用于表示数字孪生脑模型中被监听的动态参数"""
    @abstractmethod
    def __call__(
        self,
        model: "DTBModel",
        connectivity: SpaceCorrelation,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        动态参数。

        Parameters
        ----------
        model : DTBModel
            使用的数字孪生脑。
        connectivity : SpaceCorrelation
            模型使用的脑区间的连接。
        parameters : dict[str, Any]
            初始参数集合。

        Returns
        -------
        dict[str, Any]
            计算后的动态参数集合。
        """
        ...


class DTBModel(Data):
    """
    DTB模型类，即数字孪生脑模型。

    Attributes
    ----------
    name : Str
        模型的名称。
    atlas : Instance(Atlas)
        模型使用的脑图谱。
    dynamics : Instance(DynamicsModel)
        模型所使用的动力学模型。
    states : Dict(Str, Union(Float, FloatVector))
        模型的状态变量及初值。
    parameters : Dict(Str, Union(Float, FloatVector, Str))
        模型的参数及初值。
    dynamic_parameters : OptionalInstance(DynamicParameter)
        动态参数（可选）。
    solver : Instance(Solver, EulerSolver)
        用于求解模型的求解器。
    monitors : List(Instance(Monitor))
        模型监视器列表。
    t : Float
        仿真时间。
    """
    name = Str()

    atlas = Instance(Atlas, required=True)

    dynamics = Instance(DynamicsModel, required=True)

    states = Dict(Str, Union(Float, FloatVector))

    parameters = Dict(Str, Union(Float, FloatVector, Str))

    dynamic_parameters = OptionalInstance(DynamicParameter)

    solver = Instance(Solver, EulerSolver)

    monitors = List(Instance(Monitor), required=True)

    t = Float(1000)
