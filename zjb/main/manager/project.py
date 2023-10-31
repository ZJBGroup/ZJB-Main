from typing import Any

from traits.api import List, Str

from zjb._traits.types import Instance, TypedInstance
from zjb.dos.data import Data

from ..data.correlation import SpaceCorrelation
from ..dtb.atlas import Atlas
from ..dtb.dtb import DTB
from ..dtb.dtb_model import DTBModel
from ..dtb.dynamics_model import DynamicsModel
from ..dtb.subject import Subject
from ..simulation.monitor import Monitor, Raw

ProjectInstance = TypedInstance["Project"]("Project", allow_none=False, module=__name__)  # type: ignore


class Project(Data):
    name = Str()

    parent = ProjectInstance

    children = List(ProjectInstance)

    subjects = List(Instance(Subject))

    models = List(Instance(DTBModel))

    dtbs = List(Instance(DTB))

    def available_subjects(self) -> list[Subject]:
        """列出项目中所有可用的被试(包括父项目中的可用被试)

        Returns
        -------
        list[Subject]
            可用被试列表
        """
        subjects = self.subjects
        if parent := self.parent:
            subjects += parent.available_subjects()
        return subjects

    def available_models(self) -> list[DTBModel]:
        """列出项目中所有可用的DTB模型(包括父项目中的可用DTB模型)

        Returns
        -------
        list[Subject]
            可用DTB模型列表
        """
        models = self.models
        if parent := self.parent:
            models += parent.available_models()
        return models

    def add_project(self, name: str, **kwargs: Any) -> "Project":
        """新建并添加一个子项目

        Parameters
        ----------
        name : str
            子项目名
        **kwargs: Any
            子项目的其他特征

        Returns
        -------
        Project
            新建并添加的子项目实例
        """
        child = Project(name=name, parent=self, **kwargs)
        self.children += [child]
        return child

    def add_subject(self, name: str, **kwargs: Any) -> Subject:
        """新建并添加一个被试

        Parameters
        ----------
        name : str
            被试名
        **kwargs: Any
            被是的其他特征

        Returns
        -------
        Subject
            新建并添加的被试实例
        """
        subject = Subject(name=name, **kwargs)
        self.subjects += [subject]
        return subject

    def add_model(
        self,
        name: str,
        atlas: Atlas,
        dynamics: DynamicsModel,
        monitors: "list[Monitor] | None" = None,
        **kwargs: Any,
    ) -> DTBModel:
        """新建并添加一个DTB模型

        Parameters
        ----------
        name : str
            模型名
        atlas : Atlas
            模型所用的图谱
        dynamics : DynamicsModel
            模型所用的动力学
        monitors : list[Monitor] | None, optional
            模型输出结果的监测器列表, None表示使用默认的监测器, by default None
        **kwargs: Any
            DTB模型的其他特征

        Returns
        -------
        DTBModel
            新建并添加的DTB模型实例
        """
        if monitors is None:
            monitors = [Raw(expression=state) for state in dynamics.state_variables]
        model = DTBModel(
            name=name, atlas=atlas, dynamics=dynamics, monitors=monitors, **kwargs
        )
        self.models += [model]
        return model

    def add_dtb(
        self,
        name: str,
        subject: Subject,
        model: DTBModel,
        connectivity: SpaceCorrelation,
        **kwargs: Any,
    ) -> DTB:
        """新建并添加一个DTB

        Parameters
        ----------
        name : str
            DTB名称
        subject : Subject
            DTB关联的被试
        model : DTBModel
            DTB所用的模型
        connectivity : SpaceCorrelation
            DTB所用的连接矩阵
        **kwargs: Any
            DTB的其他特征

        Returns
        -------
        DTB
            新建并添加的DTB实例
        """
        dtb = DTB(
            name=name, subject=subject, model=model, connectivity=connectivity, **kwargs
        )
        self.dtbs += [dtb]
        return dtb
