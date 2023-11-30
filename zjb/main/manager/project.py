from typing import Any

from traits.api import List, Str

from zjb._traits.types import Instance, TypedInstance
from zjb.dos.data import Data

from ..data.correlation import SpaceCorrelation
from ..dtb.atlas import Atlas
from ..dtb.dtb import DTB, AnalysisResult
from ..dtb.dtb_model import DTBModel
from ..dtb.dynamics_model import DynamicsModel
from ..dtb.subject import Subject
from ..simulation.monitor import Monitor, Raw

ProjectInstance = TypedInstance["Project"]("Project", allow_none=False, module=__name__)  # type: ignore


class Project(Data):
    """
    项目类，用于管理和组织与工作空间中项目相关的所有元素，如子项目、被试、DTB模型、DTB等。

    Attributes
    ----------
    name : Str
        项目的名称。
    parent : ProjectInstance
        父项目的引用。
    children : List(ProjectInstance)
        子项目列表。
    subjects : List(Instance(Subject))
        项目中包含的被试列表。
    models : List(Instance(DTBModel))
        项目中包含的DTB模型列表。
    dtbs : List(Instance(DTB))
        项目中包含的DTB列表。
    data : List
        项目中包含的其他数据，入分析结果等
    """
    name = Str()

    parent = ProjectInstance

    children = List(ProjectInstance)

    subjects = List(Instance(Subject))

    models = List(Instance(DTBModel))

    dtbs = List(Instance(DTB))

    data = List()

    def unbind(self):
        """解除项目与其子元素的绑定关系"""
        if not self._manager:
            return
        for child in self.children:
            child.unbind()
        for subject in self.subjects:
            subject.unbind()
        for model in self.models:
            model.unbind()
        for dtb in self.dtbs:
            dtb.unbind()
        self._manager.unbind(self)

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

    def remove_project(self, project: "Project"):
        """移除一个子项目

        Parameters
        ----------
        project : Project
            要移除的项目

        Raises
        ------
        ValueError
            要移除的项目不属于本项目
        """
        children = self.children
        if project not in children:
            raise ValueError(f"The project does not belong to this project.")
        children.remove(project)
        self.children = children
        project.unbind()

    def remove_subject(self, subject: Subject):
        """移除一个被试

        Parameters
        ----------
        subject : Subject
            要移除的被试

        Raises
        ------
        ValueError
            要移除的被试不属于本项目
        """
        subjects = self.subjects
        if subject not in subjects:
            raise ValueError(f"The subject does not belong to this project.")
        subjects.remove(subject)
        self.subjects = subjects
        subject.unbind()

    def remove_model(self, model: DTBModel):
        """移除一个DTB模型

        Parameters
        ----------
        model : DTBModel
            要移除的DTB模型

        Raises
        ------
        ValueError
            要移除的DTB模型不属于本项目
        """
        models = self.models
        if model not in models:
            raise ValueError(f"The model does not belong to this project.")
        models.remove(model)
        self.models = models
        model.unbind()

    def remove_dtb(self, dtb: DTB):
        """移除一个DTB

        Parameters
        ----------
        dtb : DTB
            要移除的DTB

        Raises
        ------
        ValueError
            要移除的DTB不属于本项目
        """
        dtbs = self.dtbs
        if dtb not in dtbs:
            raise ValueError(f"The dtb does not belong to this project.")
        dtbs.remove(dtb)
        self.dtbs = dtbs
        dtb.unbind()
