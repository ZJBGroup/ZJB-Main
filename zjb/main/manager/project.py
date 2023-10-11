from traits.api import List, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..dtb.dtb import DTB
from ..dtb.dtb_model import DTBModel
from ..dtb.subject import Subject

ProjectInstance: "Project" = Instance("Project", module=__name__)  # type: ignore


class Project(Data):
    name = Str()

    parent: "Project" = ProjectInstance  # type: ignore

    children: list["Project"] = List(ProjectInstance)  # type: ignore

    subjects = List(Instance(Subject))

    models = List(Instance(DTBModel))

    dtbs = List(Instance(DTB))

    def add_project(self, name: str):
        child = Project(name=name, parent=self)
        self.children += [child]
        return child
