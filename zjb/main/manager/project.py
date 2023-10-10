from traits.api import List, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..dtb.dtb import DTB
from ..dtb.dtb_model import DTBModel
from ..dtb.subject import Subject


class Project(Data):
    name = Str()

    parent: "Project" = Instance("Project")  # type: ignore

    children: list["Project"] = List(Instance("Project"))  # type: ignore

    subjects = List(Instance(Subject))

    models = List(Instance(DTBModel))

    dtbs = List(Instance(DTB))
