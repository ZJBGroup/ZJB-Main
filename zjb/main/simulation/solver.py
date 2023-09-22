from abc import abstractmethod
from pathlib import Path

from mako.template import Template
from traits.api import ABCMetaHasTraits, Float, HasPrivateTraits

from ..dtb.dynamic_model import DynamicModel

TEMPLATE = Template(filename=str(Path(__file__).parent / "_templates" / "solvers.mako"))


class Solver(HasPrivateTraits, metaclass=ABCMetaHasTraits):
    dt = Float(0.1)

    @abstractmethod
    def render(self, model: DynamicModel) -> str:
        ...


class EulerSolver(Solver):
    def render(self, model: DynamicModel) -> str:
        return TEMPLATE.get_def("euler").render(model=model)  # type: ignore
