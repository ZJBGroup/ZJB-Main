from abc import abstractmethod
from pathlib import Path

from mako.template import Template
from traits.api import ABCMetaHasTraits, HasPrivateTraits, Str

TEMPLATE = Template(
    filename=str(Path(__file__).parent / "_templates" / "monitors.mako")
)


class Monitor(HasPrivateTraits, metaclass=ABCMetaHasTraits):
    expression = Str(required=True)

    @abstractmethod
    def render_init(self, name: str) -> str:
        ...

    @abstractmethod
    def render_sample(self, name: str) -> str:
        ...


class Raw(Monitor):
    def render_init(self, name: str) -> str:
        return TEMPLATE.get_def("raw_init").render(name=name, monitor=self)  # type: ignore

    def render_sample(self, name: str) -> str:
        return TEMPLATE.get_def("raw_sample").render(name=name, monitor=self)  # type: ignore
