from abc import abstractmethod
from pathlib import Path
from typing import Any

from mako.template import Template
from traits.api import ABCMetaHasTraits, HasPrivateTraits, Str

TEMPLATE = Template(
    filename=str(Path(__file__).parent / "_templates" / "monitors.mako")
)


class Monitor(HasPrivateTraits, metaclass=ABCMetaHasTraits):
    expression = Str(required=True)

    @abstractmethod
    def render_init(self, name: str, env: dict[str, Any]) -> str:
        ...

    @abstractmethod
    def render_sample(self, name: str, env: dict[str, Any]) -> str:
        ...


class Raw(Monitor):
    def render_init(self, name: str, env: dict[str, Any]) -> str:
        return TEMPLATE.get_def("raw_init").render(name=name, monitor=self, env=env)  # type: ignore

    def render_sample(self, name: str, env: dict[str, Any]) -> str:
        return TEMPLATE.get_def("raw_sample").render(name=name, monitor=self, env=env)  # type: ignore
