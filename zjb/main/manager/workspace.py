import ulid
from traits.api import List
from zjb._traits.types import Instance
from zjb.doj.job_manager import JobManager
from zjb.doj.worker import Worker

from ..dtb.atlas import Atlas
from ..dtb.dynamics_model import DynamicsModel
from .project import Project


class Workspace(Project):
    manager = Instance(JobManager, transient=True)

    name = "Workspace"

    parent: None = None  # type: ignore

    atlases = List(Instance(Atlas))

    dynamics = List(Instance(DynamicsModel))

    workers = List(Instance(Worker), transient=True)

    @classmethod
    def from_manager(cls, manager: JobManager):
        data = next(manager.iter(), None)
        if not data:  # manager为空
            workspace = cls()
            workspace._gid = ulid.MIN_ULID
            manager.bind(workspace)
            workspace.manager = manager
            return workspace
        if isinstance(data, cls):
            data.manager = manager
            return data
        raise ValueError("The workspace must be created from an empty JobManager")
