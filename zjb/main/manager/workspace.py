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
    def from_manager(cls, manager: JobManager, gid: "ulid.ULID | None" = None):
        if gid:
            return super().from_manager(manager, gid)
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

    def start_workers(self, count: int = 1):
        """启动一些Worker

        Parameters
        ----------
        count : int, optional
            要启动的Worker数量, by default 1
        """
        new_workers = [Worker(manager=self.manager) for _ in range(count)]
        for worker in new_workers:
            worker.start()
        self.workers += new_workers

    def remove_idle_workers(self, count: int = 0):
        """移除一些空闲的Worker

        Parameters
        ----------
        count : int, optional
            要移除的空闲Worker数量, 小于等于0时会移除所有空闲Worker, by default 0
        """
        terminated_workers: list[Worker] = []
        for worker in self.workers:
            result = worker.terminate()
            if result:
                terminated_workers.append(worker)
                count -= 1
                if count == 0:
                    break
        for worker in terminated_workers:
            self.workers.remove(worker)
