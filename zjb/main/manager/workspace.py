import ulid
from traits.api import List

from zjb._traits.types import Instance
from zjb.doj.job_manager import JobManager
from zjb.doj.worker import Worker

from ..dtb.atlas import Atlas
from ..dtb.dynamics_model import DynamicsModel
from .project import Project


class Workspace(Project):
    """
    工作空间类，继承自Project类，用于管理和处理与工作空间相关的数据和操作。

    Attributes
    ----------
    manager : Instance(JobManager)
        与工作空间相关联的作业管理器实例。
    name : str
        工作空间名称，默认为'Workspace'。
    parent : None
        工作空间没有父项目，始终为None。
    atlases : List(Instance(Atlas))
        工作空间中包含的脑图谱实例的列表。
    dynamics : List(Instance(DynamicsModel))
        工作空间中包含的动力学模型列表。
    workers : List(Instance(Worker))
        工作空间中的处理器列表。
    """
    manager = Instance(JobManager, transient=True)

    name = "Workspace"

    parent: None = None  # type: ignore

    atlases = List(Instance(Atlas))

    dynamics = List(Instance(DynamicsModel))

    workers = List(Instance(Worker), transient=True)

    @classmethod
    def from_manager(cls, manager: JobManager, gid: "ulid.ULID | None" = None):
        """
        从作业管理器创建工作空间实例。如果提供了全局唯一标识符（gid），则使用它来创建实例；否则从作业管理器获取数据创建工作空间。

        Parameters
        ----------
        manager : JobManager
            作业管理器实例。
        gid : ulid.ULID | None, optional
            全局唯一标识符，可选，默认为None。

        Returns
        -------
        Workspace
            创建的工作空间实例。
        """
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

    def unbind(self):
        """解除工作区与作业管理器的绑定关系。由于工作区不支持解绑操作，因此此方法会抛出RuntimeError。"""
        raise RuntimeError(
            "The workspace cannot unbind from the manager, delete the data files directly."
        )

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
