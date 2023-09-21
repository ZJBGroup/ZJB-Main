from traits.api import HasPrivateTraits, HasRequiredTraits, List

from zjb._traits.types import Instance
from zjb.doj.job_manager import JobManager
from zjb.doj.worker import Worker


class Workspace(HasPrivateTraits, HasRequiredTraits):
    manager = Instance(JobManager, required=True)

    workers = List(Worker)
