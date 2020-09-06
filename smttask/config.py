from warnings import warn
from typing import Optional
from dataclasses import dataclass
from multiprocessing import cpu_count
from sumatra.projects import load_project, Project
from mackelab_toolbox.utils import Singleton

@dataclass
class Config(metaclass=Singleton):
    """
    Global store of variables accessible to tasks; they can be overwritten
    in a project script.

    Public attributes
    -----------------
    project: Sumatra project variable.
        Defaults to using the one in the current directory. If the .smt project
        folder is in another location, it needs to be loaded with `load_project`
    record: bool
        When true, all RecordedTasks are recorded in the Sumatra database.
        The `False` setting is meant as a debugging option, and so also prevents
        prevents writing to disk.
    allow_uncommitted_changes: bool
        By default, even unrecorded tasks check that the repository is clean.
        Defaults to the negation of `record`.
        I'm not sure of a use case where this value would need to differ from
        `record`.
    cache_runs: bool
        Set to true to cache run() executions in memory.
        Can also be overridden at the class level.
    max_processes: int
        Maximum number of Task processes to allow running simultaneously on
        the machine. Uses lock files in the system's default temporary directory
        to detect other processes.

        Negative values are equivalent to #CPUs - 1.

    Public methods
    --------------
    load_project(path)
    """
    _project                  : Optional[Project] = None
    _record                   : bool = True
    cache_runs                : bool = False
    _allow_uncommitted_changes: Optional[bool] = None
    _max_processes            : int = -1

    def load_project(self, path=None):
        """
        Load a Sumatra project. Internally calls sumatra.projects.load_project.
        Currently this can only be called once, and raises an exception if
        called again.

        The loaded project is accessed as `config.project`.

        Parameters
        ---------
        path: str | path-like
            Project directory. Function searches for an '.smt' directory at that
            location (i.e. the '.smt' should not be included in the path)
            Path can be relative; default is to look in the current working
            directory.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError:
            If called a more than once.
        """
        if self._project is not None:
            raise RuntimeError(
                "Only call `load_project` once: I haven't reasoned out what "
                "kinds of bad things would happen if more than one project "
                "were loaded.")
        self._project = load_project(path)

    @property
    def project(self):
        """If no project was explicitely loaded, use the current directory."""
        if self._project is None:
            self.load_project()
        return self._project

    @property
    def record(self):
        """Whether to record tasks in the Sumatra database."""
        return self._record

    @record.setter
    def record(self, value):
        if not isinstance(value, bool):
            raise TypeError("`value` must be a bool.")
        if value is False:
            warn("Recording of tasks has been disabled. Task results will "
                 "not be written to disk and run parameters not stored in the "
                 "Sumatra databasa.")
        self._record = value

    @property
    def allow_uncommitted_changes(self):
        """
        By default, even unrecorded tasks check that the repository is clean
        When developing, set this to False to allow testing of uncommitted code.
        """
        if isinstance(self._allow_uncommitted_changes, bool):
            return self._allow_uncommitted_changes
        else:
            return not self.record

    @allow_uncommitted_changes.setter
    def allow_uncommitted_changes(self, value):
        if not isinstance(value, bool):
            raise TypeError("`value` must be a bool.")
        warn(f"Setting `allow_uncommitted_changes` to {value}. Have you "
             "considered setting the `record` property instead?")
        self._allow_uncommitted_changes = value

    @property
    def max_processes(self):
        if self._max_processes < 0:
            return cpu_count() - 1
        else:
            return self._max_processes

    @max_processes.setter
    def max_process(self, value):
        if not isinstance(value, int):
            value = int(value)
        if value == 0:
            warn("You specified a maximum of 0 smttask processes. This "
                 "will prevent smttask from executing any task. "
                 "Use -1 to indicate to use the default value.")
        self._max_processes = value

config = Config()
