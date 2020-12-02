import os
from warnings import warn
from typing import Optional
from dataclasses import dataclass
from multiprocessing import cpu_count
from sumatra.projects import load_project, Project
from parameters import ParameterSet as base_ParameterSet
from sumatra.parameters import NTParameterSet
from mackelab_toolbox.utils import Singleton

from ._utils import lenient_issubclass

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

        Setting this value also sets `view.config.project` to the same value.
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

        Negative values are equivalent to #CPUs - `max_processes`.
    process_number: int
        Retrieve the value of the environment variable SMTTASK_PROCESS_NUM.
        This variable is set automatically when executing `smttask run` on the
        command line, and can be used e.g. to assign different cache files
        to simultaneously executing tasks.
        If the environment variable is not set, 0 is returned.

    Public methods
    --------------
    load_project(path)
    """
    _project                  : Optional[Project] = None
    _record                   : bool = True
    cache_runs                : bool = False
    _allow_uncommitted_changes: Optional[bool] = None
    _max_processes            : int = -1
    on_error                  : str = 'raise'
    _ParameterSet             : type=NTParameterSet

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
        # Check with the view – maybe the project needed first there
        # And if it wasn't loaded, ensure that both smttask and smttask.view
        # use the same project
        import smttask.view
        if smttask.view.config._project:
            self.project = smttask.view.config.project
        else:
            self.project = load_project(path)

    @property
    def project(self):
        """If no project was explicitely loaded, use the current directory."""
        if self._project is None:
            self.load_project()
        return self._project

    @project.setter
    def project(self, value):
        if value is self._project:
            # Nothing to do, but we will still check view.config
            pass
        elif self._project is not None:
            raise RuntimeError(f"`{__name__}.project` is already set.")
        elif not isinstance(value, Project):
            raise TypeError("Project must be of type `sumatra.projects.Project`. "
                            f"Received '{type(value)}'; use `load_project` to "
                            "create a project from a path.")
        else:
            self._project = value
        # Set the project for the viewer as well; import done here to ensure
        # we don't introduce cycles.
        import smttask.view
        smttask.view.config.project = value

    @property
    def ParameterSet(self):
        """The class to use as ParameterSet. Must be a subclass of parameters.ParameterSet."""
        return self._ParameterSet
    @ParameterSet.setter
    def ParameterSet(self, value):
        if not lenient_issubclass(value, base_ParameterSet):
            raise TypeError("ParameterSet must be a subclass of parameters.ParameterSet")
        self._ParameterSet = value

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
                 "Sumatra database.")
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
        if self._max_processes <= 0:
            return cpu_count() - self._max_processes
        else:
            return self._max_processes

    @max_processes.setter
    def max_processes(self, value):
        if not isinstance(value, int):
            value = int(value)
        if value == 0:
            warn("You specified a maximum of 0 smttask processes. This "
                 "will prevent smttask from executing any task. "
                 "Use -1 to indicate to use the default value.")
        if value <= -cpu_count():
            warn(f"Tried to set `smttask.config.max_processes` to {value}, which "
                 "would translate to a zero or negative number of cores. "
                 "Setting `max_processes` to '1'.")
            value = 1
        self._max_processes = value

    @property
    def process_number(self):
        return int(os.getenv("SMTTASK_PROCESS_NUM", 0))

config = Config()
