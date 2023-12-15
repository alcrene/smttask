import os
from warnings import warn
from typing import Optional
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from sumatra.projects import load_project, Project
from sumatra.parameters import NTParameterSet

import scityping

from ._utils import Singleton, lenient_issubclass, is_parameterset

scityping.config.safe_packages.add("smttask")

# TODO: Use ValidatingConfig (also in smttask.view.config)

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
    trust_all_inputs: bool
        DEPRECATED: Use :external:`scityping.config.trust_all_inputs` instead.
        Allow deserializations which can lead to lead to arbitrary code
        execution, and therefore are potentially unsafe. Required for
        deserializing:
        - `PureFunction`
        - `Type`
        The value is synchronized with `scityping.config.trust_all_inputs` and
        defaults to False.

    terminating_types: set
        Set of types which are not expanded when we flatten a list.
        Default is `{str, bytes}`.
        To modify this set, use in-place set operations; e.g. `config.terminating_types.add(np.ndarray)`.
    cache_runs: bool
        Set to true to cache run() executions in memory.
        Can also be overridden at the class level.
        NOTE: Only applies Recorded tasks. Memoized tasks are always cached by
        default, unless their class attribute `cache` is set to False.
    allow_uncommitted_changes: bool
        If set to False, even unrecorded tasks will fail if the repository is
        not clean.
        Defaults to the negation of `record`.
        I'm not sure of a use case where this value would need to differ from
        `record`.
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
    _trust_all_inputs         : Optional[bool] = None  # Defaults to scityping.config.trust_all_inputs, who's default is False
    _terminating_types        : set = field(default_factory=lambda: {str, bytes})
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
        # DEVNOTE: If we stored the project as a task attribute when a task
        # is instantiated, it should be possible to support changing projects.
        # Probably we would want to do the same with RecordStoreView
        # If we do this, a replace-(almost)-all "config.project" ->
        # "task.project" is recommended.
        
        # Check with the view – maybe the project was needed there first
        # And if it wasn't loaded, ensure that both smttask and smttask.view
        # use the same project
        from . import view
        if view.config._project:
            self.project = view.config.project
        else:
            self.project = load_project(path)
            view.config.project = self.project

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
        from . import view
        view.config.project = value

    # DEV NOTE: Both smttask and smttask.view need ParameterSet in their config.
    #    The least surprising thing to do seems to be:
    #    - If ParameterSet is set in smttask, set both smttask and smttask.view
    #    - If ParameterSet is set in smttask.view, only set smttask.view
    @property
    def ParameterSet(self):
        """The class to use as ParameterSet. Must be a subclass of parameters.ParameterSet."""
        return self._ParameterSet
    @ParameterSet.setter
    def ParameterSet(self, value):
        if not is_parameterset(value):
            raise TypeError("ParameterSet must be a subclass of parameters.ParameterSet")
        self._ParameterSet = value
        # Keep smttask.view.config in sync
        # (Note: this is not reciprocal, for the reason given above)
        from . import view
        view.config.ParameterSet = value

    @property
    def record(self):
        """Whether to record tasks in the Sumatra database."""
        return self._record

    @record.setter
    def record(self, value):
        if not isinstance(value, bool):
            raise TypeError("`value` must be a bool.")
        if self._record and value is False:  # No need to display a warning if the setting doesn't change
            warn("Recording of tasks has been disabled. Task results will "
                 "not be written to disk and run parameters not stored in the "
                 "Sumatra database.")
        self._record = value

    @property
    def safe_packages(self):
        return scityping.config.safe_packages

    @property
    def trust_all_inputs(self):
        if self._trust_all_inputs is None:
            self._trust_all_inputs = scityping.config.trust_all_inputs
        return self._trust_all_inputs
    @trust_all_inputs.setter
    def trust_all_inputs(self, value):
        logger.warning("Deprecated: set `scitpying.config.trust_all_inputs` instead of `smttask.config.trust_all_inputs`.")
        scityping.config.trust_all_inputs = value
        self._trust_all_inputs = value

    # TODO: Provide interface like `config.terminating_types.add()`
    # TODO?: Move to scityping ?
    @property
    def terminating_types(self):  # TODO?: Merge with theano_shim.config terminating_types
        return self._terminating_types
    @terminating_types.setter
    def terminating_types(self, value):
        raise AttributeError("Use in-place set manipulation (e.g. `config.terminating_types.add`) to modify the set of terminating_types")

    @property
    def allow_uncommitted_changes(self):
        """
        If set to False, even unrecorded tasks will fail if the repository is not clean.
        Defaults to the negation of `record`.
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
            return cpu_count() + self._max_processes
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
