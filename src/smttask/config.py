import os
from pathlib import Path
from warnings import warn
from typing import Any, Type, Literal
from multiprocessing import cpu_count
from sumatra.projects import load_project, Project
from sumatra.parameters import NTParameterSet, ParameterSet as SmtParameterSet
from parameters import ParameterSet as BaseParameterSet

import scityping

from ._utils import lenient_issubclass

from pydantic import Field, field_validator, computed_field
from valconfig import ValConfig

scityping.config.safe_packages.add("smttask")

class Config(ValConfig):
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
        There are at least two big advantages to setting this to `False` while
        developing a Task:
        1) It allows defining the task in the same file/notebook as where it is run,
           so you can test changes immediately.
        2) It avoids bloating the database and results folders with worthless
           output.
    track_folder: Path | None
        If set, whenever a RecordedTasks is run, its output is saved to this folder.
        This occurs whether the task is executed or just retrieved from the on-disk cache.
        The purpose is to produce a clean set of results. For example, when a project
        is ready to be published, executing all scripts with a clean `track_folder`
        will populate it with the results from exactly those runs which were used
        for the published results.
        Results are saved as links (to the smttask data store), so this has
        negligible space requirements. When possible results are saved with hard
        links, so that the tracked results can easily be copied and shared.
        When this is not possible (e.g. if the `track_folder` and data store
        are on different hard drives), then symbolic links are used.
        HINT: The files saved in the `track_folder` follow the same layout as
              in the input data store. This means that the entire folder can
              be copied to the data store location on another machine, and that
              machine will find and use the result files.
              This is an effective way to allow re-running analysis code on a new
              machine without having to also re-run the (possibly expensive) computations.
        NOTE: Since this is meant to store the task results used in the current
              version, if a task output file already exists exists in the
             `track_folder`, it is overwritten.

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

    ParameterSet: Type
        The class to use as ParameterSet. Must be a subclass of parameters.ParameterSet.

    Public methods
    --------------
    load_project(path)
    """
    _project                 : Project | None = None
    record                   : bool = True
    hash_algo                : Literal["xxhash", "sha1"]="xxhash"
    track_folder             : Path | None = None
    terminating_types        : set = Field(default={str, bytes}, frozen=True)
    cache_runs               : bool = False
    allow_uncommitted_changes_internal: bool | None = Field(default=None, alias="allow_uncommitted_changes")
    max_processes_internal   : int = Field(default=-1, alias="max_processes")
    on_error                 : str = 'raise'
    ParameterSet             : Type = Field(default=NTParameterSet)

    ## Give more user-friendly error messages when trying to modify frozen attributes
    def __setattr__(self, name: str, value: Any):
        if name == "terminating_types":
            raise AttributeError("Use in-place set manipulation (e.g. `config.terminating_types.add`) to modify the set of terminating_types")
        super().__setattr__(name, value)

    ## Computed fields
    @property
    def safe_packages(self):
        return scityping.config.safe_packages

    @property
    def process_number(self):
        return int(os.getenv("SMTTASK_PROCESS_NUM", 0))

    ## Fields with views

    @property
    def allow_uncommitted_changes(self):
        """ If set to False, even unrecorded tasks will fail if the repository is not clean.
        If unset (i.e. equal to ``None``), returns the negation of `record`.
        """
        return (allow:=self.allow_uncommitted_changes_internal) if allow is not None \
                else not self.record

    @property
    def max_processes(self):
        """ Transform the negative value into "total cpu - value”
        """
        return (max_proc:=self.max_processes_internal) if max_proc > 0 \
                else cpu_count() + self._max_processes_internal

    ## Projects are normally loaded automatically, not set explicitly
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

    ## Field validator
    @field_validator("record")
    @classmethod
    def warn_if_not_recording(cls, value):
        # TODO: Don’t display a warning if the setting doesn't change
        if not value:
            warn("Recording of tasks has been disabled. Task results will "
                 "not be written to disk and run parameters not stored in the "
                 "Sumatra database.")
        return value

    @field_validator("track_folder", mode="before")
    @classmethod
    def convert_nonestr_to_None(cls, value):
        """
        Convert the strings 'none' and 'None' to an actual None value.
        (Also "null" and "Null" are recognized, although they are discouraged.)
        """
        if isinstance(value, str) and value in {"none", "None", "null", "Null"}:
            return None
        return value

    @field_validator("track_folder", mode="after")
    @classmethod
    def check_path_is_valid(cls, path):
        """
        Check that the path is valid
            + Create the directory if it does not already exist
            + Raise FileExistsError if the path already and it is not a directory
        """
        if isinstance(path, Path):
            if not path.exists():
                path.mkdir(parents=True)
            elif not path.is_dir():
                raise FileExistsError(f"The path '{path}' already exists and is not a directory. Cannot use it for tracking executed tasks.")
        return path


    @field_validator("allow_uncommitted_changes_internal", mode="after")
    @classmethod
    def suggest_setting_record_instead(cls, value):
        if value is not None:
            warn(f"Setting `allow_uncommitted_changes` to {value}. Have you "
                 "considered setting the `record` property instead?")
        return value

    @field_validator("max_processes_internal", mode="after")
    @classmethod
    def check_max_processes(cls, value):
        if value == 0:
            warn("You specified a maximum of 0 smttask processes. This "
                 "will prevent smttask from executing any task. "
                 "Use -1 to indicate to use the default value.")
        if value <= -cpu_count():
            warn(f"Tried to set `smttask.config.max_processes` to {value}, which "
                 "would translate to a zero or negative number of cores. "
                 "Setting `max_processes` to '1'.")
            value = 1
        return value

    # DEV NOTE: Both smttask and smttask.view need ParameterSet in their config.
    #    The least surprising thing to do seems to be:
    #    - If ParameterSet is set in smttask, set both smttask and smttask.view
    #    - If ParameterSet is set in smttask.view, only set smttask.view
    @field_validator("ParameterSet", mode="after")
    @classmethod
    def is_parameterset(cls, value):
        if not lenient_issubclass(value, (SmtParameterSet, BaseParameterSet)):
            raise TypeError("ParameterSet must be a subclass of parameters.ParameterSet")
        return value
        # Keep smttask.view.config in sync
        # (Note: this is not reciprocal, for the reason given above)
        from . import view             #  Import done here to avoid import loops
        view.config.ParameterSet = value
        return value

config = Config()
