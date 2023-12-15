from dataclasses import dataclass, field
from typing import Optional, Callable, List, Sequence
from scityping.pydantic import BaseModel
from sumatra.projects import load_project, Project
from sumatra.parameters import NTParameterSet
from .._utils import Singleton

@dataclass
class Config(metaclass=Singleton):
    """
    Global store of variables accessible to record store viewing functions;
    they can be overwritten in a project script.

    Public attributes
    -----------------
    project: Sumatra project variable.
        Defaults to using the one in the current directory. If the .smt project
        folder is in another location, it needs to be loaded with `load_project`
    data_models: List[BaseModel]
        Within attempting to load data from disc (see `get_output`), these are
        tried in sequence, and the first to load successfully is returned.
        At present these must all be subtypes of pydantic.BaseModel

    Public methods
    --------------
    load_project(path)
    """
    # DEV NOTE: The idea for if and when view is separated into its own project,
    #    it will want its own config rather than using smttask's.
    #    Smttask will probably still want to set values that should be sync'ed,
    #    like 'project' and ParameterSet.
    _project   : Optional[Project]=None
    _ParameterSet             : type=NTParameterSet
    data_models: List[BaseModel]  =field(default_factory=lambda:[])
    get_field_value: Callable=getattr

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
        # Setter throws error if _project is already set, unless it is set to the same value
        self.project = path if isinstance(path, Project) else load_project(path)
                # raise RuntimeError(
                #     "Only call `load_project` once: I haven't reasoned out what "
                #     "kinds of bad things would happen if more than one project "
                #     "were loaded.")

    @property
    def project(self):
        """If no project was explicitely loaded, use the current directory."""
        if self._project is None:
            self.load_project()
        return self._project

    @project.setter
    def project(self, value):
        if value is self._project:
            # Nothing to do
            return
        elif self._project is not None:
            raise RuntimeError(f"`{__name__}.project` is already set.")
        elif not isinstance(value, Project):
            raise TypeError("Project must be of type `sumatra.projects.Project`. "
                            f"Received '{type(value)}'; use `load_project` to "
                            "create a project from a path.")
        else:
            self._project = value
    @property
    def ParameterSet(self):
        """The class to use as ParameterSet. Must be a subclass of parameters.ParameterSet."""
        return self._ParameterSet
    @ParameterSet.setter
    def ParameterSet(self, value):
        if not lenient_issubclass(value, base_ParameterSet):
            raise TypeError("ParameterSet must be a subclass of parameters.ParameterSet")
        self._ParameterSet = value

    ## Viz config ##
    @property
    def backend(self):
        try:
            import holoviews as hv
        except ImportError:
            try:
                import matplotlib
            except:
                return 'none'
            else:
                return 'matplotlib'  # Use same string as HoloViews
        else:
            return hv.Store.current_backend

    # The following are the keyword args to Bokeh's DatetimeTickFormatter
    datetime_formats = {
      **{scale: '%Y-%m-%dT%H:%M:%S'
          for scale in ('microseconds', 'milliseconds', 'seconds', 'minsec', 'minutes')},
       **{scale: '%Y-%m-%dT%H:%M'
          for scale in ('hourmin', 'hours')},
       **{scale: '%Y-%m-%dT%H'
          for scale in ('days',)},
       **{scale: '%Y-%m-%d'
          for scale in ('months', 'years')}
      }
    @property
    def datetime_formatter(self):
        if self.backend == "bokeh":
            from bokeh.models.formatters import DatetimeTickFormatter
            return DatetimeTickFormatter(**self.datetime_formats)
        else:
            raise NotImplementedError

config = Config()
