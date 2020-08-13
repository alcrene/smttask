from __future__ import annotations

import os
import io
import logging
from warnings import warn
import abc
import inspect
import importlib
from collections import Iterable, Callable
from pathlib import Path
import numpy as np
from sumatra.projects import load_project
from sumatra.parameters import build_parameters
import mackelab_toolbox as mtb
import mackelab_toolbox.iotools
from mackelab_toolbox.utils import stablehexdigest, stableintdigest

from sumatra.parameters import NTParameterSet as ParameterSet
from sumatra.datastore.filesystem import DataFile

from . import utils
from .typing import describe_datafile
from typing import Union, Tuple, ClassVar

# For serialization
from pydantic import BaseModel, ValidationError
import pydantic.parse
from mackelab_toolbox.typing import json_encoders as mtb_json_encoders

logger = logging.getLogger()

__ALL__ = ['project', 'NotComputed', 'Task',
           'TaskInputs', 'TaskOutputs']

import types
import sys
class Config:
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
    Public methods
    --------------
    load_project(path)
    """
    def __init__(self):
        # FIXME: How does `load_project()` work if we load multiple projects ?
        self._project = None
        self._record = True
        self.cache_runs = False
        self._allow_uncommitted_changes = None
        # self._TaskTypes = set()

    def load_project(self, path=None):
        """
        Load a Sumatra project. Internally calls sumatra.projects.load_project.
        Currently this can only be called once, and raises an exception if
        called again.

        The loaded project is accessed as `config.project`.

        Parameters
        ---------
        path: str | path-lik
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
config = Config()

# Store instantiatied tasks in memory, so the same task with the same parameters
# yields the same instance. This makes in-memory caching much more useful.
# Each task is under a project name, in case different projects have tasks
# with the same name.
class TaskInstanceCache(dict):
    def __getitem__(self, k):
        """Initializes new caches with an empty dict on first access."""
        if k not in self:
            self[k] = {}
        return super().__getitem__(k)
# instantiated_tasks = {config.project.name: {}}
instantiated_tasks = TaskInstanceCache()

class NotComputed:
    pass

import types
def find_tasks(*task_modules):
    """
    Find all Tasks defined in the listed `task_modules`.
    As a special case, a namespace dictionary can also be specified;
    for instance `vars()` can be used to search for tasks in
    the current global namespace.

    Parameters
    ----------
    *task_modules: modules | dictionaries

    Returns
    -------
    dict
        Dictionary is composed of (task name : task type) pairs.
    """
    taskTs = set()
    for module in task_modules:
        if isinstance(module, dict):
            varsdict = module
        elif isinstance(module, types.ModuleType):
            varsdict = vars(module)
        else:
            raise TypeError("`find_tasks` expects modules as arguments")
        taskTs.update(
            v for v in varsdict.values()
            if isinstance(v, type) and issubclass(v, Task))
    return taskTs

# TODO: Make a task validator?, e.g. Task[float], such that we can validate
# that tasks passed as arguments have the expected output.

class Task(abc.ABC):
    """
    Task format:
    Use `RecordedTask` or `InMemoryTask` as base class
    Note that input types must always be a Union which includes the `Task` type.
    This is taken care of by the decorator.

    .. Note:: The definition of `Inputs` and `Outputs` classes, and of the
       `_run` method, is taken care of by the function decorators. In any
       case we've conceived, the decorators are an easier, more concise way
       of contructing Tasks.

    class MyTask(RecordedTask):
        class Inputs(TaskInputs):
            a: Union[Task,int],
            b: Union[Task, str, float]
        class Outputs(TaskOutputs):
            c: str
        @staticmethod
        def _run(a, b):
            c = a*b
            return str(c)

    Inputs: TaskInputs (subclass of `pydantic.BaseModel`)
        Dictionary of varname: type pairs. Types can be wrapped as a tuple.
        Inputs will (should?) be validated against these types.
        Certain types (like DataFile) are treated differently.
        TODO: document which types and how.
    outputs: TaskOutputs (subclass of `pydantic.BaseModel`)
        Dictionary of varname: format. The type is passed on to `io.save()`
        and `io.load()` to determine the data format.
        `format` may be either a type or format name registered in
        `iotools.defined_formats` or `iotools._format_types`. (See the
        `mackelab_toolbox.iotools` docs for more information.)
        If you don't need to specify the output types, can also be a list.
        Not however that the order is important, so an unordered mapping
        like a set will lead to errors.

    _run:
        Method signature must match the parameters names defined by `Inputs`.
        Since a task should be stateless, `_run()` should not depend on `self`
        and therefore can be defined as a staticmethod.
    """
    cache = None

    # Pydantic-compatible validator
    # Since `Task` is added to the possible types for each Task input,
    # in each case, Pydantic first tries this validator to see if it succeeds.
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        if isinstance(value, cls):
            return value
        else:
            return cls.from_desc(value)

    @abc.abstractmethod
    def _run(self):
        """
        This is where subclasses place their code.
        Returned value must match the shape of self.outputs:
        Either a dict with keys matching the names in self.outputs, or a tuple
        of same length as self.outputs.
        """
        pass

    def __new__(cls, arg0=None, *, reason=None, **taskinputs):
        """
        Performs two checks:
        1. Allows "constructing" a task with an instance of itself, in which
           no construction occurs and the instance is simply returned.
        2. A unique task is only instantiated once: if one tries to create
           another of same type with same parameters, the previous instance is
           returned.
        """
        if isinstance(arg0, cls):
            return arg0
        else:
            taskinputs = cls._merge_arg0_and_taskinputs(arg0, taskinputs)
            # h = cls.get_digest(taskinputs)
            h = taskinputs.digest
            taskdict = instantiated_tasks[config.project.name]
            if h not in taskdict:
                taskdict[h] = super().__new__(cls)

            return taskdict[h]

    def __init__(self, arg0=None, *, reason=None, **taskinputs):
        """
        Parameters
        ----------
        arg0: ParameterSet-like
            ParameterSet, or something which can be cast to a ParameterSet
            (like a dict or filename). The result will be parsed for task
            arguments defined in `self.Inputs`.
        reason: str
            Arbitrary string included in the Sumatra record for the run.
            Serves the same purpose as a version control commit message,
            and simarly essential.
        **taskinputs:
            Task parameters can also be specified as keyword arguments,
            and will override those in :param:arg0.
        """
        assert hasattr(self, 'Inputs')
        assert hasattr(self, 'Outputs')
        task_attributes = \
            ['taskinputs', '_loaded_inputs', '_run_result']
        if isinstance(arg0, type(self)):
            # Skip initializion of pre-existing instance (see __new__)
            assert all(hasattr(self, attr) for attr in task_attributes)
            return
        if all(hasattr(self, attr) for attr in task_attributes):
            # Task is already instantiated because loaded from cache
            return

        # TODO: this is already done in __new__
        self.taskinputs = self._merge_arg0_and_taskinputs(arg0, taskinputs)
        self._loaded_inputs = None  # Where inputs are stored once loaded
        self._run_result = NotComputed

        self._dependency_graph = None

    @classmethod
    def _merge_arg0_and_taskinputs(cls, arg0, taskinputs):
        """
        arg0: arguments passed as a dictionary to constructor
        taskinputs: arguments passed directly as keywords to constructor

        This function does the following:
          + Merge dictionary and keyword arguments. Keyword arguments take
            precedence.
          + Cast to `cls.Inputs`. This checks that all required inputs are
            provided, casts them the right type, and falls back
            to default values when needed.
        """
        if arg0 is None:
            arg0 = {}
        elif isinstance(arg0, str):
            arg0 = build_parameters(arg0)
        elif isinstance(arg0, dict):
            arg0 = ParameterSet(arg0)
        elif type(arg0) is TaskInputs:
            # Non subclassed TaskInputs; re-instantiate with correct Inputs class to catch errors
            arg0 = cls.Inputs(**arg0.dict()).dict()
        elif isinstance(arg0, cls.Inputs):
            arg0 = arg0.dict()
        else:
            raise ValueError("Use keyword arguments to specify task inputs. "
                             "A single positional argument may be provided, "
                             "but it must either be:\n"
                             "1) a ParameterSet (dictionary) of input values;\n"
                             "2) a file path to a ParameterSet;\n"
                             "3) a TaskInputs object.\n"
                             f"Instead, the intializer for {cls.__name__} "
                             f"received a value of type '{type(arg0)}'.")
        taskinputs = {**arg0, **taskinputs}

        return cls.Inputs(**taskinputs)

    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name + "(" +  \
            ', '.join(kw+'='+repr(v) for kw,v in self.taskinputs) + ")"
    # Ideally we would define just one @classproperty 'name', but that requires
    # more metaclass magic than justified
    @classmethod
    def taskname(cls):
        return cls.__qualname__
    @property
    def name(self):
        return self.taskname()

    @property
    def desc(self):
        return self.get_desc(self.taskinputs)
    @classmethod
    def get_desc(cls, taskinputs):
        module_name = getattr(cls, '_module_name', cls.__module__)
        return TaskDesc(taskname=cls.taskname(),
                        inputs  =taskinputs,
                        module  =module_name)

    @property
    def digest(self):
        # warn("Use Task.Inputs.digest instead of Task.digest.",
        #      DeprecationWarning)
        return self.taskinputs.digest

    def __hash__(self):
        return hash(self.taskinputs)
        # return stableintdigest(self.desc.json())

    @property
    def input_files(self):
        # Also makes paths relative, in case they weren't already
        store = config.project.input_datastore
        return [os.path.relpath(Path(input.full_path).resolve(),store)
                for _, input in self.taskinputs
                if isinstance(input, DataFile)]

    @staticmethod
    def from_desc(desc: TaskDesc, on_fail='raise'):
        """
        Instantiate a class from the description returned by 'desc'.
        This is especially useful to reload a task definition from disk.

        Parameters
        ----------
        desc: Task description (dict)
            Any value accepted by TaskDesc.load.
        on_fail: 'raise' | 'ignore'
            What to do if the load fails.
        """
        try:
            desc = TaskDesc.load(desc)
        except ValidationError:
            if on_fail.lower() == 'raise':
                raise
            else:
                warn("`desc` is not a valid Task description.")
                return None

        m = importlib.import_module(desc.module)
        TaskType = getattr(m, desc.taskname)
        assert desc.taskname == TaskType.taskname()
        return TaskType(desc.inputs)

        taskinputs = ParameterSet({})
        for name, θ in desc.inputs.items():
            θtype = TaskType.inputs[name]
            if utils.is_task_desc(θ):
                taskinputs[name] = Task.from_desc(θ)
            else:
                taskinputs[name] = θ
        return TaskType(**taskinputs)

    def load_inputs(self):
        """
        Return a copy of `self.taskinputs`, with all lazy inputs resolved:
          - files are loaded with `io.load()`
          - upstream tasks are executed
        If necessary, loaded values are cast to their expected type.

        Loads are cached, so subsequent calls simply return the value computed
        the first time.
        """
        if self._loaded_inputs is None:
            self._loaded_inputs = self.taskinputs.load()
        return self._loaded_inputs

    @property
    def graph(self):
        """Return a dependency graph. Uses NetworkX."""
        if self._dependency_graph is None:
            from .networkx import TaskGraph  # Don't require networkx otherwise
            self._dependency_graph = TaskGraph(self)
        return self._dependency_graph

    def draw(self, *args, **kwargs):
        """Draw the dependency graph. See `smttask.networkx.TaskGraph.draw()`"""
        G = self.graph
        G.draw(*args, **kwargs)

    def save(self, path):
        suffix = '.' + mtb.iotools.defined_formats['taskdesc'].ext.strip('.')
        with open(Path(path).with_suffix(suffix), 'w') as f:
            f.write(self.desc.json())
        # self.desc.save(Path(path).with_suffix(suffix))

    @classmethod
    def load(cls, obj):
        return cls.from_desc(obj)

class RecordedTaskBase(Task):
    """A task which is saved to disk and? recorded with Sumatra."""
    # TODO: Add missing requirements (e.g. write())
    @property
    @abc.abstractmethod
    def outputpaths(self):
        pass

# ============================
# Serialization
# ============================

# The value returned by the json encoder is used to compute the task digest, and
# therefore must reflect any change which would change the task output.
# In particular, this means resolving all file paths, because if
# an input file differs (e.g. a symbolic link points somewhere new),
# than the task must be recomputed.
def json_encoder_InputDataFile(datafile):
    return str(utils.relative_path(src=config.project.input_datastore.root,
                                   dst=datafile.full_path))

def json_encoder_OutputDataFile(datafile):
    return str(utils.relative_path(src=config.project.data_store.root,
                                   dst=datafile.full_path))

class TaskInputs(BaseModel, abc.ABC):
    """
    Base class for task inputs.
    Each Task defines a new TaskInputs class, which subclasses this one.

    The following attributes are disallowed and will raise an error if they
    are part of the list of inputs:

    - :attr:`reason`

    .. TODO:: We should check that inputs which are Task instances have
       appropriate output type.
    """
    class Config:
        # The base type is used to construct inputs when deserializing;
        # since it defines no attributes, the default config `extra`='ignore'
        # would discard all inputs. 'allow' indicates to keep them all.
        # Subclasses however should set this to 'forbid' to catch errors.
        # During instantiation, if a Task receives a non-subclassed TaskInputs,
        # it should re-instantiate that to its own TaskInputs subclass.
        extra = 'allow'
        arbitrary_types_allowed = True
        json_encoders = {**mtb_json_encoders,
                         DataFile: json_encoder_InputDataFile,
                         Task: lambda task: task.desc.dict()}

    # Ideally these checks would be in the metaclass/decorator
    def __init__(self, *args, **kwargs):
        if 'arg0' in self.__fields__:
            raise AssertionError(
                "A task cannot define an input named 'arg0'.")
        if 'reason' in self.__fields__:
            raise AssertionError(
                "A task cannot define an input named 'reason'.")
        super().__init__(*args, **kwargs)

    def load(self):
        """
        Return a copy, with all lazy inputs resolved:
          - files are loaded with `io.load()`
          - upstream tasks are executed
        If necessary, loaded values are cast to their expected type.
        """
        # Resolve lazy inputs
        obj = {attr: io.load(v.full_path) if isinstance(v, DataFile)
                     else v.run() if isinstance(v, Task)
                     else v
               for attr, v in self.dict().items()}
        # Validate, cast, and return
        return type(self)(**obj)

    @property
    def digest(self) -> str:
        return stablehexdigest(self.json())

    def __hash__(self) -> int:
        return stableintdigest(self.json())

#TODO? Move outputpaths and _outputpaths_gen to this class ?
class TaskOutputs(BaseModel, abc.ABC):
    """
    .. rubric:: Output definition logic:

       If a TaskOutputs type defines only one output, it expects the result to
       be that output::

           class Outputs:
             x: float

       Expects the task to return a single float, which will be saved with the
       name 'x'. Similarly::

           class Outputs:
             x: Tuple[float, float]

       expects a single tuple, which will be saved with the name 'x'.

       If instead a TaskOutputs type defines multiple values, it expects them
       to be wrapped with a tuple (as would multiple return values from a
       function). So::

           class Outputs:
             x: float
             y: float

       expects the task to return a tuple of length two, the components of
       which will be saved with the names 'x' and 'y'. Similarly::

           class Outputs:
             x: Tuple[float, float]
             y: float

       expects the task to return a tuple of length two, the elements of which
       would be a tuple and a float.

       .. warning:: In the interest of brevity, the snippets above are
          incomplete. In a real definition, each type is a union including
          `Task`, and each TaskOutputs subclass defines a `_task` class variable.

    .. Remark:: In contrast to most other situations, we try here to avoid
       raising exceptions. This is because once we've reached this
       method, we have already computed the result. This may have taken a
       long time, and we want to do our best to save the data in some form.
       If we can't parse the result, we store the original value to allow
       `write` to save it as-is. It will not be useable in downstream tasks,
       but at least this gives the user a chance to inspect it.
    """
    __slots__ = ('_unparsed_result', '_well_formed', '_task')

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {**TaskInputs.Config.json_encoders,
                         DataFile: json_encoder_OutputDataFile}

    # Ideally these checks would be in the metaclass
    def __init__(self, *args, _task, **kwargs):
        if len(self.__fields__) == 0:
            raise TypeError("Task defines no outputs. This must be an error, "
                            "because tasks are not allowed to have side-effects.")
        super().__init__(*args, **kwargs)
        if not isinstance(_task, Task):
            raise ValidationError("'_task' argument must be a Task instance.")
        object.__setattr__(self, '_task', _task)
        # If we made it here, output arguments were successfully validated
        object.__setattr__(self, '_unparsed_result', None)
        object.__setattr__(self, '_well_formed', True)

    def copy(self, *args, **kwargs):
        c = self.copy(*args, **kwargs)
        object.__setattr__(self, '_task', self._task)
        object.__setattr__(c, '_unparsed_result', self._unparsed_result)
        object.__setattr__(self, '_well_formed', self._well_formed)
        return c
    def dict(self, *args, **kwargs):
        if self._unparsed_result is not None:
            warn("Dict representations for malformed outputs are ill-defined.")
            return {'_unparsed_result': self._unparsed_result}
        else:
            return super().dict(*args, **kwargs)

    def __len__(self):
        if self._well_formed:
            return len(self.__fields__)
        else:
            if self._unparsed_result is None:
                return 0
            else:
                # TODO? Not sure what the best value would be to return.
                #   The number of files that would be produced by `save` ?
                return 1

    @property
    def result(self):
        """
        Return the value, as it would have been returned by the Task.

        If there is one output:
            Return the bare value.
        If there is more than one output:
            Return the outputs wrapped in a tuple, in the order they are
            defined in this TaskOutputs subclass.
        """
        if not self._well_formed:
            return self._unparsed_result
        elif len(self) == 1:
            return getattr(self, next(iter(self.__fields__)))
        else:
            return tuple(value for attr,value in self)

    @classmethod
    def parse_result(cls, result: Any, _task: Task) -> TaskOutputs:
        """
        Parameters
        ----------
        result:
            Value returned by executed the :meth:`run()` method of the
            associated task.
            If more than one output is expected, this must be either a tuple
            or dict.
        _task:
            The Task instance which produced `result`.

        Returns
        -------
        TaskOutputs:
            If parsing is successful, the values of `result` are assigned to
            the attributes of TaskOutputs.
            If parsing is unsuccessful, the value of `result` is assigned
            unchanged to `_unparsed_values`.

            `write` checks the value of `_unparsed_values` to determine which
            export function to use. If the value is ``None``, standard export
            to JSON is performed, otherwise the `emergency_dump` method is used.
        """
        try:
            taskname = _task.taskname()
        except:
            taskname = ""
        failed = False
        if len(cls.__fields__) == 1:
            # Result is not expected to be wrapped with a tuple.
            nm = next(iter(cls.__fields__))
            result = {nm: result}
        elif not isinstance(result, (tuple, dict)):
            warn(f"Task {taskname} defines multiple outputs, and "
                 "should return them wrapped with a tuple or a dict.")
            failed = True

        if isinstance(result, tuple):
            result = {nm: val for nm,val in zip(cls.__fields__, result)}

        # At this point, either `failed` == True, or `result` is a dict.

        if not failed:
            try:
                taskoutputs = cls(**result, _task=_task)
            except ValidationError as e:
                warn(f"\n\nThe output of task {taskname} was malformed. "
                     "Attempted to cast to the expected output format raised "
                     f"the following exception:\n{str(e)}\n")
                failed = True

        if failed:
            # Create a set of dummy values to allow creating the object;
            # actual values will be in _unparsed_result.
            # We used `construct` to skip model validation.
            dummy_values = {attr: None for attr in cls.__fields__}
            taskoutputs = cls.construct(**dummy_values)
            object.__setattr__(taskoutputs, '_unparsed_result', result)
            object.__setattr__(taskoutputs, '_well_formed', False)
            object.__setattr__(taskoutputs, '_task', _task)
        else:
            assert taskoutputs._unparsed_result is None
            assert taskoutputs._well_formed is True
            assert taskoutputs._task is _task

        assert isinstance(taskoutputs, cls)
        return taskoutputs

    @staticmethod
    def emergency_dump(filename, obj):
        """
        A function called when the normal saving function fails.
        If `obj` is an iterable, each element is saved individually.
        """
        filename, ext = os.path.splitext(filename)
        if isinstance(obj, dict):
            for name, el in obj.items():
                TaskOutputs.emergency_dump(f"{filename}__{name}{ext}", el)
        elif isinstance(obj, Iterable):
            for i, el in enumerate(obj):
                TaskOutputs.emergency_dump(f"{filename}__{i}{ext}", el)
        else:
            warn("An error occured while writing the task output to disk. "
                 "Unceremoniously dumping data at this location, to allow "
                 f"post-mortem: {filename}...")
            mtb.iotools.save(filename, obj)

    def write(self, **dumps_kwargs):
        """
        Save outputs to the automatically determined file location.

        **dumps_kwargs are passed on to the model's json encoder.
        """
        # If the result was malformed, use the emergency_dump and exit immediately
        try:
            taskname = self._task.taskname()
        except:
            taskname = ""
        if not self._well_formed:
            if self._unparsed_result is None:
                warn(f"{taskname}.Outputs.write: "
                     "Nothing to write. Aborting.")
                return
            outpath = next(iter(self._task._outputpaths_gen))
            outpath, ext = os.path.splitext(outpath)
            outpath += "__emergency_dump"
            warn(f"{taskname}.Outputs.write: outputs were "
                 "malformed. Falling back to emergency dump (location: "
                 f"{outpath}). Inspecting the saved output may help determine "
                 "the cause of the error.")
            self.emergency_dump(outpath+ext, self._unparsed_result)
            return []

        outroot = Path(config.project.data_store.root)
        inroot = Path(config.project.input_datastore.root)
        orig_outpaths = self._task.outputpaths
        outpaths = []  # outpaths may add suffix to avoid overwriting data
        for nm, value in self:
            # Next line copied from pydantic.main.json
            json = self.__config__.json_dumps(
                value, default=self.__json_encoder__, **dumps_kwargs)
            relpath = orig_outpaths[nm]
            f, truepath = mtb.iotools.get_free_file(outroot/relpath, bytes=True)
                # Truepath may differ from outroot/relpath if a file was already at that location
            f.write(json.encode('utf-8'))
            f.close()
            outpaths.append(truepath)
            # Add link in input store, potentially overwriting old link
            inpath = inroot/relpath.with_suffix(Path(truepath).suffix)
            if inpath.is_symlink():
                # Deal with race condition ? Wait future Python builtin ?
                # See https://stackoverflow.com/a/55741590,
                #     https://github.com/python/cpython/pull/14464
                os.remove(inpath)
            else:
                os.makedirs(inpath.parent, exist_ok=True)

            os.symlink(utils.relative_path(inpath.parent, truepath),
                       inpath)

        return outpaths

class TaskDesc(BaseModel):
    taskname: str
    module  : str
    inputs  : TaskInputs

    class Config:
        json_encoders = TaskInputs.Config.json_encoders

    @classmethod
    def load(cls, obj):
        """
        Calls either `parse_obj`, `parse_raw` or `parse_file`, depending on
        the value of `obj`.

        Parameters
        ----------
        obj: dict | str | path-like | file object
            Serialized task description
        """
        taskdesc = None
        if isinstance(obj, TaskDesc):
            return obj
        elif isinstance(obj, str):
            # May be a JSON string, or a path.
            obj_type = 'unknown'
            if '\n' in obj:
                # newlines are illegal in paths
                obj_type = 'JSON'
            elif '{' not in obj:
                # A JSON object has at least one pair of brackets
                obj_type = 'path'
            else:
                obj_type = 'likely JSON'

            if obj_type == 'JSON':
                taskdesc = cls.parse_raw(obj)
            elif obj_type == 'likely JSON':
                try:
                    taskdesc = cls.parse_raw(obj)
                except ValidationError:
                    taskdesc = cls.parse_file(obj)
            else:
                taskdesc = cls.parse_file(obj)

        elif isinstance(obj, Path):
            taskdesc = cls.parse_file(obj)

        elif isinstance(obj, io.IOBase):
            obj = pydantic.parse.load_str_bytes(obj.read())
            taskdesc = cls.parse_obj(obj)

        else:
            assert isinstance(obj, dict)
            taskdesc = cls.parse_obj(obj)

        assert isinstance(taskdesc, TaskDesc)
        return taskdesc

# ============================
# Register the taskdesc type with mackelab_toolbox.iotools
# ============================
# import mackelab_toolbox.iotools as io
ioformat = mtb.iotools.Format('taskdesc',
                              save=lambda file,task: task.save(file),
                              load=Task.load,
                              bytes=False)
mtb.iotools.defined_formats['taskdesc'] = ioformat
mtb.iotools.register_datatype(Task, format='taskdesc')
