import os
import logging
from warnings import warn
import abc
import inspect
import importlib
from collections import Iterable, Callable
from pathlib import Path
from attrdict import AttrDict
import numpy as np
from sumatra.projects import load_project
from sumatra.parameters import build_parameters
from mackelab_toolbox.parameters import params_to_lists, digest
import mackelab_toolbox.iotools as io
logger = logging.getLogger()

from sumatra.parameters import NTParameterSet as ParameterSet
from sumatra.datastore.filesystem import DataFile

from . import utils
from . import input_types
from .input_types import cast
from .input_types import *     # TODO: remove


__ALL__ = ['project', 'File', 'NotComputed', 'Task', 'RecordedTaskBase']

# Monkey patch AttrDict to allow access to attributes with unicode chars
def _valid_name(self, key):
    cls = type(self)
    return (
        isinstance(key, str) and
        key.isidentifier() and key[0] != '_' and  # This line changed
        not hasattr(cls, key)
    )
import attrdict.mixins
attrdict.mixins.Attr._valid_name = _valid_name

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
    recording: bool
        When true, all RecordedTasks are recorded in the Sumatra database.
        The `False` setting is meant as a debugging option, and so also prevents
        prevents writing to disk.
    allow_uncommited_changes: bool
        By default, even unrecorded tasks check that the repository is clean.
        Defaults to the negation of `recording`.
        I'm not sure of a use case where this value would need to differ from
        `recording`.
    cache_runs: bool
        Set to true to cache run() executions in memory.
        Can also be overridden at the class level.
    Public methods
    --------------
    load_project(path)
    """
    def __init__(self):
        # FIXME: How does `load_project()` work if we load mulitple projects ?
        self._project = None
        self._recording = True
        self.cache_runs = False
        self._allow_uncommited_changes = None
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
    def allow_uncommited_changes(self):
        """
        By default, even unrecorded tasks check that the repository is clean
        When developing, set this to False to allow testing of uncommited code.
        """
        if isinstance(self._allow_uncommited_changes, bool):
            return self._allow_uncommited_changes
        else:
            return self.record
    @allow_uncommited_changes.setter
    def allow_uncommited_changes(self, value):
        if not isinstance(value, bool):
            raise TypeError("`value` must be a bool.")
        warn(f"Setting `allow_uncommited_changes` to {value}. Have you "
             "considered setting the `record` property instead?")
        self._allow_uncommited_changes = value
    # @property
    # def TaskTypes(self):
    #     return {T.taskname(): T for T in self._TaskTypes}
    # def add_task_module(self, module):
    #     if isinstance(module, str):
    #         module = sys.modules[module]
    #     self._TaskTypes.update(find_tasks(module))
    # def add_task_type(self, TaskType):
    #     if isinstance(TaskType, Task):
    #         self._TaskTypes.add(TaskType)
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
    # return {T.taskname(): T for T in taskTs}

class Task(abc.ABC):
    """
    Task format:
    Use `Task` or `InMemoryTask` as base class

    class MyTask(Task):
        inputs = {'a': int, 'b': (str, float)}
        outputs = {'c': str}
        @staticmethod
        def _run(a, b):
            c = a*b
            return str(c)

    inputs: dict
        Dictionary of varname: type pairs. Types can be wrapped as a tuple.
        Inputs will (should?) be validated against these types.
        Certain types (like DataFile) are treated differently.
        TODO: document which types and how.
    outputs: dict | list
        Dictionary of varname: format. The type is passed on to `io.save()`
        and `io.load()` to determine the data format.
        `format` may be either a type or format name registered in
        `iotools.defined_formats` or `iotools._format_types`. (See the
        `mackelab_toolbox.iotools` docs for more information.)
        If you don't need to specify the output types, can also be a list.
        Not however that the order is important, so an unordered mapping
        like a set will lead to errors.

    _run:
        Method signature must match the parameters names defined by `inputs`.
        Since a task should be stateless, `_run()` should not depend on `self`
        and therefore can be defined as a statimethod.
    """
    cache = None

    @property
    @abc.abstractmethod
    def inputs(self):
        pass
    @property
    @abc.abstractmethod
    def outputs(self):
        """
        List of strings, corresponding to variable names.
        These names are appended to the task digest to create unique filenames.
        The order is important, so don't use e.g. a `set`.
        """
        pass
    @abc.abstractmethod
    def _run(self):
        """
        This is where subclasses place their code.
        Returned value must match the shape of self.outputs:
        Either a dict with keys matching the names in self.outputs, or a tuple
        of same length as self.outputs.
        """
        pass

    def __new__(cls, params=None, *, reason=None, **taskinputs):
        """
        Performs two checks:
        1. Allows "constructing" a task with an instance of itself, in which
           no construction occurs and the instance is simply returned.
        2. A unique task is only instantiated once: if one tries to create
           another of same type with same parameters, the previous instance is
           returned.
        """
        if isinstance(params, cls):
            return params
        else:
            taskinputs = cls._merge_params_and_taskinputs(params, taskinputs)
            h = cls.get_digest(taskinputs)
            taskdict = instantiated_tasks[config.project.name]
            if h not in taskdict:
                taskdict[h] = super().__new__(cls)

            return taskdict[h]

    def __init__(self, params=None, *, reason=None, **taskinputs):
        """
        Parameters
        ----------
        params: ParameterSet-like
            ParameterSet, or something which can be cast to a ParameterSet
            (like a dict or filename). The result will be parsed for task
            arguments defined in `self.inputs`.
        reason: str
            Arbitrary string included in the Sumatra record for the run.
            Serves the same purpose as a version control commit message,
            and simarly essential.
        **taskinputs:
            Task parameters can also be specified as keyword arguments,
            and will override those in :param:params.
        """
        assert hasattr(self, 'inputs')
        task_attributes = \
            ['taskinputs', '_loaded_inputs', 'input_descs', '_run_result']
        if 'reason' in self.inputs:
            raise AssertionError(
                "A task cannot define an input named 'reason'.")
        if 'cache_result' in self.inputs:
            raise AssertionError(
                "A task cannot define an input named 'cache_result'.")
        if isinstance(params, type(self)):
            # Skip initializion of pre-existing instance (see __new__)
            assert all(hasattr(self, attr) for attr in task_attributes)
            return
        if all(hasattr(self, attr) for attr in task_attributes):
            # Task is already instantiated because loaded from cache
            return

        # TODO: this is already done in __new__
        self.taskinputs = self._merge_params_and_taskinputs(params, taskinputs)
        self._loaded_inputs = None  # Where inputs are stored once loaded
        # TODO: input_descs are already computed in __new__
        self.input_descs = self.get_input_descs(self.taskinputs)
        self._run_result = NotComputed
        # self.inputs = self.get_inputs
        # self.outputpaths = []
        self._dependency_graph = None

    @classmethod
    def _merge_params_and_taskinputs(cls, params, taskinputs):
        """
        params: arguments passed as a dictionary to constructor
            As a special case, if a task has only one input, it does not need
            to be wrapped in a dict (i.e. `params` can be the value itself).
        taskinputs: arguments passed directly as keywords to constructor

        This function does the following:
          + Merge dictionary and keyword arguments. Keyword arguments take
            precedence.
          + Check that all arguments required by task `run()` signature are
            provided.
          + Retrieve any missing argument values from the defaults in `run()`
            signature.
          + Cast every input to its expected type. If an input defines multiple
            allowable types, the left-most one takes precedence.
        """
        if params is None:
            params = {}
        elif isinstance(params, str):
            params = build_parameters(params)
        elif isinstance(params, dict):
            params = ParameterSet(params)
        else:
            if len(cls.inputs) == 1:
                # For tasks with only one input, don't require dict
                θname, θtype = next(iter(cls.inputs.items()))
                if len(taskinputs) > 0:
                    raise TypeError(f"Argument given by name {θname} "
                                    "and position.")
                # if not isinstance(taskinputs, θtype):
                #     # Cast to correct type
                #     taskinputs = cast(taskinputs, θtype)
                taskinputs = ParameterSet({θname: taskinputs})
            else:
                raise ValueError("`params` should be either a dictionary "
                                 "or a path to a parameter file, however it "
                                 "is of type {}.".format(type(params)))
        taskinputs = {**params, **taskinputs}
        sigparams = inspect.signature(cls._run).parameters
        required_inputs = [p.name
                           for p in sigparams.values()
                           if p.default is inspect._empty]
        default_inputs  = {p.name: p.default
                           for p in sigparams.values()
                           if p.default is not inspect._empty}
        if type(cls.__dict__['_run']) is not staticmethod:
            # instance and class methods already provide 'self' or 'cls'
            firstarg = required_inputs.pop(0)
            # Only allowing 'self' and 'cls' ensures we don't accidentally
            # remove true input arguments
            assert firstarg in ("self", "cls")
        if not all((p in taskinputs) for p in required_inputs):
            raise TypeError(
                "Missing required inputs '{}'."
                .format(set(required_inputs).difference(taskinputs)))
        # Add default inputs so they are recorded as task arguments
        taskinputs = {**default_inputs, **taskinputs}

        # Finally, cast all task inputs
        for name, θ in taskinputs.items():
            θtype = cls.inputs[name]
            if isinstance(θ, LazyCastTypes):
                # Can't cast e.g. tasks: they haven't been executed yet
                continue
            elif not isinstance(θ, θtype):
                taskinputs[name] = cast(θ, θtype, 'input')

        return taskinputs

    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name + "(" +  \
            ', '.join(kw+'='+repr(v) for kw,v in self.taskinputs.items()) + ")"
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
        return self.get_desc(self.input_descs)

    @classmethod
    def get_desc(cls, input_descs):
        descset = ParameterSet({
            'taskname': cls.taskname(),
            'inputs': describe(input_descs),
            'module': cls.__module__
        })
        # for k, v in self.input_descs.items():
        #     descset['inputs'][k] = describe(v)
        return descset

    @property
    def digest(self):
        return digest(self.desc)

    def __hash__(self):
        return int(digest(self.desc), base=16)

    @classmethod
    def get_digest(cls, taskinputs):
        return digest(cls.get_desc(cls.get_input_descs(taskinputs)))

    @property
    def input_files(self):
        # Also makes paths relative, in case they weren't already
        store = config.project.input_datastore
        return [os.path.relpath(Path(input.full_path).resolve(),store)
                for input in self.input_descs.values()
                if isinstance(input, DataFile)]

    @classmethod
    def get_input_descs(cls, taskinputs):
        """
        Compares :param:taskinputs with the class's `input` descriptor,
        and constructs the input object.
        This object is what is used to compute the task digest, and therefore
        must reflect any change which would change the task output.
        In particular, this means resolving all file paths, because if
        an input file differs (e.g. a symbolic link points somewhere new),
        than the task must be recomputed.

        """
        # All paths are relative to the input datastore
        # inroot = Path(config.project.input_datastore.root)
        # outroot = Path(config.project.data_store.root)
        inputs = AttrDict()
        try:
            for name, θtype in cls.inputs.items():
                θ = taskinputs[name]
                # Check that θ matches the expected type
                assert (isinstance(θ, θtype)
                        or isinstance(θ, LazyCastTypes))
                        #isinstance(θtype, _mv.multi_rv_generic)  # HACK for dists
                # if isinstance(θ, θtype):
                #     θtype = type(θ)
                #         # Can't just use θtype: it might be a tuple,
                #         # or θ a subclass
                # elif isinstance(θtype, (tuple, list)):
                #     # There are multiple allowable types; get the first match
                #     θtypes = θtype
                #     θtype = None
                #     for T in θtypes:
                #         if (hasattr(T, 'castable')
                #             and isinstance(θ, T.castable)):
                #             θ = θtype(θ)
                #             θtype = type(θ)
                #             break
                #     if θtype is None:
                #         raise TypeError("The argument '{}' is not among the "
                #                         "expected types {}.".format(θtypes)
                # elif (hasattr(θtype, 'castable')
                #       and isinstance(θ, T.castable)):
                #     θ = θtype(θ)
                #     θtype = type(θ)
                # else:
                #     raise TypeError("The argument '{}' is not of the expected "
                #                     "type '{}'.".format(θtype)

                # If the type provides a description, use that, otherwise
                # just take the parameter value.
                inputs[name] = getattr(θ, 'desc', θ)
                # if isinstance(θ, PlainArg):
                #     inputs[name] = θ
                # elif isinstance(θ, File):
                #     # FIXME: Shouldn't this check θtype ?
                #     # inputs[name] = cls.normalize_input_path(θ.full_path)
                #     inputs[name] = θ.desc
                #         # Dereferences links, and returns path relative to inroot
                # else:
                #     inputs[name] = θ
        except KeyError:
            raise TypeError("Task {} is missing required argument '{}'."
                            .format(cls.__qualname__, name))
        return inputs

    @staticmethod
    def from_desc(desc, on_fail='raise'):
        """
        Instantiate a class from the description returned by 'desc'.
        This is especially useful to reload a task definition from disk.

        Parameters
        ----------
        desc: Task description (dict)
            Value returned from Task.desc
        on_fail: 'raise' | 'ignore'
            What to do if the load fails.
        """
        failed = False
        if not utils.is_task_desc(desc):
            failed = True
        else:
            m = importlib.import_module(desc.module)
            TaskType = getattr(m, desc.taskname)
            assert desc.taskname == TaskType.taskname()
            assert set(desc.inputs) == set(TaskType.inputs)
            taskinputs = ParameterSet({})
            for name, θ in desc.inputs.items():
                θtype = TaskType.inputs[name]
                if utils.is_task_desc(θ):
                    # taskinputs[name] = config.TaskTypes[θ.taskname].from_desc(θ)
                    taskinputs[name] = Task.from_desc(θ)
                else:
                    taskinputs[name] = θ
        if failed:
            if on_fail.lower() == 'raise':
                raise ValueError("`desc` is not a valid Task description.")
            else:
                warn("`desc` is not a valid Task description.")
                return None
        return TaskType(**taskinputs)

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

    def load_inputs(self):
        """
        Return a complete input list by loading lazy inputs:
          - files are loaded with `io.load()`
          - upstream tasks are executed
        If necessary, loaded values are cast to their expected type.
        """
        if self._loaded_inputs is None:
            self._loaded_inputs = \
                AttrDict({k: io.load(v.full_path) if isinstance(v, DataFile)
                             else io.load(v.full_path) if isinstance(v, File)
                             else v.run() if isinstance(v, Task)
                             else v
                          for k,v in self.taskinputs.items()})
            for name,value in self._loaded_inputs.items():
                θtype = self.inputs[name]
                v_orig = self.taskinputs[name]
                if isinstance(θtype, type) and issubclass(θtype, PackedTypes):
                    # Inputs are expected as a tuple; we cast each
                    # individually to its expected type
                    self._loaded_inputs[name] = θtype(value)
                elif isinstance(v_orig, Task):
                    # The output value of a task is a tuple, but we are NOT
                    # expecting a tuple (otherwise we would have gone through
                    # the InputTuple branch)
                    # This is only supported if the task returns ONE output
                    assert isinstance(value, tuple)
                    if (isinstance(v_orig, θtype)):
                        #and not isinstance(θtype, _mv.multi_rv_generic)  # HACK for dists
                        # No need to cast anything: it's expected as a task
                        continue
                    elif len(value) > 1:
                        _θtype = (_θtype if isinstance(_θtype, tuple)
                                         else (_θtype,))
                        if any(issubclass(T, Task) for T in _θtype):
                            raise TypeError(
                                f"Input {name} to task {str(self)} must be a task "
                                f"of type {θtype}, but it's of type {type(v_orig)}")
                        else:
                            raise TypeError(
                                f"Task input {type(v_orig)} returned more than "
                                "argument. The input description must define "
                                "a tuple of types to accept its arguments.")
                    elif len(value) == 0:
                        raise NotImplementedError
                    # A sungle return value and no nested types
                    # => automatic unpacking of argument
                    value = value[0]
                    if not isinstance(value, θtype):
                        #and not isinstance(θtype, _mv.multi_rv_generic)  # HACK for dists
                        value = cast(value, θtype, 'input')
                    self._loaded_inputs[name] = value
                elif not isinstance(value, θtype):
                    self._loaded_inputs[name] = cast(value, θtype, 'input')
        return self._loaded_inputs

    @staticmethod
    def normalize_input_path(path):
        """
        Dereference links: links may change, so in the db record
        we want to save paths to actual files
        Typically these are files in the output datastore, but we
        save paths relative to the *input* datastore.root,
        because that's the root we use to execute the task.
        """
        # TODO: use utils.relative_path ?
        inputstore = config.project.input_datastore
        input = DataFile(path, inputstore)
        inputs[name] = DataFile(
            os.path.relpath(Path(input.full_path).resolve(),
                            inputstore.root),
            inputstore)

InputTypes = InputTypes + (Task,)
LazyCastTypes = LazyCastTypes + (Task,)
LazyLoadTypes = LazyLoadTypes + (Task,)

class RecordedTaskBase(Task):
    """A task which is saved to disk and? recorded with Sumatra."""
    # TODO: Add missing requirements (e.g. write())
    @property
    @abc.abstractmethod
    def outputpaths(self):
        pass

#############################
# Description function
#############################

from collections import Iterable, Sequence, Mapping
import scipy as sp
import scipy.stats
import scipy.stats._multivariate as _mv

from parameters import ParameterSet as ParameterSetBase

dist_warning = """Task was not tested on inputs of type {}.
Descriptions of distribution tasks need to be
special-cased because they simply include the memory address; the
returned description is not reproducible.
""".replace('\n', ' ')

def describe(v):
    """
    Provides a single method, `describe`, for producing a unique and
    reproducible description of a variable.
    Description of sequences is intentionally not a one-to-one. From the point
    of view of parameters, all sequences are the same: they are either sequences
    of values we iterate over, or arrays used in vector operations. Both these
    uses are supported by by `ndarray`, so we treate sequences as follows:
        - For description (this function), all sequences are converted lists.
          This has a clean and compact string representation which is compatible
          with JSON.
        - When interpreting saved parameters, all sequences (which should all
          be lists) are converted to `ndarray`.
    Similarly, all mappings are converted to `ParameterSet`.

    This function is essentially one big if-else statement.
    """
    if isinstance(v, PlainArg) or v is None:
        return v
    elif isinstance(v, Sequence):
        r = [describe(u) for u in v]
        # OK, so it seems that Sumatra is ok with lists of dicts, but I leave
        # this here in case I need it later. Goes with "arg" test for Mappings
        # if not all(isinstance(u, PlainArg+(list,)) for u in r):
        #     # Sumatra only supports (nested) lists of plain args
        #     # -> Convert the list into a ParameterSet
        #     r = ParameterSet({f'arg{i}': u for i,u in enumerate(r)})
        return r
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, Mapping):  # Covers ParameterSetBase
        # I think Sumatra only supports strings as keys
        r = ParameterSet({str(k):describe(u) for k,u in v.items()})
        for k in r.keys():
            # if k[:3].lower() == "arg":
            #     warn(f"Mapping keys beginning with 'arg', such as {k}, "
            #          "are reserved by `smttask`.")
            #     break
            if k.lower() == "type":
                warn("The mapping key 'type' is reserved by Sumatra and will "
                     "prevent the web interface from displaying the "
                     "parameters.")
        return r
    elif isinstance(v, Iterable):
        warn(f"Attempting to describe an iterable of type {type(v)}. Only "
             "Sequences (list, tuple) and ndarrays are properly supported.")
        return v
    # elif isinstance(v, File):
    #     return v.desc
    elif isinstance(v, DataFile):
        return File.get_desc(v.full_path)
    # elif isinstance(v, (Task, StatelessFunction, File)):
    elif hasattr(v, 'desc'):
        return v.desc
    elif isinstance(v, type):
        s = repr(v)
        if '<locals>' in s:
            warn(f"Type {s} is dynamically generated and thus not reproducible.")
        return s

    # scipy.stats Distribution types
    # elif isinstance(v,
    #     (_mv.multi_rv_generic, _mv.multi_rv_frozen)):
    #     if isinstance(v, _mv.multivariate_normal_gen):
    #         return "multivariate_normal"
    #     elif isinstance(v, _mv.multivariate_normal_frozen):
    #         return f"multivariate_normal(mean={v.mean()}, cov={v.cov()})"
    #     else:
    #         warn(dist_warning.format(type(v)))
    #         return repr(v)
    # elif isinstance(v, _mv.multi_rv_frozen):
    #     if isinstance(v, _mv.multivariate_normal_gen):
    #         return f"multivariate_normal)"
    #     else:
    #         warn(dist_warning.format(type(v)))
    #         return repr(v)

    else:
        warn("Task was not tested on inputs of type {}. "
             "Please make sure task digests are unique "
             "and reproducible.".format(type(v)))
        return repr(v)
