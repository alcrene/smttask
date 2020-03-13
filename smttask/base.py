import os
import logging
from warnings import warn
import abc
import inspect
from collections import Iterable, Callable
from pathlib import Path
from attrdict import AttrDict
import numpy as np
from sumatra.projects import load_project
from sumatra.parameters import build_parameters
from mackelab_toolbox.parameters import params_to_lists, digest
import mackelab_toolbox.iotools as io
logger = logging.getLogger()

from numbers import Number
from sumatra.parameters import NTParameterSet as ParameterSet
from sumatra.datastore.filesystem import DataFile
PlainArg = (Number, str)

from . import utils

__ALL__ = ['project', 'File', 'NotComputed', 'Task', 'RecordedTaskBase']

InputTypes = (PlainArg, DataFile)
    # Append to this all types that can be used as task inputs
LazyLoadTypes = (DataFile,)
    # Append to this types which aren't immediately loaded
    # Not really used at the moment, other than for some assertions
LazyCastTypes = ()
    # Append to this types which aren't immediately cast to their expected type
    # Not really used at the moment, other than for some assertions

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

class Config:
    """If needed, there variables could be overwritten in a project script"""
    def __init__(self):
        # FIXME: How does `load_project()` work if we load mulitple projects ?
        self._project = None
        # PlainArg
        self.cache_runs = False  # Set to true to cache run() executions in memory
        # Can also be overriden at the class level
        self._allow_uncommited_changes = False

    def load_project(self, path=None):
        "`path` is relative; default is to look in current working directory."
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
    def allow_uncommited_changes(self):
        """Only relevant for InMemoryTasks
        By default, even unrecorded tasks check that the repository is clean
        When developing, set this to False to allow testing of uncommited code.
        """
        return self._allow_uncommited_changes
    @allow_uncommited_changes.setter
    def allow_uncommited_changes(self, value):
        assert isinstance(value, bool)
        if value is True:
            warn("Allowing uncommitted changes is meant as a development "
                 "option and does not apply to recorded tasks. If you "
                 "need to allow uncommitted changes on recorded tasks, use "
                 "Sumatra's 'store-diff' option.")
        self._allow_uncommited_changes = value

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

class InputTuple:
    """
    The purpose of this class is to define type inputs for Tasks returning
    multiple values. If an upstream task returns say a float and either an int
    or a float, then one can specify this by writing
    >>> class Downstream(Task):
    >>>     inputs = {'upstream_data': InputTuple(float, (int, float))}
    The resulting type will cast and validate inputs as expected.
    >>> T = InputTuple(float, (int, float))
    >>> # T(4)
    ValueError: InputTuple_float__int_float_ takes exactly 2 arguments
    >>> T(4,4)
    (4.0, 4)

    The constructor uses a bit of metaclass magic so that `InputTuple`
    creates a *subclass* of InputTuple, rather than an instance, with the
    special attribute `types`:
    >>> T.types
    (float, (int, float))
    """
    def __new__(cls, *types):
        typenames = []
        for T in types:
            if isinstance(T, type):
                typenames.append(T.__qualname__)
            elif (isinstance(T, tuple)
                  and all(isinstance(_T, type) for _T in T)):
                typenames.append('_' + '_'.join(
                    [_T.__qualname__ for _T in T]) + '_')
            else:
                raise ValueError("Arguments to `InputTuple` must be types "
                                 "(or tuples of types).")
        name = 'InputTuple_' + '_'.join(typenames)
        T = type(name, (InputTuple,), {'types': types})
        T.__new__ = cls.__subclassnew__
        T.__init__ = cls.__subclassinit__
        return T
    def __subclassnew__(cls, *args):
        if len(args) != len(cls.types):
            raise ValueError("{} takes exactly {} arguments."
                             .format(cls.__name__, len(cls.types)))
        newargs = (a if isinstance(a, T) else cast(a, T)
                   for a, T in zip(args, cls.types))
        return tuple.__new__(cls, newargs)
    def __subclassinit__(self, *args):
        tuple.__init__(self)
    # TODO: Is it possible to get `isinstance` to work against a tuple ?
InputTypes = InputTypes + (InputTuple,)

class File(abc.ABC):
     """Use this to specify a dependency which is a filename."""
     @property
     @abc.abstractmethod
     def root(self):
         raise NotImplementedError("Root depends on whether file is for input "
                                   "or output")
     def __init__(self, filename):
         # outroot = Path(config.project.data_store.root)
         self.filename = Path(filename)
         if self.filename.is_absolute():
             self.filename = utils.relative_path(self.root, filename)
         # self.inputfilename = utils.relative_path(inroot, filename)
     def __str__(self):
         return str(self.filename)
     def __repr__(self):
         return "File({})".format(self.filename)
     def __hash__(self):
         return int(digest(self.desc), base=16)
     @property
     def full_path(self):
         return self.root.joinpath(self.filename)
     def get_desc(self, filename):
         filename = Path(filename)
         if not filename.is_absolute():
             filename = self.root.joinpath(filename)
         return ParameterSet({
             'input type': 'File',
             'filename': Task.normalize_input_path(
                filename)
         })
     @property
     def desc(self):
         return self.get_desc(self.filename)
class InputFile(File):
    @property
    def root(self):
        return Path(config.project.input_datastore.root)
class OutputFile(File):
    @property
    def root(self):
        return Path(config.project.data_store.root)
InputTypes = InputTypes + (File,)
LazyLoadTypes = LazyLoadTypes + (File,)

class StatelessFunction:
    """
    Use this to specify a dependency which is a function.
    The function must be stateless, and at some point this may check raise an
    if a class method is specified.
    Please don't do self-defeating undectable things, like using global module
    variables inside your function.

    We use `inspect.getsource` to compute the dependency hash, so any source
    code change will invalidate previous cached results (as it should).
    Changing even the name of the function, even for a lambda function,
    suffices to invalidate the cache.
    """
    def __init__(self, f, name=None):
        """
        If you use a nameless function (i.e. lambda) for `f`, consider setting
        the name attribute after creation to something more human-readable.

        Parameters
        ----------
        f: Callable
            You must ensure that `f` is stateless; the initializer is unabl
            to verify this.
        name: str
            Defaults to `f.__qualname__`, or if that fails, `str(f)`.

        Examples
        -------
        >>> x = StatelessFunction(np.arange)
        >>> x.name
        "arange"
        >>> y = StatelessFunction(lambda t: t**2)
        >>> y.name
        "<function <lambda> at 0x7f9bf42abb00>"
        >>> z = StatelessFunction(lambda t: t**2)'
        >>> y.name
        "y"
        """
        if not isinstance(f, Callable):
            raise ValueError("`f` argument must be callable.")
        self.f = f
        if name == "<lambda>":
            if name is None:
                self.name = str(f)
            else:
                self.name = "<lambda {}>".format(name)
            warn("Using a lambda function as a dependency is fragile "
                 "(e.g. the name to which you assign it changes the hash)."
                 "Consider adding the function to a module with a proper "
                 "definition.\nLambda function name: {}".format(self.name))
        else:
            self.name = (name if name is not None
                         else getattr(f, '__qualname__', str(f)))
    def __str__(self):
        return self.name
    def __repr__(self):
        return '.'.join((self.f.__module__, self.name))
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    def __hash__(self):
        return int(digest(self.desc), base=16)
    @property
    def desc(self):
        return ParameterSet({
            'input type': 'Function',
            'source': inspect.getsource(self.f)
            })
InputTypes = InputTypes + (StatelessFunction,)

class NotComputed:
    pass


import scipy.stats._multivariate as _mv
    # TODO: Make unnecessary.
    # See below in `cast()`, `get_input_descs()`, `load_inputs()`
def cast(value, totype, input_or_output=None):
    """
    For each parameter θ, this method will check whether it matches the
    expected type T (as defined by `self.inputs`). If it doesn't, it checks
    if θ can be cast to that type (i.e. that the T has an attribute
    `castable`, and that `isinstance(θ, T)` is True; in this case
    the value is cast to the appropriate type.
    If the input descriptor defines more than one possible input, left-
    most types in the definition take precedence. All types are checked
    for an exact match before casting is attempted.

    Parameters
    ----------
    value:
        Value to cast
    totype: type | tuple of types
        Type to which to cast. If multiple, left-most takes precedence.
        [NOT TRUE: Casting is only attempted with types with a `castable` attribute;
        `castable` attribute must be a list of types.]
    input_or_output: 'input' | 'output'
        Specify whether we are casting an input or output parameter.
        This is required for File parameters.
    """
    if isinstance(value, (_mv.multi_rv_generic, _mv.multi_rv_frozen)):
        # scipy stats distributions use generator functions and don't work
        # with the casting mechanism
        # (`isinstance()` even fails b/c `totype` is not a type)
        # TODO: Fix this so we don't need to special case each generated class
        return value
    elif isinstance(value, totype):
        # Don't cast if `value` is already of the right type
        return value
    elif isinstance(totype, tuple):
        for T in totype:
            try:
                r = cast(value, T, input_or_output)
            except (ValueError, TypeError):
                pass
            else:
                return r
        raise TypeError("Unable to cast {} to type {}"
                        .format(value, totype))
    else:
        T = totype
        if T is File:
            if input_or_output is None:
                raise ValueError("`input_or_output` must be specified to "
                                 "cast `File` arguments.")
            else:
                T = {'input': InputFile, 'output': OutputFile}[input_or_output]
        return T(value)

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
        return type(self).__qualname__
    def __repr__(self):
        return type(self).__qualname__ + "(" +  \
            ', '.join(kw+'='+repr(v) for kw,v in self.taskinputs.items()) + ")"

    @property
    def desc(self):
        return self.get_desc(self.input_descs)

    @classmethod
    def get_desc(cls, input_descs):
        descset = ParameterSet({
            'taskname': cls.__qualname__,
            'inputs': describe(input_descs)
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
                assert (isinstance(θtype, _mv.multi_rv_generic)  # HACK for dists
                        or isinstance(θ, θtype)
                        or isinstance(θ, LazyCastTypes))
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
                if isinstance(θtype, type) and issubclass(θtype, InputTuple):
                    # Inputs are expected as a tuple; we cast each
                    # individually to its expected type
                    self._loaded_inputs[name] = θtype(value)
                elif isinstance(v_orig, Task):
                    # The output value of a task is a tuple, but we are NOT
                    # expecting a tuple (otherwise we would have gone through
                    # the InputTuple branch)
                    # This is only supported if the task returns ONE output
                    assert isinstance(value, tuple)
                    if (not isinstance(θtype, _mv.multi_rv_generic)  # HACK for dists
                        and isinstance(v_orig, θtype)):
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
                    if (not isinstance(θtype, _mv.multi_rv_generic)  # HACK for dists
                        and not isinstance(value, θtype)):
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
from numbers import Number

from parameters import ParameterSet as ParameterSetBase

dist_warning = """Task was not tested on inputs of type {}.
Descriptions of distribution tasks need to be
special-cased because they simply include the memory address; the
returned description is this not reproducible.
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
    elif isinstance(v, (Task, StatelessFunction, File)):
        return v.desc
    elif isinstance(v, type):
        s = repr(v)
        if '<locals>' in s:
            warn(f"Type {s} is dynamically generated and thus not reproducible.")
        return s
    # elif isinstance(v, File):
    #     return v.desc
    elif isinstance(v, DataFile):
        return File.get_desc(v.full_path)

    # scipy.stats Distribution types
    elif isinstance(v,
        (_mv.multi_rv_generic, _mv.multi_rv_frozen)):
        if isinstance(v, _mv.multivariate_normal_gen):
            return "multivariate_normal"
        elif isinstance(v, _mv.multivariate_normal_frozen):
            return f"multivariate_normal(mean={v.mean()}, cov={v.cov()})"
        else:
            warn(dist_warning.format(type(v)))
            return repr(v)
    elif isinstance(v, _mv.multi_rv_frozen):
        if isinstance(v, _mv.multivariate_normal_gen):
            return f"multivariate_normal)"
        else:
            warn(dist_warning.format(type(v)))
            return repr(v)

    else:
        warn("Task was not tested on inputs of type {}. "
             "Please make sure task digests are unique "
             "and reproducible.".format(type(v)))
        return repr(v)
