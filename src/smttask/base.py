from __future__ import annotations

import sys
import os
import io
import logging
import builtins
from warnings import warn
import abc
import inspect
import importlib
from collections import defaultdict
from collections.abc import Iterable, Collection, Mapping
from pathlib import Path
import json
import textwrap
from collections_extended import setlist, frozensetlist

import numpy as np
from tabulate import tabulate
from tqdm.auto import tqdm
# from parameters import ParameterSet as BaseParameterSet  # Used for type checking, since all ParameterSet types should inherit from this

from sumatra.parameters import build_parameters
from sumatra.datastore.filesystem import DataFile

from . import _utils
from . import iotools
from .config import config
from .hashing import stablehexdigest, stableintdigest
from .typing import SeparateOutputs, json_encoders as smttask_json_encoders
from typing import (NamedTuple, Union, Optional, ClassVar, Any, Type, Callable,
    Generator, Tuple, List, Dict)

# For serialization
from pydantic import ValidationError, PrivateAttr
from pydantic.main import ModelMetaclass
import pydantic.parse
from scityping.pydantic import BaseModel
from scityping.numpy import Array

logger = logging.getLogger(__name__)

__all__ = ['NotComputed', 'Task', 'TaskInput', 'TaskOutput', 'TaskDesc',
           'DataFile']

# Especially when we use workflows, it is easy to end up re-executing class
# definitions, which results in duplicate class types and breaks isinstance
# checks. Rather than require that tasks always be in imported modules, separate
# from the workflow file, we cache Task types based on their full name (module 
# + class name).
created_task_types = {}

# Store instantiatied tasks in memory, so the same task with the same parameters
# yields the same instance. This makes in-memory caching much more useful.
# Each task is under a project name, in case different projects have tasks
# with the same name.
instantiated_tasks = defaultdict(lambda: {})

# TODO: Use the task type caching key in instantiated tasks, alongside digest.
#       Is organizing them under project name still useful ?

class NotComputed:
    pass

class TaskExecutionError(RuntimeError):
    def __init__(self, task: Task, message: str="", *args, **kwargs):
        from .task_types import RecordedTask
        try:
            task_name = type(task).__name__
        except Exception:
            task_name = "???"
        try:
            digest = task.digest
        except Exception:
            digest = "???"
        if isinstance(task, RecordedTask):
            try:
                outputpaths = ("\n" + " "*26
                    ).join((str(p) for p in task.outputpaths.values()))
            except Exception:
                outputpaths = "???"
        else:
            outputpaths = None
        msg = f"\nFailed Task: {task_name}\nFailed Task digest: {digest}"
        if outputpaths:
            msg += f"\nFailed Task output paths: {outputpaths}"
        if message:
            msg += "\n" + message
        super().__init__(msg, *args, **kwargs)


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

# Mixin class; Tasks created by generators inherit from this, which TaskDesc
# uses to modify their serialization.
class GeneratedTask(abc.ABC):
    """
    Task generators always depend on additional "generator args", in addition
    to task arguments (otherwise a plain Task would do).
    They are used as follows::
      
       TaskGenerator(generator args)(task args)
    """
    generator_function: Callable
    generator_args  : tuple  # Args & kwargs to the task _generator_, not the task itself.
    generator_kwargs: dict

    @classmethod
    def get_desc(cls, taskinputs, reason=None) -> GeneratedTaskDesc:
        module_name = getattr(cls, '_module_name', cls.__module__)
        # NB: taskname & module are not used for deserialization, but are
        #     useful metadata when inspecting the task JSON
        return GeneratedTaskDesc(
            taskname=cls.taskname(),
            inputs  =taskinputs,
            module  =module_name,
            reason  =reason,
            generator_module  =cls.generator_function.__module__,
            generator_function=cls.generator_function.__qualname__,
            generator_args    =cls.generator_args,
            generator_kwargs  =cls.generator_kwargs
        )

    @staticmethod
    def generate_task_type(desc: GeneratedTaskDesc) -> Union[Task, Callable[...,Task]]:
        # FIXME: Use a whitelist of modules to avoid executing arbitrary code
        from smttask.decorators import _make_input_class
        m = importlib.import_module(desc.generator_module)
        task_generator = getattr(m, desc.generator_function)

        # Construct a Pydantic model to deserialize the generator args
        # (similar to how we create an TaskInput model to parse task inputs)
        input_parser = _make_input_class(task_generator)
        input_parser._disallowed_input_names.discard("reason")  # Task generators are allowed to define a 'reason' argument – they can pass it explicitely to the wrapped task
        pos_argnames = [nm for nm in input_parser.__fields__
                        if nm not in input_parser._digest_params]
        n_received_args = len(desc.generator_args) + len(desc.generator_kwargs)
        n_argnames = len(pos_argnames)
        assert n_received_args <= n_argnames, (
            f"Task generator {task_generator} expects at most {n_argnames} arguments, "
            f"but it received {n_received_args} arguments.\n"
            f"Expected args           : {pos_argnames}\n"
            f"Received positional args: {str(desc.generator_args)[:3000]}\n"   # Truncation is a safeguard to prevent really 
            f"Received keyword args   : {str(desc.generator_kwargs)[:3000]}")  # long lists from filling the command line history
        pos_kwargs = {nm: val for nm, val in zip(pos_argnames, desc.generator_args)}
        repeated_kwargs = set(pos_kwargs) & set(desc.generator_kwargs)
        assert not repeated_kwargs, (
            "The following arguments were received both as positional and "
            f"keyword argument: {repeated_kwargs}")
        kwargs = dict(input_parser(**pos_kwargs, **desc.generator_kwargs,
                                   digest="none"))  # Passing a dummy 'digest' avoids the need of specifying json_encoders, for a digest we will anyway discard
            # NB: .dict() would also return the _digest_params
        # Call the task generator with the deserialized arguments
        TaskType = task_generator(**kwargs)
        return TaskType


# Task Metaclass is used to implement the task type cache (see
# `created_task_types` above).
## WARNING: The code below was subject to a hotfix for Python 3.11. ##
##   There’s duplication of code and other ugliness                 ##
class TaskMeta(abc.ABCMeta):
    def __new__(metacls, cls, bases, namespace):
        frame = sys._getframe(1)  # Could also do inspect.stack()[1].frame, but that seems more wasteful
        try:
            module = inspect.getmodule(frame)
            if module is None:   # With Python 3.11+ this seems to happen, whereas before the module would be the generic `smttask.base`
                module_name = "fallback module name"
            else:
                module_name = module.__name__
        finally:
            del frame  # Prevent reference cycles; c.f. Python Library Reference->inspect->The interpreter stack
            try:
                del module
            except NameError:
                raise RuntimeError("Could not infer the module in which the "
                                   f"task '{cls}' was defined.")
        if module_name == "fallback module name":
            # With the fallback name, we don’t know if two tasks with the same name are
            # actually the same or from two different modules.
            # So we only allow instantiating such tasks once.
            if (module_name, cls) in created_task_types:
                raise RuntimeError(f"Could not infer the module in which the task '{cls}' was defined. "
                                   "Because of this, it is only allowed to instantiate it once.")
            else:
                Task = super().__new__(metacls, cls, bases, namespace)
                created_task_types[(module_name, cls)] = Task
        else:  # Normal code path
            try:
                Task = created_task_types[(module_name, cls)]
            except KeyError:
                if "__module__" not in namespace:
                    namespace["__module__"] = module_name
                Task = super().__new__(metacls, cls, bases, namespace)
                created_task_types[(module_name, cls)] = Task
        return Task

# TODO: Make a task validator?, e.g. Task[float], such that we can validate
# that tasks passed as arguments have the expected output.

class Task(abc.ABC, metaclass=TaskMeta):
    """
    Task format:
    Use `RecordedTask` or `MemoizedTask` as base class
    Note that input types must always be a Union which includes the `Task` type.
    This is taken care of by the decorator.

    .. Note:: The definition of `Inputs` and `Outputs` classes, and of the
       `_run` method, is taken care of by the function decorators. In any
       case we've conceived, the decorators are an easier, more concise way
       of contructing Tasks.

    .. code-block:: python
    
       class MyTask(RecordedTask):
           class Inputs(TaskInput):
               a: Union[Task,int],
               b: Union[Task, str, float]
           class Outputs(TaskOutput):
               c: str
           @staticmethod
           def _run(a, b):
               c = a*b
               return str(c)

    Inputs: TaskInput (subclass of `pydantic.BaseModel`)
        Dictionary of varname: type pairs. Types can be wrapped as a tuple.
        Inputs will (should?) be validated against these types.
        Certain types (like DataFile) are treated differently.
        TODO: document which types and how.
    outputs: TaskOutput (subclass of `pydantic.BaseModel`)
        Dictionary of varname: format. The type is passed on to `io.save()`
        and `io.load()` to determine the data format.
        `format` may be either a type or format name registered in
        `iotools.defined_formats` or `iotools._format_types`. (See the
        `smttask.iotools` docs for more information.)
        If you don't need to specify the output types, can also be a list.
        Not however that the order is important, so an unordered mapping
        like a set will lead to errors.

    _run:
        Method signature must match the parameters names defined by `Inputs`.
        Since a task should be stateless, `_run()` should not depend on `self`
        and therefore can be defined as a staticmethod.
    """
    cache = None
    logger = None

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
            if isinstance(value, Task):
                raise TypeError(f"Expected an instance of {cls}, but received "
                                f"{value}.")
            if TaskDesc.taskdesk_like(value):
                try:
                    return cls.from_desc(value)
                except (ValidationError, OSError) as e:
                    raise ValidationError from e
            else:
                raise ValueError("Value is not a Task instance or a serialized Task.")

    @abc.abstractmethod
    def _run(self):
        """
        This is where subclasses place their code.
        Returned value must match the shape of self.outputs:
        Either a dict with keys matching the names in self.outputs, or a tuple
        of same length as self.outputs.
        """
        pass

    def __init_subclass__(cls):
        # Prepend the docstring with the input-output schematic
        if {'Inputs', 'Outputs'} <= set(dir(cls)):  # Exclude base classes
            doc = cls.__doc__  # NB: could be `None`
            doc = "\n\n" + doc if doc else ""
            cls.__doc__ = cls.schematic() + textwrap.dedent(doc)

    def __getnewargs__(self):
        logger.warning("Pickle support for tasks is WIP and does not yet work reliably.")
        return (self.taskinputs,)
    def __new__(cls, arg0=None, *, reason=None, **taskinputs):
        """
        Performs two checks:
        1. Allows "constructing" a task with an instance of itself, in which
           case no construction occurs and the instance is simply returned.
        2. A unique task is only instantiated once: if one tries to create
           another of same type with same parameters, the previous instance is
           returned.
        """
        if isinstance(arg0, cls):
            return arg0
        else:
            taskinputs = cls._merge_arg0_and_taskinputs(arg0, taskinputs)
            required_inputs = {kw for kw, field in cls.Inputs.__fields__.items() if field.required}
            if required_inputs <= taskinputs.keys():
                # Standard branch: all required parameters provided. Create new Task
                taskinputs = cls.Inputs(**taskinputs)
                h = taskinputs.digest
                taskdict = instantiated_tasks[config.project.name]
                if h not in taskdict:
                    taskdict[h] = super().__new__(cls)
                    taskdict[h].taskinputs = taskinputs

                return taskdict[h]
            elif taskinputs:
                # Alternative branch: partially specified parameters. Create a new Task *type*, with bound parameters
                logger.debug(f"Binding arguments to create a partial task {cls.taskname()}.\n"
                             f"Provided args: {taskinputs.keys()}  –  Missing args: {required_inputs - taskinputs.keys()}")
                return cls.bind(**taskinputs)
            else:
                raise RuntimeError(f"Cannot instantiate task {cls.taskname()}: "
                                   f"missing task parameters {required_inputs}.")

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
            if self.reason != reason:
                logger.warning("Task was reloaded from cache, but had a "
                               "different 'reason'.\nPrevious reason: "
                               f"{self.reason}\nNew reason: {reason}")
                self.reason = reason
            return

        # TODO: this is already done in __new__
        self.reason = reason
        assert hasattr(self, 'taskinputs')  # Set in __new__
        # self.taskinputs = self._merge_arg0_and_taskinputs(arg0, taskinputs)
        self._loaded_inputs = None  # Where inputs are stored once loaded
        self._run_result = NotComputed

        self._dependency_graph = None
        self.logger = logging.getLogger(f"smttask.Task.{self.name}")

    # TODO?: Also do a version for the Task instance, with the argument types & values ?
    @classmethod
    def schematic(cls):
        """
        Display an ascii-art schematic of the task inputs and outputs. Example:

            time   : TimeAxis │
            decay  : float    │───>| trace  : Array
            seed   : int      │
        """
        # Utility function
        def _center(block: List[str], nlines: int) -> List[str]:
            if len(block) < nlines:
                Δ = nlines - len(block)
                ntop = Δ // 2
                block = [""]*ntop + block + [""]*(Δ-ntop)
                # NB: Also adding the bottom lines ensures the function is idempotent
            return block

        input_block = cls.Inputs.display_block().split("\n")
        output_block = cls.Outputs.display_block().split("\n")

        inwidth = max(len(line) for line in input_block)
        outwidth = max(len(line) for line in output_block)
        nlines = max(len(input_block), len(output_block))

        # Add vertical pipes delimiting middle-facing side of blocks
        input_block = [f"{s:<{inwidth}}" + "│" for s in input_block]
        output_block = ["│" + f"{s:<{outwidth}}" for s in output_block]

        # Center the blocks vertically
        input_block = _center(input_block, nlines)
        output_block = _center(output_block, nlines)

        # Join input and output blocks
        arrow_str = "───>"
        spacer_str = " "*len(arrow_str)
        center_line = nlines // 2
        midwidth = len(arrow_str) + 2  # +2 because of the '│' added to each block

        joined_block = [ins + (arrow_str if i == center_line else spacer_str) + outs
                        for i, (ins, outs) in enumerate(zip(input_block, output_block))]

        # Build header
        fullwidth = inwidth+midwidth+outwidth
        headertxt = f"{cls.__name__} [{','.join(C.__name__ for C in cls.__bases__)}]"
        header1 = f"{headertxt:^{fullwidth}}\n"
        header1 += "─"*fullwidth + "\n"

        inhead = "Inputs"      ; outhead = "Outputs"
        insep = "─"*len(inhead)*2; outsep = "─"*len(outhead)*2
        header2 = f"{inhead:^{inwidth}}{' '*midwidth}{outhead:^{outwidth}}\n"
        header2 += f"{insep:^{inwidth}}{' '*midwidth}{outsep:^{outwidth}}\n"

        # Return
        return header1 + header2 + "\n".join(joined_block)


    def clear(self):
        """If the result of a previous run was cached, deallocate it."""
        self._run_result = NotComputed

    # Task inputs are used for two things:
    # 1) Evaluating the task
    # 2) Computing a digest and recording the task conditions
    # For the purpose of 1), inputs may be changed (e.g. by reloading a partial
    # computation from disk). However, for 2) they must NOT change – the
    # recorded parameters must not depend on whether a partial result existed
    # on disk. The @properties below implement this via private attributes
    # _task_inputs and _orig_taskinputs – the latter is only set if it differs
    # from the first.

    @property
    def taskinputs(self):
        return self._taskinputs
    @property
    def orig_taskinputs(self):
        return getattr(self, '_orig_taskinputs', self.taskinputs)
    @taskinputs.setter
    def taskinputs(self, value):
        if (hasattr(self, '_taskinputs')
            and not hasattr(self, '_orig_taskinputs')):
            # On the first modification, save the orig taskinputs as a copy
            self._orig_taskinputs = self.taskinputs
            if self._orig_taskinputs is value:
                logger.warn("Both the original and new task inputs have the "
                            "same id. This may lead to incorrect task digests.")
        self._taskinputs = value

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
            arg0 = config.ParameterSet(arg0)
        elif type(arg0) is TaskInput:
            # Non subclassed TaskInput; re-instantiate with correct Inputs class to catch errors
            arg0 = cls.Inputs(**arg0.dict()).dict()
        elif isinstance(arg0, cls.Inputs):
            arg0 = arg0.dict()
        else:
            raise ValueError("Use keyword arguments to specify task inputs. "
                             "A single positional argument may be provided, "
                             "but it must either be:\n"
                             "1) a ParameterSet (dictionary) of input values;\n"
                             "2) a file path to a ParameterSet;\n"
                             "3) a TaskInput object.\n"
                             f"Instead, the intializer for {cls.__name__} "
                             f"received a value of type '{type(arg0)}'.")
        taskinputs = {**arg0, **taskinputs}
        
        return taskinputs

    def __dir__(self):  # Add fields in taskinputs to autocomplete (these are made accessible by __getattr__)
        return sorted(set(self.__dict__) | set(self.taskinputs.__fields__))
    def __getattr__(self, attr):
        if attr not in ('taskinputs', '_run_result', '_taskinputs', '_orig_taskinputs'):
            try:
                return getattr(self.taskinputs, attr)
            except AttributeError:
                pass
            try:
                return getattr(self._run_result, attr)
            except AttributeError:
                pass
        raise AttributeError(f"Task {self.name} has no attribute '{attr}'.")
    def __str__(self):
        return self.describe(indent=None,
                             param_names_only=True, print=False)
    def __repr__(self):
        return self.name + "(" +  \
            ', '.join(kw+'='+repr(v) for kw,v in self.taskinputs) + ")"

    def describe(self, indent=2, type_join_str=" | ",
                 param_names_only: bool=None, print: bool=True):
        """
        A more human-friendly representation of the Task parameters.

        Parameters
        ----------
        indent: int | str | None
            Behaves as with `json.dump`: integer values print each parameter on
            a new line, indented by the given amount. A value of 'None' produces
            the most compact representation: parameters all on the same line,
            separated by a comma and space (', '). String values are used as-is
            to join the parameter description strings.
        type_join_str: str
            If a parameter accepts multiple types, this is the string used to
            join them.
        param_names_only: bool | None
            True: Show only the parameter names.
            False: Show both the parameter names and types.
            None (default): Equivalent to True if indent=None, False otherwise.
        print: bool
            Whether to print the result instead of returning it.
            Default is True.

        Returns
        -------
        str (if `print` == False)
            : String containing the Task name, its parameter names and their
              expected types.

        Raises
        ------
        ValueError
            : If `indent` is not of an expected type.
        """
        # TODO: - Allow printing different types on different lines, with
        #         proper indentation.
        #       - Allow shortening types, so e.g. scityping.functions.PureFunction
        #         shows up as PureFunction. Note that this should work with
        #         type args too, e.g. scityping.functions.PureFunction[[scityping.numpy.Array], float
        #       - Don't print the 'Task' type which each parameter accepts ?
        # Resolve arguments
        if isinstance(indent, str):
            join_str = indent
        elif isinstance(indent, int):
            indent = max(0, indent)  # In case `indent` is negative
            join_str = "\n" + " "*indent
        elif indent is None:
            join_str = ", "
        else:
            raise ValueError("`join_str` must either be an int, str or None.")
        if param_names_only is None:
            param_names_only = (indent is None)
        # Construct the string
        s = self.name
        if indent is None:
            s += '('
        else:
            s += join_str
        if param_names_only:
            s += join_str.join(self.Inputs.__fields__)
        else:
            s += join_str.join(
                f"{field.name}: {self._describe_field_type(field, type_join_str)}"
                for field in self.Inputs.__fields__.values()
            )
        if indent is None:
            s += ')'
        # Print or return the string
        if print:
            builtins.print(s)
        else:
            return s
    def _describe_field_type(self, field, type_join_str=" | "):
        if field.sub_fields:
            return type_join_str.join(
                self._describe_field_type(f, type_join_str=type_join_str)
                for f in field.sub_fields)
        else:
            return str(field.type_)

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
        # Use the original inputs, so description is consistent
        return self.get_desc(self.orig_taskinputs, self.reason)
    @classmethod
    def get_desc(cls, taskinputs, reason=None):
        module_name = getattr(cls, '_module_name', cls.__module__)
        return TaskDesc(taskname=cls.taskname(),
                        inputs  =taskinputs,
                        module  =module_name,
                        reason  =reason)

    @property
    def digest(self):
        return self.taskinputs.digest
    @property
    def hashed_digest(self):
        return self.taskinputs.hashed_digest
    @property
    def unhashed_digests(self):
        return self.taskinputs.unhashed_digests

    def __hash__(self):
        return hash(self.taskinputs)
        # return stableintdigest(self.desc.json())

    @property
    def inroot(self):
        """Root of the input datastore.
        This is where tasks look to to recover results of past runs.
        """
        return Path(config.project.input_datastore.root)
    @property
    def outroot(self):
        """Root of the output datastore.
        This is where task results are permanently archived tasks.
        """
        return Path(config.project.data_store.root)

    @property
    def has_run(self):
        """
        Returns True if a cached result (either in memory or on disk) exists
        and would be used on a call to `run()`.
        """
        return (self._run_result is not NotComputed
                or self.saved_to_input_datastore)
    @property
    def saved_to_datastore(self):
        """
        Return True if the outputs are saved to the _output_ data store.

        .. remark:: Checks for existence of files on disk, irrespective
           of whether this task has been executed or whether its output
           would match those files.
        """
        return all(path.exists() for path in self.outputpaths.values())
        # return all((self.outroot/path).exists()
        #            for path in self.relative_outputpaths.values())
    @property
    def saved_to_input_datastore(self):
        """
        Return True if links matching the outputs exist the _input_ data store.

        .. Remark:: Checks for existence of files/symlinks on disk, irrespective
           of where they point to, whether this task has been executed or
           whether its output would match those files.
        """
        return all(path.exists() for path in self.resultpaths.values())
        # return all((self.inroot/path).exists()
        #            for path in self.relative_outputpaths.values())


    # FIXME?: inconsistent names input_files & outputpaths
    @property
    def input_files(self):
        # Also makes paths relative, in case they weren't already
        store = config.project.input_datastore
        return [os.path.relpath(Path(input.full_path).resolve(),store)
                for _, input in self.taskinputs
                if isinstance(input, DataFile)]

    @property
    def relative_outputpaths(self):
        return self.Outputs.outputpaths(self)
    @property
    def outputpaths(self):
        """Permanent paths to which task results are *saved*.
        These are NOT the same as the paths from whih results are *retrieved*;
        for those, use `resultpaths`.
        """
        return {name: self.outroot/path
                for name, path in self.relative_outputpaths.items()}
    @property
    def resultpaths(self):
        """Paths from which task results are *retrieved*.
        These are symbolic links, and will be updated if the task is re-run.
        """
        return {name: self.inroot/path
                for name, path in self.relative_outputpaths.items()}

    def parse_result(self, result: Any) -> TaskOutput:
        """
        Parse the task result as an object of type `cls.Outputs`.
        This is most useful when a task has multiple output variables, as this
        provides a namespace where each is retrievable by name.
        """
        return self.Outputs.parse_result(result, _task=self)
        
    @classmethod
    def bind(cls, **kwargs) -> Type[Task]:
        """
        Bind task parameters to given values.
        
        Return a new Task *type*, where the provided parameter values are
        used as defaults.
        
        .. Note:: Although primarily intended to bind argument values, in
           fact the provided values are only used as defaults. Therefore
           they can still be changed when the task is instantiated.
        """
        newname = cls.taskname()
        ## Create the new TaskInput class ##
        # Retrieve the types for each variable we want to bind
        # NB: To avoid errors, we need to redefine the annotation for each
        #     variable we assign a default value to.
        annotations = {}
        for nm in kwargs:
            try:
                field = cls.Inputs.__fields__[nm]
            except KeyError:
                raise KeyError(f"Cannot assign a default value to '{nm}': it is not "
                               f"the name of any input parameter of task {cls}.")
            else:
                annotations[nm] = field.type_
        # NB: Default values are set by assigning them to a class variable
        NewInputs = ModelMetaclass(f"{newname}.Inputs", (cls.Inputs,),
                                   {**kwargs, '__annotations__': annotations})
        ## Create the new Task ##
        NewTask = abc.ABCMeta(newname, (cls,), {'Inputs': NewInputs})
        return NewTask
        
    partial = bind  # Alternative name for `bind`, to match the naming in functools
        
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

        Raises
        ------
        ValidationError

        OSError:
            If `desc` is an invalid path.
        """
        # TODO: Use TaskDesc.looks_compatible to avoid unnecessary calls to
        #       TaskDesc.load() -> They make the stack trace confusing
        try:
            desc = TaskDesc.load(desc)
        except (ValidationError,
                OSError, pydantic.parse.json.JSONDecodeError) as e:
            if on_fail.lower() == 'raise':
                if isinstance(e, ValidationError):
                    raise e
                else:
                    raise ValidationError([e], TaskDesc)
            else:
                warn("`desc` is not a valid Task description.")
                return None

        # FIXME: Use a whitelist of modules to avoid executing arbitrary code
        if isinstance(desc, GeneratedTaskDesc):
            TaskType = GeneratedTask.generate_task_type(desc)
            # task_gen will not be a Task in general, but a Callable
        else:
            m = importlib.import_module(desc.module)
            TaskType = getattr(m, desc.taskname)
            assert isinstance(TaskType, type) and issubclass(TaskType, Task)
            assert desc.taskname == TaskType.taskname()

        taskinputs = config.ParameterSet({})
        for name, θ in desc.inputs:
            try:
                subdesc = TaskDesc.load(θ)
            except (ValidationError,
                    OSError, pydantic.parse.json.JSONDecodeError,
                    TypeError):
                # Value is not a task desc: leave as is
                taskinputs[name] = θ
            else:
                # Value is a task desc: replace by the task
                taskinputs[name] = Task.from_desc(subdesc)
        # # To allow task generators to have positional variational arguments, we 
        # # need to split off positional arguments.
        # # (I ended up not needing this; maybe it is still useful ?
        # args = []
        # for name, info in inspect.signature(TaskType).parameters.items():
        #     if info.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
        #         args.append(taskinputs.pop(name))
        #     elif info.kind is inspect.Parameter.VAR_POSITIONAL:
        #         args.extend(taskinputs.pop(name))
        return TaskType(**taskinputs, reason=desc.reason)

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
        return self._loaded_inputs._input_values

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

    def save(self, path, allow_overwrite=False, **json_kwargs) -> Path:
        """
        Save a task description. The resulting file can be run from the command
        line with ``smttask run [taskdesc]``.
        Extra keyword arguments are passed on to the TaskDesc's JSON exporter.
        For example, if the exported description should human-readable,
        `indent=2` is a useful options.
        
        Returns the path to the saved file.
        """
        return self.desc.save(path, allow_overwrite, **json_kwargs)

    def get_output(self, name=""):
        """
        Return the value of the output associated with name `name`; if the
        Task has only one unnamed output, specifying the name is not necessary.
        This will never trigger computation: if the task result is neither in
        memory nor on disk, raises FileNotFoundError.
        """
        if self._run_result is not NotComputed:
            return getattr(self._run_result, name)
        else:
            # inroot = Path(config.project.input_datastore.root)
            abspath = self.resultpaths[name]
            output_type = self.Outputs._output_types(self)[name]
            with open(abspath, 'r') as f:
                json_data = json.load(f)
            validate = getattr(output_type, "validate", None)
            if validate:
                return validate(json_date)
            else:
                try:
                    logger.debug(f"Type {output_type} does not provide a `validate` method; attempting a direct cast.")
                    return output_type(json_data)
                except (TypeError, ValidationError):
                    logger.debug("Direct cast failed:; returning JSON data as-is.")
                return json_data
            # try:
            #     parse_file = output_type.parse_file
            # except AttributeError:
            #     raise AttributeError(
            #         "`get_output` only supports parsing Pydantic models.")
            # return parse_file(inroot/path)

    def _parse_output_file(self, path) -> dict:
        """
        Returns
        -------
        dict representation of the task

        Raises
        ------
        FileNotFoundError:
            If the file given by *input data store root*/`path` doesn't exist.
        """
        inroot = Path(config.project.input_datastore.root)
        # Next line copied from pydantic.main.parse_file
        output = pydantic.parse.load_file(
            inroot/path,
            proto=None, content_type='json', encoding='utf-8',
            allow_pickle=False,
            json_loads=self.Outputs.__config__.json_loads)
        return output


# ============================
# Serialization
# ============================

# The value returned by the json encoder is used to compute the task digest, and
# therefore must reflect any change which would change the task output.
# In particular, this means resolving all file paths, because if
# an input file differs (e.g. a symbolic link points somewhere new),
# than the task must be recomputed.
def json_encoder_InputDataFile(datafile):
    return str(_utils.relative_path(src=config.project.input_datastore.root,
                                    dst=datafile.full_path))

def json_encoder_OutputDataFile(datafile):
    return str(_utils.relative_path(src=config.project.data_store.root,
                                   dst=datafile.full_path))

def make_digest(hashed_digest: str, unhashed_digests: Optional[Dict[str, str]]=None) -> str:
    if unhashed_digests is None:
        return hashed_digest
    else:
        return hashed_digest + ''.join(f"__{nm}_{val}"
                                       for nm,val in unhashed_digests.items())

class ValueContainer(BaseModel, abc.ABC):
    "Base class for `TaskInput` and `TaskOutput`."

    # TODO: Is there a way to define this as the class' __repr__ without using
    #       a metaclass ?
    @classmethod
    def describe(cls):
        return (f"{cls.__name__} ("
                + ', '.join(f"{nm}: {field._type_display()}"
                            for nm, field in cls.__fields__.items())
                + ")")

    def __repr__(self):
        return self.describe()

    @classmethod
    def __pretty__(cls, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
        """
        Used by devtools (https://python-devtools.helpmanual.io/) to provide a
        human readable representations of objects
        """
        yield cls.__name__ + '('
        yield 1
        for name, field in cls.__fields__.items():
            yield name + ': '
            yield fmt(field._type_display())
            yield ','
            yield 0
        yield -1
        yield ')'

    @classmethod
    def _type_display(cls, T: Type) -> str:
        # NB: Pydantic provides _type_display, but we want something more compact:
        #     It should fit it half a column, so we can display input --> output
        # TODO: Add indent for nested types on multiple lines. Eg. {int: {int: float}}
        origin = getattr(T, '__origin__', None)
        if origin is Union:
            args = [cls._type_display(arg)
                    for arg in T.__args__ if arg is not Task]
            Tstr = " | ".join(args)  # NB: The space before '|' is non-breaking, to encourage line breaks after the pipe instead of before
        elif origin is list:
            if hasattr(T, '__args__'):
                Tstr = f"List[{','.join(cls._type_display(arg) for arg in T.__args__)}]"
            else:
                Tstr = "list"
        elif origin is tuple:
            if hasattr(T, '__args__'):
                Tstr = f"Tuple[{','.join(cls._type_display(arg) for arg in T.__args__)}]"
            else:
                Tstr = "tuple"
        elif origin is dict:
            if hasattr(T, '__args__'):
                try:  # Paranoia guard: I don’t have a reason to think that __args__ may have anything else than length 2
                    KT, VT = T.__args__
                except TypeError:
                    Tstr = str(T)
                else:
                    Tstr = "{" + cls._type_display(KT) + ": " + cls._type_display(VT) + "}"
            else:
                Tstr = "dict"
        elif T is type(None):
            Tstr = "None"
        elif T is Ellipsis:
            Tstr = "..."
        else:
            # We use __name__ instead of __qualname__ because we want the type to be as compact as possible
            # Users can always look at the signature to get the full type
            Tstr = getattr(T, '__name__', str(T))
        return Tstr

    @classmethod
    def _param_desc(cls, field: ModelField):
        internal_name = (f"stored as '{field.name}'" if field.name != field.alias
                         else "")
        optional = "optional" if not field.required else ""
        default = f"default: {repr(field.default)}" if field.default is not None else ""
        extra = ", ".join(s for s in (internal_name, optional, default) if s)
        if extra:
            extra = f" ({extra})"
        name = field.alias if field.alias else "<no name>"
        Tstr = cls._type_display(field.type_)
        return (name,
                "\n".join(textwrap.wrap(": " + Tstr, 40,
                                        subsequent_indent="  ",
                                        break_long_words=False)),
                extra)

    @classmethod
    def display_block(cls):
        digest_params = getattr(cls, '_digest_params', set())
        block = tabulate(
            [cls._param_desc(field)
             for nm, field in cls.__fields__.items()
             if nm not in digest_params],
            tablefmt='plain')
        return block

class TaskInput(ValueContainer):
    """
    Base class for task inputs.
    Each Task defines a new TaskInput class, which subclasses this one.

    The following attributes are disallowed and will raise an error if they
    are part of the list of inputs:

    - :attr:`reason`
    - :attr:`digest`
    - :attr:`arg0`
    - :attr:`_digest_length`
    - :attr:`_unhashed_params`
    - :attr:`_disallowed_input_names`

    The class variable `_unhashed_params` is a list of attribute names which
    are not hashed as part of the digest, but appended to it in the format
    '{hash}__{θ1 name}_{θ1 val}__{θ2 name}_{θ2 val}...'. This is used by the
    Iterative task, to be able to recognize runs with the same parameters but
    just different numbers of iterations.

    .. TODO:: We should check that inputs which are Task instances have
       appropriate output type.
    """
    ## Class variables
    _disallowed_input_names = {'arg0', 'reason', '_unhashed_params',
                               '_disallowed_input_names', '_digest_length',
                               'hashed_digest', 'unhashed_digests'}
    _digest_length = 10  # Length of the hex digest
    _unhashed_params: ClassVar[Set[str]] = frozensetlist()  # Ordered set; compute_unhashed_digest() relies on order being predictable. Frozen, because can be shared with child classes
    _digest_params: ClassVar[Set[str]] = frozensetlist(("digest", "hashed_digest", "unhashed_digests"))
    ## Internally managed attributes
    # `digest` is set immediately in __init__, so that it doesn't change if the
    # inputs are changed – we want to lock the digest to the values used to
    # initialize the task.
    digest: str = None
    hashed_digest: str = None
    unhashed_digests: dict = {}

    class Config:
        # The base type is used to construct inputs when deserializing;
        # since it defines no attributes, the default config `extra`='ignore'
        # would discard all inputs. 'allow' indicates to keep them all.
        # Subclasses however should set this to 'forbid' to catch errors.
        # TODO?: Use __init_subclass__ to set 'forbid' automatically ?
        # During instantiation, if a Task receives a non-subclassed TaskInput,
        # it should re-instantiate that to its own TaskInput subclass.
        extra = 'allow'
        arbitrary_types_allowed = True
        allow_mutation = False
            # Tasks digests depend only on the inputs at CREATION TIME, so
            # inputs must not be changed. Modifying values in place is very
            # likely to lead to undesirable (read: irreproducible) behaviour.
            # This can be worked around by creating an entirely new TaskInput
            # object and assigning that as `.taskinputs` (this is how we
            # reload from partial computations). It is assumed that the person
            # doing this would know what they were doing.
        # validate_on_assignment = True
        #     # Because we allow changing inputs, e.g. when continuing from a
        #     # previous IterativeTask. Not sure if this is the best way.
        json_encoders = {**smttask_json_encoders,  #Re-added in decorators.py, to reflect dynamic changes to json_encoders
                         DataFile: json_encoder_InputDataFile,
                         Task: lambda task: task.desc.dict()}

    def __init__(self, *args, **kwargs):
        ## Validity checks
        # Ideally these checks would be in the metaclass/decorator
        for nm in self._disallowed_input_names:
            if (nm in self.__fields__ and nm not in TaskInput.__fields__):
                raise AssertionError(
                    f"A task cannot define an input named '{nm}'.")
        for nm in self._unhashed_params:
            if nm not in self.__fields__:
                raise AssertionError(
                    f"The parameter name {nm} is excluded from the hash, but "
                    "not part of the model.")
        ## Initialize
        # HACK: Because Pydantic does not preserve order for extra parameters (https://github.com/samuelcolvin/pydantic/issues/1234)
        #       we assign them in order after the class has been created
        #       (The extra parameters appear when we initialize a TaskInput object
        #       with the base class.)
        # HACK #2: To do this, we temporarily set 'allow_mutation' to True, and
        #       reset it to the value in Config afterwards.
        extra_kwargs = {}
        for k in list(kwargs):
            if k not in self.__fields__:
                extra_kwargs[k] = kwargs.pop(k)
        # if not extra_kwargs:    # Useful for investigating creating of inputs;
        #     breakpoint()        # `if` skips temporary TaskInput used during deserialization
        super().__init__(*args, **kwargs)
        # vvvv Enable mutations vvvv
        old_allow_mutation = self.__config__.allow_mutation
        self.__config__.allow_mutation = True
        for k, v in extra_kwargs.items():
            setattr(self, k, v)
        ## Compute digest
        if self.digest is None:
            self.hashed_digest = self.compute_hashed_digest()
            self.unhashed_digests = self.compute_unhashed_digests()
            self.digest = make_digest(self.hashed_digest, self.unhashed_digests)
        self.__config__.allow_mutation = old_allow_mutation
        # ^^^^ Reset mutability as before ^^^^

    def compute_hashed_digest(self):
        """
        .. Warning:: You probably want to use the `hashed_digest` attribute
           instead of this function. Since tasks may modify their inputs (e.g.
           a task integrating a model may modify the data stored in the model),
           dynamically computed digests are not stable. That is why digests
           are computed immediately upon TaskInput creation, and stored in
           the `digest`, `hashed_digest` and `unhashed_digest` attributes.
        """
        # I haven't found an obvious way to remove all the digest keys from nested models
        # So instead, since the `json` method is just `dict` + `json_dumps`,
        # we build the dictionary ourselves and then serialize it the same way
        # Pydantic does.
        data = {}
        for k, v in self:
            if k in self._unhashed_params | self._digest_params:
                continue
            elif isinstance(v, (Task, TaskInput)):
                v = v.compute_hashed_digest()
            elif isinstance(v, dict):
                # TODO: Check dict for taskdesc fields ? Cheaper than always attempting construction
                # TODO: Do we even want to create throwaway tasks here ? Seems wasteful (advantage is better consistency of nested digests)
                try:
                    task = Task.from_desc(v)
                except (ValidationError, TypeError):
                    # Not a task description
                    v = list(v.items())
                else:
                    v = task.compute_hashed_digest()
            data[k] = v
        # See pydantic.main:BaseModel.json()
        if self.__custom_root_type__:
            data = data[ROOT_KEY]
        json = self.__config__.json_dumps(data, default=self.digest_encoder)
        return stablehexdigest(json)[:self._digest_length]

    def compute_unhashed_digests(self):
        """
        .. warning:: You probably want to use the `unhashed_digest` attribute
           instead of this function. Since tasks may modify their inputs (e.g.
           a task integrating a model may modify the data stored in the model),
           dynamically computed digests are not stable. That is why digests
           are computed immediately upon TaskInput creation, and stored in
           the `digest`, `hashed_digest` and `unhashed_digest` attributes.
        """
        return {nm: str(getattr(self, nm))
                for nm in self._unhashed_params}

    # Exclude 'digest' attribute when iterating over parameters
    def __iter__(self):
        for attr, value in super().__iter__():
            if attr not in self._digest_params:
                yield (attr, value)

    def load(self, progbar: int=0):
        """
        Return a copy, with all lazy inputs resolved:
          - files are loaded with `io.load()`
          - upstream tasks are executed
        If necessary, loaded values are cast to their expected type.
        
        Parameters
        ----------
        progbar: Whether to show a progress bar (one step per input item).
            Mostly intended for internal use for the Join task generator.
            A progress bar is shown iff `progbar` == 1; the value of `progbar`
            is subtracted by 1 for each recursion level.
        """
        from .workflows import ParamColl
        # Resolve lazy inputs
        def load_collection_elements(coll: Collection, progbar: int=0) -> Collection:
            # Recall: A Collection is a sized iterable: includes tuple, set, list, but not generators
            if isinstance(coll, tuple(config.terminating_types)):
                it = None
            elif isinstance(coll, ParamColl):
                # ParamColls are designed to be passed as-is, and should be unpacked within the task
                it = None
            elif _utils.is_parameterset(coll):
                # ParameterSet doesn't support initialization with a generator
                it = {k: load_element(v, progbar-1) for k, v in coll.items()}
            elif isinstance(coll, Mapping):
                it = ((k, load_element(v, progbar-1)) for k, v in coll.items())
            else:
                it = (load_element(v, progbar-1) for v in coll)
            if it:
                if progbar == 1:
                    it = tqdm(it, total=len(coll),
                              desc=f"Loading {type(self).__qualname__}")
                coll = type(coll)(it)
            return coll

        def load_element(v: Any, progbar: int=0):
            return (io.load(v.full_path) if isinstance(v, DataFile)
                    else v.run() if isinstance(v, Task)
                    else load_collection_elements(v, progbar) if isinstance(v, Collection)
                    else v)

        # Use super().__iter__ to include digest params
        it = super().__iter__()
        if progbar == 1:
            # TODO: Would be nicer to use the task's name, rather than hope it's part of the qualified name
            # FIXME: This reports the total without digest params, which should be less confusing
            #        and give more reliable time estimates. But unless digest params are
            #        are loaded last, it will report the wrong number of complete items.
            it = tqdm(it, total=len(self.__fields__.keys() - self._digest_params),
                      desc=f"Loading {type(self).__qualname__}")
        obj = {attr: load_element(v, progbar-1) for attr, v in it}

        # Validate, cast, and return
        return type(self)(**obj)

    @property
    def _input_values(self):
        """
        Save as .dict(), but excludes internal attributes (i.e. 'digest')
        """
        return {k:v for k,v in self if k != 'digest'}

    def __hash__(self) -> int:
        return hash(self.digest)

    def digest_encoder(self, value):
        """
        Specialized encoder for computing digests.
        For NumPy arrays, skips the compression so that digests are consistent
        across machines.
        For other values, uses the BaseModel's default __json_encoder__.
        """
        if type(value) is np.ndarray:
            # Indexed type is inconsequential
            # However, we don't want to catch subclasses of ndarray, since they
            # might have specialized serializers (e.g. sparse arrays, or quantities objects)
            return Array.Data.encode(value, compression='none')
        else:
            return self.__json_encoder__(value)

class TaskOutput(ValueContainer):
    """
    .. rubric:: Output definition logic:

    If a TaskOutput type defines only one output, it expects the result to
    be that output::

        class Outputs:
          x: float

    Expects the task to return a single float, which will be saved with the
    name 'x'. Similarly::

        class Outputs:
          x: Tuple[float, float]

    expects a single tuple, which will be saved with the name 'x'.

    If instead a TaskOutput type defines multiple values, it expects them
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

    .. Warning:: In the interest of brevity, the snippets above are
       incomplete. In a real definition, each type is a union including
       `Task`.

    .. Note:: In contrast to most other situations, we try here to avoid
       raising exceptions. This is because once we've reached the point of
       constructing a `TaskOutput` object, we have already computed the result.
       This may have taken a long time, and we want to do our best to save the
       data in some form. If we can't parse the result, we store the original
       value to allow `write` to save it as-is. It will not be useable in
       downstream tasks, but at least this gives the user a chance to inspect it.
    """
    __slots__ = ('_unparsed_result', '_well_formed', '_task', 'outcome')
    _disallowed_input_names = {'_task', 'outcome'}
    _digest_length = 10  # Length of the hex digest
    _unhashed_params: ClassVar[Set[str]] = frozensetlist()  # Ordered set; compute_unhashed_digest() relies on order being predictable. Frozen, because can be shared with child classes
    _emergency_dumping: bool=PrivateAttr(False)
    _already_dumping: set=PrivateAttr(set())
        # Last two fields are used to deal with recursion in emergency_dump

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {**TaskInput.Config.json_encoders,
                         DataFile: json_encoder_OutputDataFile}

    def __init_subclass__(cls):
        # FIXME?: Is result_type redundant with _output_types ?
        # HACK/FIXME: Getting the task name like this is super fragile
        #     Other class functions require a `_task` argument
        # NB: There are situations where we define subclasses of TaskOutputs
        #     outside of a Task, so we need to allow for that as well.
        name_components = cls.__qualname__.split('.')
        if len(name_components) == 1:
            # Probably a definition outside a Task
            base_name = name_components[0]
        else:
            base_name = name_components[-2]
        if len(cls.__fields__) == 1:
            cls.result_type = next(iter(cls.__fields__.values())).type_
        else:
            cls.result_type = NamedTuple(
                f"{base_name}Result",
                [(nm, field.outer_type_) for nm, field in cls.__fields__.items()]
                )

    # Ideally these checks would be in the metaclass
    # NB: The parsing of result is not defined here, but rather in `parse_result`
    def __init__(self, *args, _task, outcome="", **kwargs):
        if len(args) and not len(kwargs):
            # We can end up here if we try to initialize a TaskOutputs with
            # plain task results (i.e., what is returned by `self.result`)
            if len(args) == 1:
                args = args[0]
            self.parse_result(args, _task)
            return  # EARLY EXIT: __init__ has already been executed inside parse_result
        if len(self.__fields__) == 0:
            raise TypeError("Task defines no outputs. This must be an error, "
                            "because tasks are not allowed to have side-effects.")
        for nm in self._disallowed_input_names:
            if (nm in self.__fields__ and nm not in TaskOutput.__fields__):
                raise AssertionError(
                    f"A task cannot define an output named '{nm}'.")
        if not isinstance(_task, Task):
            raise TypeError("'_task' argument must be a Task instance.")
        if not isinstance(outcome, str):
            # POSSIBILITY: Also accept Tuple[str]
            raise TypeError("'outcome' argument must be a string.")
        object.__setattr__(self, 'outcome', outcome)
        # Reassemble separated outputs, if they are passed separately
        for nm, field in self.__fields__.items():
            # FIXME? Reduce duplication with _outputnames_gen, __iter__
            type_ = field.type_
            if isinstance(type_, type) and issubclass(type_, SeparateOutputs):
                sep_names = type_.get_names(
                    **{k:v for k,v in _task.taskinputs
                       if k in type_.get_names_args}
                )
                # Allow only two possibilities:
                # 1. Sub values are all passed separately -> `nm` not in `kwargs`
                # 2. None of the values are passed separately -> `nm` in `kwargs` (checked by Pydantic validator)
                if any(sep_nm in kwargs for sep_nm in sep_names):
                    assert all(sep_nm in kwargs for sep_nm in sep_names)
                    assert nm not in kwargs
                    kwargs[nm] = tuple(kwargs[sep_nm] for sep_nm in sep_names)
                    for sep_nm in sep_names:
                        del kwargs[sep_nm]
        # Set public attributes with Pydantic initializer
        super().__init__(*args, **kwargs)
        # Set hidden attributes directly
        object.__setattr__(self, '_task', _task)
        # (If we made it here, output arguments were successfully validated)
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
            i = 0
            for nm, field in self.__fields__.items():
                type_ = field.type_
                if isinstance(type_, type) and issubclass(type_, SeparateOutputs):
                    i += len(type_.get_names(**{k:v for k,v in self._task.taskinputs
                                                if k in type_.get_names_args}))
                else:
                    i += 1
            return i
        else:
            if self._unparsed_result is None:
                return 0
            else:
                # TODO? Not sure what the best value would be to return.
                #   The number of files that would be produced by `save` ?
                return 1

    def __iter__(self):
        # FIXME: At present, returned names must be synced w/ _outputnames_gen
        for nm, val in super().__iter__():
            type_ = self.__fields__[nm].type_
            # Special case for separate outputs
            if isinstance(type_, type) and issubclass(type_, SeparateOutputs):
                sep_names = type_.get_names(
                    **{k:v for k,v in self._task.taskinputs
                       if k in type_.get_names_args})
                if not isinstance(val, Iterable):
                    warn(f"The output '{nm}' from task '{self._task.name}' "
                         "is intended to be separated, but the received value "
                         f"`{val}` is not iterable.")
                    yield nm, val
                else:
                    if len(val) != len(sep_names):
                        warn(f"The output '{nm}' from task '{self._task.name}' "
                             f"is intended to be separated into {len(sep_names)} "
                             f"values, but {len(val)} values were received.\n"
                             "If more values were received, THIS HAS LIKELY "
                             "LEAD TO DATA LOSS !!!")
                    for sep_nm, sep_val in zip(sep_names, val):
                        if sep_nm in self.__fields__:
                            warn(f"Output name {sub_nm} is associate to both a "
                                 "normal and a separate output.")
                        yield sep_nm, sep_val
            else:
                yield nm, val

    @property
    def hashed_digest(self):
        return stablehexdigest(
            self.json(exclude=self._unhashed_params)
            )[:self._digest_length]
    @property
    def unhashed_digests(self):
        return {nm: str(getattr(self, nm))
                for nm in self._unhashed_params}
    @property
    def digest(self) -> str:
        return make_digest(self.hashed_digest, self.unhashed_digests)

    def __hash__(self):
        return hash(self.digest)

    @property
    def result(self):
        """
        Return the value, as it would have been returned by the Task.

        If there is one output:
            Return the bare value.
        If there is more than one output:
            Return the outputs wrapped in a tuple, in the order they are
            defined in this TaskOutput subclass.
        """
        if not self._well_formed:
            return self._unparsed_result
        elif len(self) == 1:
            nm, val = next(iter(self))
            return val
        else:
            # Use super()'s __iter__ to avoid unpacking SeparateOutputs
            return self.result_type(*(value for attr,value in super().__iter__()))

    @classmethod
    def parse_result(cls, result: Any, _task: Task) -> TaskOutput:
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
        TaskOutput:
            If parsing is successful, the values of `result` are assigned to
            the attributes of TaskOutput.
            If parsing is unsuccessful, the value of `result` is assigned
            unchanged to `_unparsed_values`.

            `write` checks the value of `_unparsed_values` to determine which
            export function to use. If the value is ``None``, standard export
            to JSON is performed, otherwise the `emergency_dump` method is used.
        """
        try:
            taskname = _task.taskname()
        except Exception:
            taskname = ""
        failed = False
        output_fields = cls.__fields__
        if len(output_fields) == 0:
            raise TypeError("Task defines no outputs. This must be an error, "
                            "because tasks are not allowed to have side-effects.")
        elif len(output_fields) == 1:
            # Result is not expected to be wrapped with a tuple.
            nm = next(iter(output_fields))
            result = {nm: result}
        elif not isinstance(result, (tuple, dict)):
            warn(f"Task {taskname} defines multiple outputs, and "
                 "should return them wrapped with a tuple or a dict.")
            failed = True

        if isinstance(result, tuple):
            result = {nm: val for nm,val in zip(output_fields, result)}

        # At this point, either `failed` == True, or `result` is a dict.
        if not failed:
            try:
                taskoutputs = cls(**result, _task=_task)
            except ValidationError as e:
                warn(f"\n\nThe output of task {taskname} was malformed. "
                     "Attempting to cast to the expected output format raised "
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

    def emergency_dump(self, filename, obj):
        """
        A function called when the normal saving function fails.
        If `obj` is an iterable, each element is saved individually.
        """
        root_dump = not self._emergency_dumping
        self._emergency_dumping = True
        # NB: It is possible for `obj` to contain a reference to itself.
        #     We want to avoid that leading to infinite recursion.

        if id(obj) in self._already_dumping:
            return
        self._already_dumping.add(id(obj))

        filename, ext = os.path.splitext(filename)
        try:
            if isinstance(obj, dict):
                for name, el in obj.items():
                    self.emergency_dump(f"{filename}__{name}{ext}", el)
            elif (isinstance(obj, Iterable)
                  and not isinstance(obj, (str, bytes))
                  and not hasattr(obj, '__array__')):  # NumPy array
                for i, el in enumerate(obj):
                    self.emergency_dump(f"{filename}__{i}{ext}", el)
            else:
                warn("An error occured while writing the task output to disk. "
                     "Unceremoniously dumping data at this location, to allow "
                     f"post-mortem: {filename}...")
                self._already_dumping.add(id(obj))
                iotools.save(filename, obj)
        except Exception:
            # Don't let one exception prevent saving the rest
            try:
                obj_str = str(obj)
                if len(obj_str) > 200:
                    obj_str = obj_str[:200] + "…"
            except:
                pass
            else:
                warn(f"Emergency dump faild with object {obj_str}")

        if root_dump:
            # We've exited all recursive calls; the emergency dump is done
            self._emergency_dumping = False
            self._already_dumping.clear()

    def write(self, **dumps_kwargs) -> List[str]:
        """
        Save outputs; file locations are determined automatically.

        **dumps_kwargs are passed on to the model's json encoder.

        Returns a list of absolute paths.
        """
        # If the result was malformed, use the emergency_dump and exit immediately
        try:
            taskname = self._task.taskname()
        except Exception:
            taskname = ""
        if not self._well_formed:
            if self._unparsed_result is None:
                warn(f"{taskname}.Outputs.write: "
                     "Nothing to write. Aborting.")
                return
            outpath = next(iter(self.outputpaths(self._task).values()))
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
        orig_outpaths = self.outputpaths(self._task)
        outpaths = []  # outpaths may add suffix to avoid overwriting data
        for nm, value in self:
            if hasattr(value, 'json'):
                json = value.json(**dumps_kwargs)
            else:
                # Next line copied from pydantic.main.json
                json = self.__config__.json_dumps(
                    value, default=self.__json_encoder__, **dumps_kwargs)
            relpath = orig_outpaths[nm]
            f, truepath = iotools.get_free_file(outroot/relpath, bytes=True)
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

            os.symlink(_utils.relative_path(inpath.parent, truepath),
                       inpath)

        return outpaths

    @classmethod
    def _outputnames_gen(cls, _task):
        yield from cls._output_types(_task)

    @classmethod
    def _output_types(cls, _task):
        # FIXME: At present, returned names must be synced w/ __iter__
        output_types = {}
        for nm, field in cls.__fields__.items():
            type_ = field.type_
            # Special case for separate outputs
            if isinstance(type_, type) and issubclass(type_, SeparateOutputs):
                for sub_nm in type_.get_names(
                    **{k:v for k,v in _task.taskinputs
                       if k in type_.get_names_args}):
                    if sub_nm in cls.__fields__:
                        warn(f"Output name {sub_nm} is associate to both a "
                             "normal and a separate output.")
                    output_types[sub_nm] = type_.item_type
            else:
                output_types[nm] = type_
        return output_types

    @classmethod
    def outputpaths(cls, _task):
        """
        Returns
        -------
        Dictionary of output name: output path pairs.
            Paths are relative to the data store root.
        """
        try:
            taskname = _task.taskname()
        except Exception:
            taskname = ""
        # '_task.digest' uses either Inputs.digest or Outputs.digest, depending
        # on the task, and includes both hashed & unhashed parts
        return {nm: Path(taskname) / f"{_task.digest}_{nm}.json"
                for nm in cls._outputnames_gen(_task)}

class EmptyOutput(BaseModel):
    """
    A special substitute class for the Output object when a Task fails or
    terminates prematurely.

    Attributes:
       - status  (typically one of 'killed', 'failed')

    Properties:
       - result: returns `self`, so that if a Task fails, `task.run()` returns
         an instance of `EmptyOutput`.

    """
    status: str
    def __len__(self):
        return 0
    def __iter__(self):
        raise StopIteration
    @property
    def hashed_digest(self):
        return ""
    @property
    def unhashed_digests(self):
        return "<EmptyOutput>"
    @property
    def digest(self) -> str:
        return "<EmptyOutput>"
    def __hash__(self):
        return hash(self.digest)
    @property
    def result(self):
        return self
    @classmethod
    def parse_result(cls, result, _task):
        raise NotImplementedError
    def write(self, **dumps_kwargs):
        raise NotImplementedError
    @classmethod
    def _outputnames_gen(cls, _task):
        raise NotImplementedError
    @classmethod
    def outputpaths(cls, _task):
        raise NotImplementedError

class TaskDesc(BaseModel):
    taskname: str
    module  : str
    inputs  : TaskInput
    reason  : Optional[str]=None

    class Config:
        json_encoders = TaskInput.Config.json_encoders

    def json(self, *args, encoder=None, **kwargs):
        encoder = encoder or self.inputs.__json_encoder__
        return super().json(*args, encoder=encoder, **kwargs)

    @classmethod
    def looks_compatible(cls, obj):
        """
        Returns True if `obj` is a dict with fields matching those expected
        by TaskDesc.
        """
        if not isinstance(obj, dict):
            return False
        required_params = set(name for name, field in cls.__fields__.items()
                              if field.required)
        all_params = set(cls.__fields__)
        return required_params <= set(obj.keys()) <= all_params

    @classmethod
    def taskdesk_like(cls, obj):
        """
        Return `True` if it is worth trying to load `obj` with `TaskDesc.load`.
        The main purpose of this is to make error messages less confusing, by
        not reporting "TaskDesc.load expects its argument to be ..." 
        unnecessarily. We want to err on the side of returning `True` rather
        than `False`: Incorrectly returning `True` just means a more
        verbose error, if nothing loads correctly. Incorrectly returning `False`
        means that something that should work suddenly does not work at all.

        Used in Task.validate()
        """
        return ((isinstance(obj, dict) and cls.looks_compatible(obj))
                or (isinstance(obj, str) and (".json" in obj
                                              or ".taskdesc" in obj))
                )

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
            elif len(obj) > 255:
                # Filenames on ext4 can have at most 255 characters.
                # On Windows it may be even less.
                obj_type = 'JSON'
            elif '{' not in obj:
                # A JSON object has at least one pair of brackets
                obj_type = 'path'
            else:
                obj_type = 'likely JSON'

            if obj_type == 'JSON':
                data = json.loads(obj)
                # taskdesc = cls.parse_raw(obj)
            elif obj_type == 'likely JSON':
                try:
                    data = json.loads(obj)
                    # taskdesc = cls.parse_raw(obj)
                except ValidationError:
                    with open(obj) as f:
                        data = json.load(f)
                    # taskdesc = cls.parse_file(obj)
            else:
                with open(obj) as f:
                    data = json.load(f)
                # taskdesc = cls.parse_file(obj)

        elif isinstance(obj, Path):
            with open(obj) as f:
                data = json.load(f)
            # taskdesc = cls.parse_file(obj)

        elif isinstance(obj, io.IOBase):
            data = pydantic.parse.load_str_bytes(obj.read())
            # taskdesc = cls.parse_obj(data)

        elif isinstance(obj, dict):
            data = obj
            # taskdesc = cls.parse_obj(obj)

        else:
            raise TypeError("TaskDesc.load expects its argument to be either "
                            "a string, a path, an IO object or a dictionary. "
                            f"It received a {type(obj)}.")

        if "generator_module" in data:
            taskdesc = GeneratedTaskDesc.parse_obj(data)
        else:
            taskdesc = TaskDesc.parse_obj(data)

        assert isinstance(taskdesc, TaskDesc)
        return taskdesc
        
    def save(self, path, allow_overwrite, **json_kwargs) -> Path:
        """
        Save the task description. The resulting file can be run from the
        command line with ``smttask run [taskdesc]``.
        Extra keyword arguments are passed on to the JSON exporter.
        For example, if the exported description should human-readable,
        `indent=2` is a useful options.
        
        Returns the path to the saved file.
        """
        if os.path.isdir(path):
            fname = f"{self.taskname}__{self.inputs.digest}"
            dirpath = Path(path)
        else:
            path = Path(path)
            dirpath = path.parent
            fname = path.name
        suffix = '.' + iotools.defined_formats['taskdesc'].ext.strip('.')
        # NB: don't use `with_suffix`, because `fname` may contain a period
        outpath = Path(f"{dirpath/fname}{suffix}")
        if outpath.exists():
            if outpath.is_dir():
                raise IsADirectoryError(f"Location {outpath} points to a directory.")
            elif allow_overwrite:
                logger.info(f"Overwriting file at location {outpath}.")
            else:
                raise FileExistsError(f"A file already exists at location {outpath}. ")
        with open(outpath, 'w') as f:
            f.write(self.json(**json_kwargs))
        return outpath

class GeneratedTaskDesc(TaskDesc):
    generator_module  : str
    generator_function: str
    generator_args    : tuple
    generator_kwargs  : dict

# ============================
# Register the taskdesc type with iotools
# ============================
# import mackelab_toolbox.iotools as io
ioformat = iotools.Format('taskdesc.json',
                          save=lambda file,task: task.save(file),
                          load=Task.from_desc,
                          bytes=False)
iotools.defined_formats['taskdesc'] = ioformat
iotools.register_datatype(Task, format='taskdesc')
