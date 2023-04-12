import inspect
import abc
import sys
import typing
import textwrap
from typing import ForwardRef, Union, Dict
from numbers import Integral
from pydantic.typing import evaluate_forwardref
from scityping.pydantic import ModelMetaclass
from . import base
from . import task_types
from .config import config
from .typing import json_encoders as smttask_json_encoders
from ._utils import lenient_issubclass

__all__ = ["RecordedTask", "RecordedIterativeTask",
           "MemoizedTask", "NonMemoizedTask", "UnpureMemoizedTask"
           ]

def _make_input_class(f, json_encoders=None):
    defaults = {}
    annotations = {}
    json_encoders_arg = json_encoders if json_encoders else {}
    class Config:
        # Override the lenience of the base TaskInput class and only allow expected arguments
        extra = 'forbid'
        json_encoders = {**json_encoders_arg,
                         **smttask_json_encoders,
                         **base.TaskInput.Config.json_encoders}
    for nm, param in inspect.signature(f).parameters.items():
        if nm == "self":  # When wrapping a callable class, don't include `self`
            continue
        if param.annotation is inspect._empty:
            raise TypeError(
                "Constructing a Task requires that all function arguments "
                f"be annotated. Offender: argument '{nm}' of '{f.__qualname__}'.")
        annotation = param.annotation
        if isinstance(annotation, str):
            # HACK to resolve forward refs
            globalns = sys.modules[f.__module__].__dict__.copy()
            annotation = evaluate_forwardref(ForwardRef(annotation), globalns=globalns, localns=None)
        annotations[nm] = Union[base.Task, annotation]
        if param.default is not inspect._empty:
            defaults[nm] = param.default
    # Infer the task name from the function name.
    # If f is the __call__ method of a class, use the class name.
    task_name = f.__qualname__
    if task_name.endswith(".__call__"):
        task_name = task_name.rsplit(".", 1)[0]
    Inputs = ModelMetaclass(f"{task_name}.Inputs", (base.TaskInput,),
                            {**defaults,
                             'Config': Config,
                             '__annotations__': annotations}
                            )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Inputs.__module__ = f.__module__
    # update_forward_refs required for 3.9+ style annotations
    # DEVNOTE: When one inspects Inputs.__fields__, some types may still contain 'ForwardRef'
    #          I don't know why that is, but checking Inputs.__fields__[field name].__args__[ForwardRef index].__forward_evaluated__ should still be True
    #          and [...].__forward_value__ should be the expected type
    Inputs.update_forward_refs()
    return Inputs

def _make_output_class(f, json_encoders=None):
    return_annot = typing.get_type_hints(f).get('return', inspect._empty)
    if return_annot is inspect._empty:
        raise TypeError(
            f"Unable to construct a Task from function '{f.__qualname__}': "
            "the annotation for the return value is missing. "
            "This may be specified using a type, or a subclass of TaskOutput.")
    json_encoders_arg = json_encoders if json_encoders else {}
    class Config:  # NB: MIGHT NOT be used if `f` includes a 'return' annotation
        json_encoders = {**json_encoders_arg,
                         **smttask_json_encoders,
                         **base.TaskOutput.Config.json_encoders}
    if lenient_issubclass(return_annot, base.TaskOutput):
        # Nothing to do
        Outputs = return_annot
        # Add the json_encoders to the Output type, but only if they were not
        # given explicitely in the Output type.
        if json_encoders:
            if hasattr(Outputs, 'Config') and not hasattr(Outputs.Config, 'json_encoders'):
                Outputs.Config.json_encoders = Config.json_encoders
            elif not hasattr(Outputs, 'Config'):
                Outputs.Config = Config

    else:
        assert isinstance(return_annot, (type, typing._GenericAlias))
        # A bare annotation does not define a variable name; we set it to the
        # empty string (i.e., the variable is only identified by the task name)
        Outputs = ModelMetaclass(f"{f.__qualname__}.Outputs", (base.TaskOutput,),
                                 {'__annotations__': {"": return_annot},
                                  'Config': Config}
                                 )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Outputs.__module__ = f.__module__
    # update_forward_refs required for 3.9+ style annotations
    Outputs.update_forward_refs()
    return Outputs

def _make_task(f, task_type, json_encoders=None, Inputs=None, Outputs=None):
    """
    Generate a task by inspecting the type annotations of the function `f`.
    `f` may be either a normal Python function or a callable class (i.e. one
    which defined `__call__`); lambdas or methods are not supported.

    The generated task has the following properties:

    Task name
        Set to `f.__qualname__`.
    Inheritance
        The generated task inherits from `task_type`.
    Docstring
        Starts with a schematic document task inputs and outputs, appended with
        the docstring of `f`.
    `_run`
        Set to `f`.
    """
    if isinstance(f, type):
        return _make_task_from_class(f, task_type, json_encoders, Inputs, Outputs)

    if not Inputs:
        Inputs = _make_input_class(f, json_encoders)
    if not Outputs:
        Outputs = _make_output_class(f, json_encoders)
    if f.__module__ == "__main__" and config.record:
        raise RuntimeError(
            f"Function {f.__qualname__} is defined in the '__main__' script. "
            "It needs to be in a separate module, and imported into the "
            "main script.\nException: to facilitate testing, defining tasks in "
            "the __main__ script is allowed when recording is disabled.")
    Task = abc.ABCMeta(f.__qualname__, (task_type,),
                       {'Inputs': Inputs,
                        'Outputs': Outputs,
                        '_run': staticmethod(f),
                        '_module_name': f.__module__}
                       )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Task.__module__ = f.__module__
    # Use the decorated function’s docstring, and prepend the input->output schematic
    doc = "_" + Task.schematic()  # Prepended with '_', to avoid the initial whitespace being removed (`dedent` is automatically applied to docstrings)
    if f.__doc__:
        doc += "\n\n" + textwrap.dedent(f.__doc__)
    Task.__doc__ = doc
    return Task

def _make_task_from_class(cls, task_type, json_encoders=None, Inputs=None, Outputs=None):
    """
    Generate a task by inspecting the type annotations of a *callable class*.
    (Class must have a `__call__` method.)

    The generated task has the following properties:

    Task name
        Set to `cls.__qualname__`.
    Inheritance
        The generated task inherits from both `task_type` and `cls`.
    Docstring
        Starts with a schematic document task inputs and outputs, appended with
        either the docstring `cls.__call__` or `cls` (the first non-empty one).
    `_run`
        Set to `cls.__call__`.
    """
    if not Inputs:
        Inputs = _make_input_class(cls.__call__, json_encoders)
    if not Outputs:
        Outputs = _make_output_class(cls.__call__, json_encoders)
    if cls.__module__ == "__main__" and config.record:
        raise RuntimeError(
            f"Class {cls.__qualname__} is defined in the '__main__' script. "
            "It needs to be in a separate module, and imported into the "
            "main script.\nException: to facilitate testing, defining tasks in "
            "the __main__ script is allowed when recording is disabled.")
    Task = abc.ABCMeta(cls.__qualname__, (task_type, cls),
                       {'Inputs': Inputs,
                        'Outputs': Outputs,
                        '_run': cls.__call__,
                        '_module_name': cls.__module__}
                       )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Task.__module__ = cls.__module__
    # Use the decorated function’s docstring, and prepend the input->output schematic
    doc = "_" + Task.schematic()  # Prepended with '_', to avoid the initial whitespace being removed (`dedent` is automatically applied to docstrings)
    fdoc = cls.__call__.__doc__ or cls.__doc__
    if fdoc:
        doc += "\n\n" + textwrap.dedent(fdoc)
    Task.__doc__ = doc
    return Task

def RecordedTask(arg0=None, *, cache=None, json_encoders=None):
    """
    The default value for the 'cache' attribute may optionally be specified
    as an argument to the decorator.
    """
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, task_types.RecordedTask, json_encoders)
            if cache is not None:
                task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, task_types.RecordedTask, json_encoders)
RecordedTask.__doc__ = f"{task_types.RecordedTask.__doc__}\n{RecordedTask.__doc__}"

def RecordedIterativeTask(iteration_parameter=None, *, map: Dict[str,str]=None,
                          cache=None, json_encoders=None):
    """
    In contrast to other decorators, `RecordedIterativeTask` cannot be used
    without arguments.

    Parameters
    ----------
    map: dict (required)
        Key:value pairs correspond to [output var name]:[input var name].
        They describe how modify the input arguments, given the outputs from
        a previous run, such the final state of that run can be recreated
        and iterations continued from that point.
    """
    if iteration_parameter is None or map is None:
        raise TypeError(
            "In contrast to other Task decorators, `RecordedIterativeTask` "
            "cannot be used without arguments. You must specify an "
            "iteration parameter and how output parameters from previous "
            "iterations are mapped to inputs.")
    def decorator(f):
        task = _make_task(f, task_types.RecordedIterativeTask, json_encoders)
        in_fields = set(task.Inputs.__fields__) - set(base.TaskInput.__fields__)
        out_fields = set(task.Outputs.__fields__) - set(base.TaskOutput.__fields__)
        if len(map) == 0:
            raise ValueError(f"The task {task.taskname()} does not define how "
                             "previous iterations are mapped to new ones: its "
                             "`map` argument is empty.")
        elif not set(map.keys()) <= set(out_fields):
            raise ValueError("The keys of the iteration map of task "
                             f"{task.taskname()} do not all correspond to "
                             f"output variables.\nMap keys: {sorted(map.keys())}\n"
                             f"Output variables: {sorted(out_fields)}")
        elif not set(map.values()) <= set(in_fields):
            raise ValueError("The values of the iteration map of task "
                             f"{task.taskname()} do not all correspond to "
                             f"input variables.\nMap keys: {sorted(map.values())}\n"
                             f"Input variables: {sorted(in_fields)}")
        iterp_type = task.Outputs.__fields__[iteration_parameter].type_
        if not isinstance(iterp_type, type) or not issubclass(iterp_type, Integral):
            raise TypeError(f"Task '{task.taskname()}': The iteration parameter "
                            f"'{iteration_parameter}' does not have integer type.")
        task._iteration_parameter = iteration_parameter
        task._iteration_map = map
        task.Inputs._unhashed_params = task.Inputs._unhashed_params.union([iteration_parameter])  # Returns a copy because _unhased_params is a frozenset
        if cache is not None:
            task.cache = cache
        return task
    return decorator
RecordedIterativeTask.__doc__ = f"{task_types.RecordedIterativeTask.__doc__}\n{RecordedIterativeTask.__doc__}"

def MemoizedTask(arg0=None, *, cache=True, json_encoders=None):
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, task_types.MemoizedTask, json_encoders)
            if cache is not None:
                task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, task_types.MemoizedTask, json_encoders)
MemoizedTask.__doc__ = task_types.MemoizedTask.__doc__

def NonMemoizedTask(arg0=None, *, cache=False, json_encoders=None):
    """Same as `MemoizedTask`, but defaults to not memoizing the result."""
    return MemoizedTask(arg0, cache=cache, json_encoders=json_encoders)

def UnpureMemoizedTask(arg0=None, *, cache=None, json_encoders=None):
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, task_types.UnpureMemoizedTask, json_encoders)
            if cache is not None:
                task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, task_types.UnpureMemoizedTask, json_encoders)
UnpureMemoizedTask.__doc__ = task_types.UnpureMemoizedTask.__doc__
