import inspect
import abc
import typing
from typing import Union, Dict
from numbers import Integral
from pydantic.main import ModelMetaclass
from . import base
from . import smttask
from .utils import lenient_issubclass

__ALL__ = ["RecordedTask", "MemoizedTask"]

def _make_input_class(f):
    defaults = {}
    annotations = {}
    # Override the lenience of the base TaskInput class and only allow expected arguments
    class Config:
        extra = 'forbid'
    for nm, param in inspect.signature(f).parameters.items():
        if param.annotation is inspect._empty:
            raise TypeError(
                "Constructing a Task requires that all function arguments "
                f"be annotated. Offender: argument '{nm}' of '{f.__name__}'.")
        annotations[nm] = Union[base.Task, param.annotation]
        if param.default is not inspect._empty:
            defaults[nm] = param.default
    Inputs = ModelMetaclass(f"{f.__name__}.Inputs", (base.TaskInput,),
                            {**defaults,
                             'Config': Config,
                             '__annotations__': annotations}
                            )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Inputs.__module__ = f.__module__
    return Inputs

def _make_output_class(f):
    return_annot = typing.get_type_hints(f).get('return', inspect._empty)
    if return_annot is inspect._empty:
        raise TypeError(
            f"Unable to construct a Task from function '{f.__name__}': "
            "the annotation for the return value is missing. "
            "This may be a type, or a subclass of TaskOutput.")
    if lenient_issubclass(return_annot, base.TaskOutput):
        # Nothing to do
        Outputs = return_annot
    else:
        assert isinstance(return_annot, (type, typing._GenericAlias))
        # A bare annotation does not define a variable name; we set it to the
        # empty string (i.e., the variable is only identified by the task name)
        Outputs = ModelMetaclass(f"{f.__name__}.Outputs", (base.TaskOutput,),
                                 {'__annotations__': {"": return_annot}
                                  }
                                 )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Outputs.__module__ = f.__module__
    return Outputs

def _make_task(f, task_type):
    Inputs = _make_input_class(f)
    Outputs = _make_output_class(f)
    if f.__module__ == "__main__":
        raise RuntimeError(
            f"Function {f.__name__} is defined in the '__main__' script. "
            "It needs to be in a separate module, and imported into the "
            "main script.")
    Task = abc.ABCMeta(f.__name__, (task_type,),
                       {'Inputs': Inputs,
                        'Outputs': Outputs,
                        '_run': staticmethod(f),
                        '_module_name': f.__module__}
                       )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Task.__module__ = f.__module__
    return Task

def RecordedTask(arg0=None, *, cache=None):
    """
    The default value for the 'cache' attribute may optionally be specified
    as an argument to the decorator.
    """
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, smttask.RecordedTask)
            if cache is not None:
                task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, smttask.RecordedTask)

def RecordedIterativeTask(iteration_parameter=None, *, map: Dict[str,str]=None, cache=None):
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
        task = _make_task(f, smttask.RecordedIterativeTask)
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
        task.Inputs._unhashed_params = [iteration_parameter]
        if cache is not None:
            task.cache = cache
        return task
    return decorator

def MemoizedTask(arg0=None, *, cache=None):
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, smttask.MemoizedTask)
            if cache is not None:
                task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, smttask.MemoizedTask)

def UnpureMemoizedTask(arg0=None, *, cache=None):
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, smttask.UnpureMemoizedTask)
            if cache is not None:
                task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, smttask.UnpureMemoizedTask)
