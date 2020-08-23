import inspect
import abc
import typing
from typing import Union
from pydantic.main import ModelMetaclass
from . import base
from . import smttask
from .utils import lenient_issubclass

__ALL__ = ["RecordedTask", "InMemoryTask"]

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
        raise(f"Function {f.__name__} is defined in the '__main__' script. "
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

def RecordedTask(arg0=None, *, cache=False):
    """
    The default value for the 'cache' attribute may optionally be specified
    as an argument to the decorator.
    """
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, smttask.RecordedTask)
            task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, smttask.RecordedTask)


def InMemoryTask(arg0=None, *, cache=False):
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, smttask.InMemoryTask)
            task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, smttask.InMemoryTask)

def UnpureMemoizedTask(arg0=None, *, cache=False):
    if arg0 is None:
        def decorator(f):
            task = _make_task(f, smttask.UnpureMemoizedTask)
            task.cache = cache
            return task
        return decorator
    else:
        return _make_task(arg0, smttask.UnpureMemoizedTask)
