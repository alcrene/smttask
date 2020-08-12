import inspect
import abc
from typing import Union
from pydantic.main import ModelMetaclass
from . import base
from . import smttask
from .utils import lenient_issubclass

__ALL__ = ["RecordedTask", "InMemoryTask"]

def _make_input_class(f):
    defaults = {}
    annotations = {}
    # Override the lenience of the base TaskInputs class and only allow expected arguments
    class Config:
        extra = 'forbid'
    for nm, param in inspect.signature(f).parameters.items():
        if param.annotation is inspect._empty:
            raise TypeError(
                "Constructing a Task requires that all function arguments "
                f"be annotated. Offender: argument '{nm}' of '{f.__name__}'.")
        annotations[nm] = Union[base.Task, param.annotation]
        if param.default is not inspect._empty:
            namespace[nm] = param.default
    Inputs = ModelMetaclass("Inputs", (base.TaskInputs,),
                            {**defaults,
                             'Config': Config,
                             '__annotations__': annotations}
                            )
    return Inputs

def _make_output_class(f):
    return_annot = f.__annotations__.get('return', inspect._empty)
    if return_annot is inspect._empty:
        raise TypeError(
            f"Unable to construct a Task from function '{f.__name__}': "
            "the annotation for the return value is missing. "
            "This may be a type, or a subclass of TaskOutput.")
    if lenient_issubclass(return_annot, base.TaskOutputs):
        # Nothing to do
        Outputs = return_annot
    else:
        assert isinstance(return_annot, type)
        # A bare annotation does not define a variable name; we set it to the
        # empty string (i.e., the variable is only identified by the task name)
        Outputs = ModelMetaclass("Outputs", (base.TaskOutputs,),
                                 {'__annotations__': {"": return_annot}
                                  }
                                 )
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
    return Task

def RecordedTask(f):
    return _make_task(f, smttask.RecordedTask)

def InMemoryTask(f):
    return _make_task(f, smttask.InMemoryTask)
