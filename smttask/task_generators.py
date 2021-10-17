"""
Task generators

Functions which generate tasks.

.. Remark:: Constructor arguments are deserialized by inspecting their
   signature, so all arguments must be typed.
"""
from typing import Optional, Dict
from .typing import (Type, Callable, PureFunction,
                     json_encoders as smttask_json_encoders)
from .base import GeneratedTask, TaskInput, TaskOutput
from .task_types import MemoizedTask

__all__ = ["Create"]

class GeneratedMemoizedTask(GeneratedTask, MemoizedTask):
    # Reminder – these additional attributes must be set by the creator function:
    # - generator_function: Callable
    # - args  : Tuple
    # - kwargs: Dict
    pass

def Create(cls: Type, json_encoders: Optional[Dict[Type,PureFunction]]=None):
    """
    Syntactic sugar for creating Tasks which simply bind class arguments.
    Doing this with a `MemoizedTask` requires a few lines of
    boilerplate and repeating all arguments in the Task signature.
    
    Usage (main file)::
    
        from smttask import CreatorTask
        from somewhere import MyNonSerializableType
        obj = Create(MyNonSerializableType)(arg1=<value1>, arg2=<value2>, ...)
    
    In comparison, using a `MemoizedTask` is more verbose and requires a second file:
    
    In _tasks.py_::
    
        from smttask import MemoizedTask
        from somewhere import MyNonSerializableType
        @MemoizedTask
        def CreateMyType(arg1: <type1>, arg2: <type2>, ...) -> MyNonSerializableType:
            return MyNonSerializableType(arg1=arg1, arg2=arg2, ...)
            
    In main file::
    
        from tasks import CreateMyType
        obj = CreateMyType(arg1=<value1>, arg2=<value2>, ...)
    
    Note that in order to be serialized, `Create` still requires that the type
    it wraps (`MyNonSerializableType` in the example above) be defined in its
    own module. The type is searched for by name, so certain shenanigans with
    dynamic types may break deserialization (the break cases are mostly the
    same as for `pickle`).
    
    Another advantage compared to using a plain `MemoizedTask` is that a
    `CreatorTask` supports variadic keyword arguments without requiring to
    wrap them in a dictionary.
    """
    json_encoders_arg = json_encoders if json_encoders else {}
    class CreatorTask(GeneratedMemoizedTask):
        generator_function: Callable = Create  # Used to serialize the generator
        generator_args    : tuple = (cls,)
        generator_kwargs  : dict = {}
        _module_name      : cls.__module__
        class Inputs(TaskInput):
            obj_to_create: Type[cls]
            kwargs: dict
            class Config:  # Re-add json encoders to reflect dynamic changes to json_encoders
                json_encoders = {**TaskInput.Config.json_encoders,
                                 **smttask_json_encoders,
                                 **json_encoders_arg}
                
        class Outputs(TaskOutput):
            obj: cls
            class Config:  # Re-add json encoders to reflect dynamic changes to json_encoders
                json_encoders = {**TaskOutput.Config.json_encoders,
                                 **smttask_json_encoders,
                                 **json_encoders_arg}
        @staticmethod
        def _run(obj_to_create: Type[cls], kwargs: dict) -> cls:
            return obj_to_create(**kwargs)
    CreatorTask.__name__ = f"Create{cls.__name__}"
    CreatorTask.__qualname__ = f"Create.Create{cls.__name__}"

    def creator(reason=None, __cls__=cls, **kwargs):
        _cls = kwargs.pop('obj_to_create', __cls__)  # Must accept the same signature as the Task itself; _cls is thrown away
        # TODO? Parse _cls e.g. with __cls__.validate and check that it is the same as __cls__ ?
        if 'kwargs' in kwargs:
            assert len(kwargs) == 1, f"Arguments to {CreatorTask.taskname()} should either all be passed as keywords, or as a dictionary with the `kwargs` argument."
            kwargs = kwargs['kwargs']
        return CreatorTask(obj_to_create=__cls__, kwargs=kwargs, reason=reason)

    return creator
    
