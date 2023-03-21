"""
Task generators

Functions which generate tasks.

.. Remark:: Constructor arguments are deserialized by inspecting their
   signature, so all arguments must be typed.
"""
from typing import Optional, Union, Dict, Tuple, Callable, List
from collections.abc import Collection
from scityping import Type
from scityping.functions import PureFunction
from .base import Task, GeneratedTask, TaskInput, TaskOutput
from .task_types import MemoizedTask, RecordedTask
from .typing import json_encoders as smttask_json_encoders

__all__ = ["Create", "Join"]

# DEVNOTE: All automatically generated tasks should inherit from *GeneratedTask*,
#   to ensure that they serialize correctly. GeneratedTask requires that these
#   additional attributes be set by the creator function:
#     - generator_function: Callable
#     - args  : Tuple
#     - kwargs: Dict
#     `_module_name` may also optionally be set
# Task generators are used as follows::
#    TaskGenerator(generator args)(task args)

class GeneratedMemoizedTask(GeneratedTask, MemoizedTask):
    pass
class GeneratedRecordedTask(GeneratedTask, RecordedTask):
    pass

## Create Task ##
    
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
        def CreateMyType(arg1: <type1>, arg2: <type2>, ...) -> MyNonSerializableType:
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
    # NB: We need to ensure that different Create Tasks for different `cls`
    #     have different names, since base.created_task_types uses
    #     `(module name, task name)` pairs to avoid recreating task types.
    #     Thus simply setting __name__ after the task was created is insufficient
    #     Instead, we declare the contents first (*without* subclassing Task),
    #     and then create the Task with the `cls`-dependent name. (We could
    #     also do the whole thing dynamically in `type()` and avoid the
    #     multiple inheritance, but this seems easier to read.)
    class CreatorTaskContent:
        generator_function: Callable = Create  # Used to serialize the generator
        generator_args    : Tuple[Type] = (cls,)
        generator_kwargs  : dict = {}
        _module_name      : str=cls.__module__
        class Inputs(TaskInput):
            obj_to_create: Type[cls]
            kwargs: dict
            class Config:  # Re-add json encoders to reflect dynamic changes to json_encoders
                json_encoders = {**json_encoders_arg,
                                 **smttask_json_encoders,
                                 **TaskInput.Config.json_encoders}
                
        class Outputs(TaskOutput):
            obj: cls
            class Config:  # Re-add json encoders to reflect dynamic changes to json_encoders
                json_encoders = {**json_encoders_arg,
                                 **smttask_json_encoders,
                                 **TaskOutput.Config.json_encoders}
        @staticmethod
        def _run(obj_to_create: Type[cls], kwargs: dict) -> cls:
            return obj_to_create(**kwargs)
    CreatorTask = type(f"Create{cls.__name__}",
                       (CreatorTaskContent, GeneratedMemoizedTask),
                       {})
    # If we don't set __name__, we get a name like Create.<locals>.CreatorTask
    # CreatorTask.__name__ = f"Create{cls.__name__}"
    CreatorTask.__qualname__ = f"Create.Create{cls.__name__}"

    def creator(reason=None, __cls__=cls, **kwargs):
        _cls = kwargs.pop('obj_to_create', __cls__)  # Must accept the same signature as the Task itself; _cls is thrown away
        # TODO? Parse _cls e.g. with __cls__.validate and check that it is the same as __cls__ ?
        if 'kwargs' in kwargs:
            assert len(kwargs) == 1, f"Arguments to {CreatorTask.taskname()} should either all be passed as keywords, or as a dictionary with the `kwargs` argument."
            kwargs = kwargs['kwargs']
        return CreatorTask(obj_to_create=__cls__, kwargs=kwargs, reason=reason)

    return creator

## Join Task ##
# TODO: Test

# Keep a cache of already created Join Task types.
# This ensures that if a Join is created with the same arguments, the same
# task instance is returned.
join_cache = {}

# NB: The reason we annotate with type `List[Task]` is because when the Join
#     task is serialized, this is always the form used. (See `generator_args` below)
def Join(*tasks, reason: Optional[str]=None):
    """
    Analogous to a 'join' operation in multiprocessing: Given a list of tasks,
    returns a new Task, which when executed runs all tasks in the list.
    The result is currently just a list of the returned values of each task,
    in the order in which they were specified.
    
    All tasks must be instances of the same Task subclass.
    Tasks may be passed as separate arguments, or wrapped in a list.
    
    .. Note:: The return type may be enriched in the future. Mostly we are
       waiting for good motivating use cases before adding functionality.
       
    .. Note:: If tasks or some of their dependencies are also recorded, the
       recordstore will contain duplicate entries.
       
    .. Todo:: Allow Join to alternatively return a MemoizedTask.
    
    .. Todo:: Create a `SavedTask` type that can be used to cache on disk
       computation results without being recorded. Note that one has to take
       care that computations remain reproducible, probably by invalidating the
       cache whenever the project's git hash changes.
    
    Returns
    -------
    RecordedTask
    """
    task_types = set(type(t) for t in tasks)
    # # NB: If tasks are defined in a workflow, which is run multiple times, they will be
    # #     different classes (we might want to change this, but that's the current state)
    # #     To get around this, we check only if the full task name (project.module.my.class.name)
    # #     is shared by all
    # full_task_names = set(f"{T.__module__}.{T.__qualname__}" for T in task_types)
    if len(task_types) > 1:
        raise ValueError("All arguments to Join must be the same Task type.\n"
                         f"Received: {task_types}.")
    task_type = next(iter(task_types))
    if issubclass(task_type, Collection) and len(tasks) == 1:
        # We most likely received tasks wrapped in a list; unwrap the list
        return Join(*tasks[0], reason=reason)  # EARLY EXIT
    elif not issubclass(task_type, Task):
        raise TypeError("Join only accepts Task arguments.")
    return JoinCreator(task_type)(tasks=tasks, reason=reason)

def JoinCreator(task_type: Type):  # Deserialization currently doesn't when type is specified; see smttask.typing.Type
    """
    Low-level function for creating `Join` tasks.
    For most usages, the `Join` function should be preferred.
    """
    if task_type in join_cache:
        Join = join_cache[task_type]
    else:
        class Join(GeneratedRecordedTask):
            generator_function: Callable = JoinCreator  # Used to serialize the generator
            generator_args    : tuple = (task_type,)
            generator_kwargs  : dict = {}
            _module_name      : str=task_type.__module__
            class Inputs(TaskInput):
                tasks: List[Union[task_type, task_type.Outputs.result_type]]
                class Config:
                    json_encoders = task_type.Inputs.__config__.json_encoders
                # Override `load` to show progress bar
                def load(self, progbar=2):
                    return super().load(progbar)
                    
            class Outputs(TaskOutput):
                results: List[task_type.Outputs.result_type]
                class Config:
                    json_encoders = task_type.Outputs.__config__.json_encoders

            @staticmethod
            def _run(tasks):
                # NB: Running this task will recursively run all inputs tasks.
                #     All we need to do is replace the TaskOutputs by their result;
                #     If a task has only one result value, `result` removes the
                #     outer dimension. Otherwise, values are wrapped in a namedtuple.
                # return [task.result for task in tasks]
                return tasks
        # If we don't set __name__, we get a name like Join.<locals>.Join
        Join.__name__ = f"Join[{task_type.__name__}]"
        Join.__qualname__ = f"JoinCreator.Join[{task_type.__name__}]"
        join_cache[task_type] = Join
        
    def creator(reason=None, **kwargs):
        return Join(reason=reason, **kwargs)
        
    return creator
