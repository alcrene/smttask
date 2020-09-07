import inspect
import abc
import typing
from typing import Union, Dict
from numbers import Integral
from pydantic.main import ModelMetaclass
from . import base
from . import task_types
from .utils import lenient_issubclass
from .typing import PureFunction

__all__ = ["RecordedTask", "RecordedIterativeTask",
           "MemoizedTask", "NonMemoizedTask", "UnpureMemoizedTask",
           "Partial"]

def _make_input_class(f, json_encoders=None):
    defaults = {}
    annotations = {}
    # Override the lenience of the base TaskInput class and only allow expected arguments
    json_encoders_arg = json_encoders
    class Config:
        extra = 'forbid'
        if json_encoders_arg:
            json_encoders = {**base.TaskInput.Config.json_encoders, **json_encoders_arg}
    for nm, param in inspect.signature(f).parameters.items():
        if param.annotation is inspect._empty:
            raise TypeError(
                "Constructing a Task requires that all function arguments "
                f"be annotated. Offender: argument '{nm}' of '{f.__qualname__}'.")
        annotations[nm] = Union[base.Task, param.annotation]
        if param.default is not inspect._empty:
            defaults[nm] = param.default
    Inputs = ModelMetaclass(f"{f.__qualname__}.Inputs", (base.TaskInput,),
                            {**defaults,
                             'Config': Config,
                             '__annotations__': annotations}
                            )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Inputs.__module__ = f.__module__
    # update_forward_refs required for 3.9+ style annotations
    Inputs.update_forward_refs()
    return Inputs

def _make_output_class(f, json_encoders=None):
    return_annot = typing.get_type_hints(f).get('return', inspect._empty)
    if return_annot is inspect._empty:
        raise TypeError(
            f"Unable to construct a Task from function '{f.__qualname__}': "
            "the annotation for the return value is missing. "
            "This may be a type, or a subclass of TaskOutput.")
    json_encoders_arg = json_encoders
    class Config:
        if json_encoders_arg:
            json_encoders = {**base.TaskOutput.Config.json_encoders, **json_encoders_arg}
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
    if not Inputs:
        Inputs = _make_input_class(f, json_encoders)
    if not Outputs:
        Outputs = _make_output_class(f, json_encoders)
    if f.__module__ == "__main__":
        raise RuntimeError(
            f"Function {f.__qualname__} is defined in the '__main__' script. "
            "It needs to be in a separate module, and imported into the "
            "main script.")
    Task = abc.ABCMeta(f.__qualname__, (task_type,),
                       {'Inputs': Inputs,
                        'Outputs': Outputs,
                        '_run': staticmethod(f),
                        '_module_name': f.__module__}
                       )
    # Set correct module; workaround for https://bugs.python.org/issue28869
    Task.__module__ = f.__module__
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
        task.Inputs._unhashed_params = [iteration_parameter]
        if cache is not None:
            task.cache = cache
        return task
    return decorator

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


########
# Partial decorator

from functools import partial, wraps
from typing import Callable
from inspect import Signature, Parameter
import itertools

def Partial(kwargs=None, *, cache=True, json_encoders=None, **kwargs0):
    """Creates a Task which binds function arguments to the provided values.

    Implemented as a wrapper around `functools.partial`: converts the call
    into a Task, allowing the partial function to be used anywhere in a
    workflow. In particular, the bound argument values can be task inputs.

    Intended use-case: when a partial function should be passed as argument to
    a task; i.e. a function for which some of the arguments are bound to
    pre-determined values.

    Values for the bound parameters are typically passed as keyword arguments.
    However, they can also be passed as a dictionary (required if arguments
    class with the argument names of `Partial` itself).

    Parameters
    ----------
    **kwargs:
        Any unrecognized keyword argument is passed as an argument to
        `functools.partial` to bind the decorated function's parameter.
        Combined with `kwargs` before passing the `partial` (with precedence
        to keyword parameters).
    kwargs: Optional, dict
        Merged with the keyword arguments.
    cache: bool
        Whether to memoize the result. The default is to do so (True).
    json_encoders: dict
        Use this argument to specify additional JSON encoders for the Pydantic
        object containing the inputs. The encoders in `smttask.typing.json_encoders`
        are included by default.

    Returns
    -------
    Decorator: (Callable) -> Task[->Callable]
       When applied to a function ``f``, returns an *instantiated* `Task`.
       This task has as input the wrapped function, and the keyword arguments
       to `Partial`. When run, it returns a new function, created by calling
       ``functools.partial(f, **kwargs)``.
    """
    # Normalize arguments / combine with defaults
    if kwargs is None:
        kwargs = {}
    kwargs = {**kwargs, **kwargs0}
    # `Partial` is called with arguments, so we return a decorator
    def decorator(f):
        func_name, PartialTask = _get_partial_task_type(
            f, kwargs, cache=cache, json_encoders=json_encoders)
        return PartialTask(**{func_name: f}, **kwargs)
    return decorator

_partial_task_types = {}
def _get_partial_task_type(f, kwargs, cache, json_encoders):
    """
    Create a Task type which binds values to the arguments of `f`.
    A cache is maintained so that multiple calls with the same `f` and
    `kwarg` keys return the same type object.
    (The values of `kwargs` are only used to provide annotation defaults,
    if the wrapped function does not annotate its arguments.)
    """
    key = (f, tuple(sorted(kwargs)))
    if key in _partial_task_types:
        func_name, Task = _partial_task_types[key]
        if (cache != Task.cache
            or not set(json_encoders) <= set(Task.Inputs.__config__.json_encoders)):
            raise RuntimeError(f"A Partial task with function {f} was already ",
                               "created, but it differs either in its `cache` "
                               "or `json_encoders` arguments.")
    else:
        # Find a name for the function which isn't already taken by the kwargs
        name_gen = itertools.chain(["func"], (f"func{n}" for n in range(20)))
        func_name = None
        for name in name_gen:
            if name not in kwargs:
                func_name = name
                break
        if func_name is None:
            raise RuntimeError("Unable to find a unique function name. This might "
                               "happen if you nest an inordinate number of "
                               "`tasks.Partial` calls.")
        ## Create the Input object
        # The task itself takes as arguments the original function, and the
        # arguments we want to bind. The remaining function arguments will be
        # arguments to the returned partial function.
        # >> Begin copy (with modifications) from decorators._make_input_class
        annotations = {func_name: PureFunction}
        json_encoders_arg = json_encoders
        class Config:
            extra = 'forbid'
            if json_encoders_arg:
                json_encoders = {**base.TaskInput.Config.json_encoders,
                                 **json_encoders_arg}
        orig_sig = inspect.signature(f)
        for nm, value in kwargs.items():
            param = orig_sig.parameters[nm]
            if param.annotation is inspect._empty:
                annotations[nm] = type(value)
            else:
                annotations[nm] = Union[base.Task, param.annotation]
        Inputs = ModelMetaclass(f"{f.__qualname__}.Inputs", (base.TaskInput,),
                                {
                                 'Config': Config,
                                 '__annotations__': annotations}
                                )
        # Set correct module; workaround for https://bugs.python.org/issue28869
        Inputs.__module__ = f.__module__
        # update_forward_refs required for 3.9+ style annotations
        Inputs.update_forward_refs()
        # << End copy
        ## Create the Output object
        # >> Begin copy (with modifications) from _make_output_class
        return_annot = typing.get_type_hints(f).get('return', inspect._empty)
        if return_annot is inspect._empty:
            raise TypeError(
                f"Unable to construct a Task from function '{f.__qualname__}': "
                "the annotation for the return value is missing. "
                "This may be a type, or a subclass of TaskOutput.")
        if lenient_issubclass(return_annot, base.TaskOutput):
            raise NotImplementedError("Having an explicit `TaskOutput` return "
                                      "type for an argument to `Partial` is "
                                      "unsupported at thist time.")
        task_return_annot = Callable[..., return_annot]
        Outputs = ModelMetaclass(f"{f.__qualname__}.Outputs", (base.TaskOutput,),
                                 {'__annotations__': {"": task_return_annot}})
        # << End copy
        ## Create the Task.
        @wraps(f)  # Copy __name__, __qualname__, __module__, etc
        def Bind(**kwargs_):
            f_ = kwargs_.pop(func_name)
            assert f is f_
            bound_f = partial(f_, **kwargs_)
            # Without explicitly assigning to __signature__, `functools.update_wrapper` would use
            # the signature of `f`, and so still include the arguments we bound
            bound_f.__dict__['__signature__'] = inspect.signature(bound_f)
            return bound_f
        task_sig = Signature(
            [Parameter(func_name, Parameter.KEYWORD_ONLY)]
            + [Parameter(nm, Parameter.KEYWORD_ONLY) for nm in kwargs])
        # Bind.__dict__['__signature__'] = task_sig
        Bind.__module__ = Partial.__module__
        Task = _make_task(Bind, task_types.MemoizedTask, Inputs=Inputs, Outputs=Outputs)
        _partial_task_types[key] = func_name, Task

    return func_name, Task
