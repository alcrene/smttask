import sys
import abc
from warnings import warn
import importlib
import inspect
import numpy as np

import typing
from typing import (Optional, TypeVar, Generic,
                    Callable, Iterable, Tuple, List)
from types import new_class
from pydantic.fields import sequence_like
import pydantic.generics

from sumatra.datastore.filesystem import DataFile

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
    return DataFile(
        os.path.relpath(Path(input.full_path).resolve(),
                        inputstore.root),
        inputstore)

# def describe_datafile(datafile: DataFile):
#     assert isinstance(datafile, DataFile)
#     filename = Path(filename.full_path)
#     return {
#         'input type': 'File',
#         'filename': str(normalize_input_path(filename))
#     }

# This class was originally adapted from pydantic.types.ConstrainedList
# To understand what is happening with the __origin__ and __args__, one needs
# to refer to pydantic.fields.ModelField._type_analysis
# __origin__ is used for two things:  - Determining which validator to use
#                                     - Determining the type to which the result is cast
# __args__ is the list of arguments in brackets given to the type. Like List, we only support one argument.
# The challenge is that Pydantic, if it recognizes SeparateOutputs or __origin__
# as a subclass of tuple, removes the subclass and returns a plain tuple.
# We work around this by creating two nested subclasses of SeparateOutputs within
# the function `separate_outputs`; the child subclass is the type of the field,
# while the grandchild subclass is an empty class that inherits from the first
# AND tuple. The `validate` function (after using some Pydantic internals to
# ensure that any Pydantic-compatible type works as an item type), then returns
# by casting to the grandchild subtype.
# It's a complicated solution and I would be happy to find a simpler one.
T = TypeVar('T')
class SeparateOutputs:
    """
    This class returns values with two properties:
    
    - They verify ``isinstance(v, tuple)`` and are equivalent to tuple.
    - They have a type distinct from tuple, so that smttask can recognize and
      treat them differently from a tuple.
    """
    # Setting __origin__ = tuple would allow Pydantic to recognize this as a tuple and
    # validate appropriately. Unfortunately it also removes the `SeparateOutputs`
    # type, which is the whole point of this class
    # __origin__ = Iterable
    __args__: Tuple[typing.Type[T]]

    # result_type: type
    item_type: typing.Type[T]

    _get_names: Callable[..., Iterable[str]]
    get_names_args: Tuple[str]

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v, values, field):
        # Loosely based on pydantic.fields.ModelField._validate_sequence_like,_validate_singleton
        if not sequence_like(v):
            raise TypeError(f"Field '{field.name}' expects a sequence. "
                            f"Received: {v}.")

        assert len(field.sub_fields) == 1
        result = []
        for v_ in v:
            r, err = field.sub_fields[0].validate(
                v_, values, loc=field.name, cls=cls)
            if err:
                raise TypeError(
                    f"Field '{field.name}' expects a sequence of '{cls.item_type.__name__}', "
                    f"but received a sequence containing {v_} (type '{type(v_)}')."
                    ) from err.exc
            result.append(r)
        return cls.result_type(result)

    @classmethod
    def get_names(cls, **kwargs):
        # HACK/FIXME: Awful hack to avoid circular imports.
        #   Assumes base.py is called before a SeparateOutputs instance is constructed.
        Task = sys.modules['smttask'].Task
        # TODO? Support Task inputs ? (Perhaps by automatically running them ?)
        # Currently we just have an 'unsupported' warning if any input is a Task.
        if any(isinstance(v, Task) for v in kwargs.values()):
            task_inputs = [k for k,v in kwargs.items() if isinstance(v, Task)]
            warn("The `separate_outputs` constructor has only been tested "
                 "with concrete, non-Task types.\n "
                 f"Task inputs: {task_inputs}")
        return cls._get_names(**kwargs)
SeparateOutputs.__origin__ = SeparateOutputs  # This is the pattern, but overriden in separate_outputs()

# This function is adapted from pydantic.types.conlist
def separate_outputs(item_type: typing.Type[T], get_names: Callable[...,List[str]]):
    """
    In terms of typing, equivalent to `Tuple[T,...]`, but indicates to smttask
    to save each element separately. This was conceived for two use cases:
    
    1. When the number of outputs is dependent on the input variables.
    2. When some or all of the outputs may be very large. For example, we
       may have a Task which allows different recorder objects to track
       quantities during a simulation.

    Parameters
    ----------
    get_names:
       Function used to determine the names under which name each
       value is saved. Takes any number of arguments, but their names must
       match the name of a task input. Returns a list of strings.
       E.g., if the associated Task defines inputs 'freq' and 'phase', then
       the `get_names` function may have any one of these signatures:

       - `get_names` () -> List[str]
       - `get_names` (freq) -> List[str]
       - `get_names` (phase) -> List[str]
       - `get_names` (freq, phase) -> List[str]

       This allows the output names to depend on any of the Task parameters.
       CAVEAT: Currently Task values are not supported, so in the example
       above, if `freq` may be provided as a Task instance, it should not be
       used in `get_names`.
    """
    sig = inspect.signature(get_names)
    namespace = {'item_type':item_type,
                 '__args__': [item_type],
                 '_get_names': get_names,
                 'get_names_args': tuple(sig.parameters)}
    base_class = new_class('SeparateOutputsValueBase', (SeparateOutputs,), {},
                           lambda ns: ns.update(namespace))
    base_class.__origin__ = base_class
    result_class = type('SeparateOutputsValue', (tuple, base_class), {})
    base_class.result_type = result_class
    return base_class


json_encoders = {
    # DataFile: lambda filename: describe_datafile(filename),
    # pydantic.main.ModelMetaclass: Type.json_encoder_pydantic_generic,
    # typing._GenericAlias        : Type.json_encoder_generic,
    # type                        : Type.json_encoder,
    set                         : lambda s: sorted(s), # Default serializer has undefined order => inconsistent task digests
    frozenset                   : lambda s: sorted(s)  # Idem
}
