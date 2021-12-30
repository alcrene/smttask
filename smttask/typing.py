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
from .config import config

from mackelab_toolbox.typing import json_like, safe_packages
# Import PureFunction & friends, which used to be defined here, to avoid breaking downstream packages
from mackelab_toolbox.typing import PureFunction, PartialPureFunction, CompositePureFunction

# from scipy.stats._distn_infrastructure import rv_generic, rv_frozen
# from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
# RVScalarType = (rv_generic, rv_frozen)
# RVMVType = (multi_rv_generic, multi_rv_frozen)
# RVFrozenType = (rv_frozen, multi_rv_frozen)
# RVType = RVScalarType + RVMVType

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

T = TypeVar('T')
class Type(typing.Type[T], Generic[T]):
    """
    Make types serializable; the serialization format is
        ('Type', <module name>, <type name>)
    During deserialization, it effectively executes
        from <module name> import <type name>

    .. Caution:: **Bug** As with `typing.Type`, one can indicate the specific type between
       brackets; e.g. ``Type[int]``, and Pydantic will enforce this restriction.
       However at present deserialization only works when the type is unspecified.

    .. Warning:: This kind of serialization will never be 100% robust and
       should be used with care. In particular, since it relies on <module name>
       remaining unchanged, it is certainly not secure.
       Because of the potential security issue, it requires adding modules where
       tasks are defined to the ``smttask.config.safe_packages`` whitelist.
    """
    # FIXME: When the type T is specified, the specialized type doesn't inherit __get_validators__
    #   (although it does inherit the other methods)
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        if isinstance(value, (type, typing._GenericAlias)):
            return value
        elif json_like(value, ["Type", "Type (Generic)"]):
            from importlib import import_module
            if json_like(value, "Type"):
                module = value[1]
                if (any(module.startswith(sm) for sm in config.safe_packages)
                      or config.trust_all_inputs):
                    m = import_module(module)
                    T = getattr(m, value[2])
                else:
                    raise RuntimeError(
                        "As with pickle, deserialization of types can lead to "
                        "arbitrary code execution. It is only permitted after "
                        f"adding '{module_name}' to ``smttask.config.safe_packages`` "
                        "(recommended) or setting the option "
                        "``smttask.config.trust_all_inputs = True``.")
            else:
                baseT = Type.validate(value[1])
                args = tuple(Type.validate(argT) for argT in value[2])
                T = baseT[args]
            return T
        else:
            raise TypeError("Value is neither a type, nor a recognized serialized "
                            f"type. Received: {value} (type: {type(value)})")

    @staticmethod
    def json_encoder(T: typing.Type) -> Tuple[str, str, str]:
        if not isinstance(T, type):
            raise TypeError(f"'{T}' is not a type.")
        if T.__module__ == "__main__":
            raise ValueError("Can't serialize types defined in the '__main__' module.")
        return ("Type", T.__module__, T.__name__)
            
    @staticmethod
    def json_encoder_generic(T: typing._GenericAlias) -> Tuple[str, str, str, Tuple[typing.Type]]:
        if T.__parameters__:  # __parameters__ is the list of non specified types
            if not all(isinstance(argT, typing.TypeVar) for argT in T.__args__):
                raise NotImplementedError(
                    "We only support generic types for which either non of the "
                    "type arguments are specified, or all of them are.\n"
                    f"Type {T} has both.")
            # For non-concrete types, the standard Type serialization format suffices
            # (NB: We can't use Typing.json_encoder, because we need '_name' instead of '__name__'
            return ("Type", T.__module__, T._name)
            # raise NotImplementedError("Only concrete generic types can be serialized. "
            #                           "(So e.g. `List[int]`, but not `List`.)")
        if T.__module__ == "__main__":
            raise ValueError("Can't serialize types defined in the '__main__' module.")
        # TODO: I would prefer returning the bare base type here, instead of nested serializations
        return ("Type (Generic)", ("Type", T.__module__, T._name), T.__args__)
        
    @staticmethod
    def json_encoder_pydantic_generic(T: pydantic.generics.GenericModel
    ) -> Tuple[str, str, str, Tuple[typing.Type]]:
        # Get the base Generic type and type arguments
        #   E.g. For `Foo[int]`, retrieve `Foo` and `(int,)`
        # NB: In contrast to normal generic types, Pydantic Generics don't have
        #   __origin__ or __args__ attributes. But Pydantic maintains a cache
        #   of instantiated generics, to avoid re-instantiating them; this
        #   cache is keyed by the base generic type, and the argument types
        try:
            genT, paramT = next(k for k, v in pydantic.generics._generic_types_cache.items()
                                if v is T)
        except StopIteration:
            # The cache is updated as soon as a concrete generic type is first created,
            # i.e. the first time `Foo[int]` appears.
            # The only way it should happen that T is not in the cache, is if
            # T is a pure generic type. In this case, the normal Type serializer
            # works fine.
            return Type.json_encoder(T)
        
        return ("Type (Generic)", genT, paramT)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='array',
                            items=[{'type': 'string'}]*3)

# class RV:
#     __slots__ = 'rv', 'frozen', 'gen', 'module'
#         # We use __slots__ to prevent infinite recursion in getattr
#     # Definining __slots__ means RV has no __dict__, and we need to define
#     # our own pickling/unpickling methods
#     def __getstate__(self):
#         return {nm: getattr(self, nm) for nm in self.__slots__}
#     def __setstate__(self, state):
#         for nm in self.__slots__:
#             setattr(self, nm, state[nm])
#     def __init__(self, rv):
#         """
#         Implements a wrapper around random variables (aka distributions), adding
#         the extra methods required by tasks:
#             - `__str__`
#             - `__hash__`
#             - `desc`
#             - `valid_desc`
#             - `from_desc`
#               (Although `from_desc` will not work with frozen multivariate
#               distributions.)
#         All unwrapped attributes are redirected to the random variable, so
#         you should be able to use this instance as a drop-in replacement to
#         the original RV.
# 
#         Parameters
#         ----------
#         rv: scipy.stats random variable
#             Either a 'bare' or 'frozen' random variable.
#             Examples:
#                 - 'bare'  : scipy.stats.norm,   scipy.stats.cauchy
#                 - 'frozen': scipy.stats.norm(), scipy.stats.cauchy(-1,1)
#         """
#         if isinstance(rv, dict) and self.valid_desc(rv):
#             rv = self.from_desc(rv, instantiate=False)
#         if not isinstance(rv, RVType):
#             raise ValueError("`rv` must be a scipy random variable.")
#         self.rv = rv
#         self.frozen = isinstance(rv, RVFrozenType)
#         # Generators are only supposed to be instantiated once, when the
#         # stats module is loaded. So we need to find its identifier in the
#         # RV module. We do this by searching through the module's variables to
#         # find the already instantiated one.
#         # This assumes that the generator is always instantiated inside the
#         # module where its type is defined, but this seems safe to me – the
#         # whole point of these generators is to hide the actual type.
#         T = type(rv)
#         if self.frozen:
#             gen = T.__qualname__
#         else:
#             gen = None
#             for nm,v in vars(sys.modules[T.__module__]).items():
#                 if isinstance(v, T):
#                     gen = nm
#                     break
#             if gen is None:
#                 raise ValueError(
#                     "Unable to find a generator for random variables of type "
#                     f"{str(T)} in {T.__module__}.\n"
#                     "(scipy.stats uses generator instances to create "
#                     "random variables like `scipy.stats.norm`.)")
#         self.gen = gen
#         self.module = T.__module__
#     # ----------------------------------
#     # Emulate underlying RV
#     def __getattr__(self, attr):
#         if attr in self.__slots__:
#             # This attribute should be defined, but isn't (otherwise we
#             # wouldn't be here). This is probably because the class isn't
#             # yet initialized
#             return AttributeError
#         else:
#             return getattr(self.rv, attr)
#     def __call__(self, *args, **kwargs):
#         return self.rv(*args, **kwargs)
#     @property
#     def args(self):
#         # Multivariate RVs don't save args, kwds
#         return getattr(self.rv, 'args', None)
#     @property
#     def kwds(self):
#         # Multivariate RVs don't save args, kwds
#         return getattr(self.rv, 'kwds', None)
#     # ----------------------------------
#     def __str__(self):
#         return self.gen
#     def __repr__(self):
#         s = '.'.join((self.__module__, self.gen))
#         args = self.args
#         kwds = self.kwds
#         if None in (args, kwds):
#             s += "([unknown args])"
#         elif self.frozen:
#             s += ('(' + ', '.join(args)
#                   + ', '.join([f'{kw}={val}' for kw,val in kwds.items()]))
#         return s
#     def __hash__(self):
#         return int(digest(self.desc), base=16)
#     @staticmethod
#     def valid_desc(desc):
#         raise NotImplementedError
#         # return utils.is_valid_desc(
#         #     desc,
#         #     required_keys=['input type', 'generator', 'module', 'frozen'],
#         #     optional_keys=['args', 'kwds'],
#         #     expected_types={'input type': str, 'generator': str,
#         #                     'module': str, 'frozen': bool,
#         #                     'args': (tuple, list), 'kwds': dict}
#         #     )
#     @property
#     def desc(self):
#         desc = config.ParameterSet({
#             'input type': 'Random variable',
#             'generator': self.gen,
#             'module': self.module,  # Module where constructor is defined
#             'frozen': self.frozen,
#         })
#         if self.frozen:
#             if None in (self.args, self.kwds):
#                 warn("Cannot produce a valid description for a frozen "
#                      "distribution if it doesn't not save `args` and `kwds` "
#                      "attributes (this happens for multivariate distributions).")
#                 desc.frozen = 'invalid'  # Will make valid_desc return False
#             else:
#                 desc.args = self.rv.args
#                 desc.kwds = self.rv.kwds
#         return desc
#     @classmethod
#     def from_desc(cls, desc, instantiate=True):
#         """
#         Creates and returns a new instance, unless `instantiate` is False,
#         In the latter case the unwrapper random variable is returned instead.
# 
#         Parameters
#         ----------
#         desc: dict
#             RV description as returned by desc
#         instantiate: bool
#             Set to False to return an unwrapped random variable, which can
#             be used as argument to __init__.
#         """
#         assert cls.valid_desc(desc)
#         desc = config.ParameterSet(desc)
#         m = importlib.import_module(desc.module)
#         gen = getattr(m, desc.generator)
#         if desc.frozen:
#             rv = gen(*desc.args, **desc.kwds)
#         else:
#             rv = gen
#         if instantiate:
#             return cls(rv)
#         else:
#             return rv


json_encoders = {
    # DataFile: lambda filename: describe_datafile(filename),
    pydantic.main.ModelMetaclass: Type.json_encoder_pydantic_generic,
    typing._GenericAlias        : Type.json_encoder_generic,
    type                        : Type.json_encoder,
    set                         : lambda s: sorted(s), # Default serializer has undefined order => inconsistent task digests
    frozenset                   : lambda s: sorted(s)  # Idem
}
