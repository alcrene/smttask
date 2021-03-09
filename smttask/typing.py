import sys
import abc
from numbers import Number
from warnings import warn
import importlib
import inspect
import functools
import numpy as np
import operator

from typing import (Optional, Type, TypeVar,
                    Callable, Iterable, Tuple, List, Sequence, _Final)
from types import new_class
from pydantic.fields import sequence_like

from sumatra.datastore.filesystem import DataFile
from .config import config

import mackelab_toolbox.serialize as mtbserialize

from scipy.stats._distn_infrastructure import rv_generic, rv_frozen
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
RVScalarType = (rv_generic, rv_frozen)
RVMVType = (multi_rv_generic, multi_rv_frozen)
RVFrozenType = (rv_frozen, multi_rv_frozen)
RVType = RVScalarType + RVMVType

PlainArg = (Number, str, np.ndarray)

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
    - They verify `isinstance(v, tuple)` and are equivalent to tuple.
    - They have a type distinct from tuple, so that smttask can recognize and
      treat them differently from a tuple.
    """
    # Setting __origin__ = tuple would allow Pydantic to recognize this as a tuple and
    # validate appropriately. Unfortunately it also removes the `SeparateOutputs`
    # type, which is the whole point of this class
    # __origin__ = Iterable
    __args__: Tuple[Type[T]]

    # result_type: type
    item_type: Type[T]

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
def separate_outputs(item_type: Type[T], get_names: Callable[...,List[str]]):
    """
    In terms of typing, equivalent to `Tuple[T,...]`, but indicates to smttask
    to save each element separately. This was conceived for two use cases:
    1. When the number of outputs is dependent on the input variables.
    2. When some or all of the outputs may be very large. For example, we
       may have a Task which allows different recorder objects to track
       quantities during a simulation.

    Parameters
    ----------
    get_names: Function used to determine the names under which name each
        value is saved. Takes any number of arguments, but their names must
        match the name of a task input. Returns a list of strings.
        E.g., if the associated Task defines inputs 'freq' and 'phase', then
        the `get_names` function may have any one of these signatures:

        - `get_names`() -> List[str]
        - `get_names`(freq) -> List[str]
        - `get_names`(phase) -> List[str]
        - `get_names`(freq, phase) -> List[str]

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

class PureFunctionMeta(type):
    _instantiated_types = {}
    def __getitem__(cls, args):
        """
        Returns a subclass of `PureFunction`
        """
        # Parse the arguments
        # They may consist of only a type, only the modules, or both type and modules.
        callableT = {'inT': None, 'outT': None}
        modules = []
        for a in args:
            if isinstance(a, str):
                modules.append(a)
            elif inspect.ismodule(a):
                for nm, m in sys.modules.items():
                    if m is a:
                        modules.append(nm)
                        break
            elif isinstance(a, list):
                if callableT['inT'] is not None:
                    raise TypeError("Only one input type argument may be specified to"
                                     f"`PureFunction`. Received {callableT['inT']} and {a}.")
                callableT['inT'] = a
            elif isinstance(a, (_Final, type)):
                if callableT['outT'] is not None:
                    raise TypeError("Only one output type argument may be specified to"
                                     f"`PureFunction`. Received {CallableT} and {a}.")
                callableT['outT'] = a
            else:
                raise TypeError("Arguments to the `PureFunction` type can "
                                "consist of zero or one type and zero or more "
                                f"module names. Received {a}, which is of type "
                                f"type {type(a)}.")
        # Treat the callable type, if present
        if (callableT['inT'] is None) != (callableT['outT'] is None):
            raise TypeError("Either both the input and output type of a "
                            "PureFunction must be specified, or neither.")
        if callableT['inT']:
            assert callableT['outT'] is not None
            baseT = Callable[callableT['inT'], callableT['outT']]
            argstr = f"{callableT['inT']}, {callableT['outT']}"
        else:
            baseT = Callable
            argstr = ""
        # Treat the module names, if present
        if modules:
            if argstr:
                argstr += ", "
            argstr += ", ".join(modules)
        # Check if this PureFunction has already been created, and if not, do so
        key = (cls, baseT, tuple(modules))
        if key not in cls._instantiated_types:
            PureFunctionSubtype = new_class(
                f'{cls.__name__}[{argstr}]', (cls,))
            cls._instantiated_types[key] = PureFunctionSubtype
            PureFunctionSubtype.modules = modules
        # Return the PureFunction type
        return cls._instantiated_types[key]
class PureFunction(metaclass=PureFunctionMeta):
    """
    A Pydantic-compatible function type, which supports deserialization.
    A “pure function” is one with no side-effects, and which is entirely
    determined by its inputs.

    Accepts also partial functions, in which case an instance of the subclass
    `PartialPureFunction` is returned.

    .. Warning:: Deserializing functions is necessarily fragile, since there
       is no way of guaranteeing that they are truly pure.
       When using a `PureFunction` type, always take extra care that the inputs
       are sane.

    .. Note:: Functions are deserialized without the scope in which they
       were created.

    .. Hint:: If ``f`` is meant to be a `PureFunction`, but defined as::

       >>> import math
       >>> def f(x):
       >>>   return math.sqrt(x)

       then it has dependency on ``math`` which is outside its scope, and is
       thus impure. It can be made pure by putting the import inside the
       function::

       >>> def f(x):
       >>>   import math
       >>>   return math.sqrt(x)

    .. Note:: Like `Callable`, `PureFunction` allows to specify the type
       within brackets: ``PureFunction[[arg types], return y]``. However the
       returned type doesn't support type-checking.

    .. WIP: One or more modules can be specified to provide definitions for
       deserializing the file, but these modules are not serialized with the
       function.
    """
    modules = []  # Use this to list modules that should be imported into
                  # the global namespace before deserializing the function
    # Instance variable
    func: Callable

    def __new__(cls, func=None):
        # func=None allowed to not break __reduce__ (due to metaclass)
        # – inside a __reduce__, it's fine because __reduce__ will fill __dict__ after creating the empty object
        if cls is PureFunction and isinstance(func, functools.partial):
            # Redirect to PartialPureFunction constructor
            return PartialPureFunction(func)
        return super().__new__(cls)
    def __init__(self, func):
        if hasattr(self, 'func'):
            # This is our second pass through __init__, probably b/c of __new__redirect
            assert hasattr(self, '__signature__')
            return
        self.func = func
        # Copy attributes like __name__, __module__, ...
        functools.update_wrapper(self, func)
        self.__signature__ = inspect.signature(func)
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    ## Function arithmetic ##
    def __abs__(self):
        return CompositePureFunction(operator.abs, self)
    def __neg__(self):
        return CompositePureFunction(operator.neg, self)
    def __pos__(self):
        return CompositePureFunction(operator.pos, self)
    def __add__(self, other):
        return CompositePureFunction(operator.add, self, other)
    def __radd__(self, other):
        return CompositePureFunction(operator.add, other, self)
    def __sub__(self, other):
        return CompositePureFunction(operator.sub, self, other)
    def __rsub__(self, other):
        return CompositePureFunction(operator.sub, other, self)
    def __mul__(self, other):
        return CompositePureFunction(operator.mul, self, other)
    def __rmul__(self, other):
        return CompositePureFunction(operator.mul, other, self)
    def __truediv__(self, other):
        return CompositePureFunction(operator.truediv, self, other)
    def __rtruediv__(self, other):
        return CompositePureFunction(operator.truediv, other, self)
    def __pow__(self, other):
        return CompositePureFunction(operator.pow, self, other)

    ## Serialization / deserialization ##
    # The attribute '__func_src__', if it exists,
    # is required for deserialization. This attribute is added by
    # mtb.serialize when it deserializes a function string.
    # We want it to be attached to the underlying function, to be sure
    # the serializer can find it
    @property
    def __func_src__(self):
        return self.func.__func_src__
    @__func_src__.setter
    def __func_src__(self, value):
        self.func.__func_src__ = value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        if isinstance(value, PureFunction):
            pure_func = value
        elif isinstance(value, Callable):
            pure_func = PureFunction(value)
        elif isinstance(value, str):
            modules = [importlib.import_module(m_name) for m_name in cls.modules]
            global_ns = {k:v for m in modules
                             for k,v in m.__dict__.items()}
            # Since decorators are serialized with the function, we should at
            # least make the decorators in this module available.
            local_ns = {'PureFunction': PureFunction,
                        'PartialPureFunction': PartialPureFunction,
                        'CompositePureFunction': CompositePureFunction}
            pure_func = mtbserialize.deserialize_function(
                value, global_ns, local_ns)
            # It is possible for a function to be serialized with a decorator
            # which returns a PureFunction, or even a subclass of PureFunction
            # In such a case, casting as PureFunction may be destructive, and
            # is at best useless
            if not isinstance(pure_func, cls):
                pure_func = cls(pure_func)
        elif (isinstance(value, Sequence)
              and len(value) > 0 and value[0] == "PartialPureFunction"):
            pure_func = PartialPureFunction._validate_serialized(value)
        elif (isinstance(value, Sequence)
              and len(value) > 0 and value[0] == "CompositePureFunction"):
            pure_func = CompositePureFunction._validate_serialized(value)
        else:
            cls.raise_validation_error(value)
        return pure_func

    # TODO: Add arg so PureFunction subtype can be specified in error message
    @classmethod
    def raise_validation_error(cls, value):
        raise TypeError("PureFunction can be instantiated from either "
                        "a callable, "
                        "a Sequence `([PureFunction subtype name], func, bound_values)`, "
                        "or a string. "
                        f"Received {value} (type: {type(value)}).")

    @staticmethod
    def json_encoder(v):
        if isinstance(v, PartialPureFunction):
            return PartialPureFunction.json_encoder(v)
        elif isinstance(v, CompositePureFunction):
            return CompositePureFunction.json_encoder(v)
        elif isinstance(v, PureFunction):
            f = v.func
        elif isinstance(v, Callable):
            f = v
        else:
            raise TypeError("`PureFunction.json_encoder` only accepts "
                            f"functions as arguments. Received {type(v)}.")
        return mtbserialize.serialize_function(f)

class PartialPureFunction(PureFunction):
    """
    A `PartialPureFunction` is a function which, once made partial by binding
    the given arguments, is pure (it has no side-effects).
    The original function may be impure.
    """
    def __init__(self, partial_func):
        super().__init__(partial_func)

    @classmethod
    def _validate_serialized(cls, value):
        if not (isinstance(value, Sequence)
                and len(value) > 0 and value[0] == "PartialPureFunction"):
            cls.raise_validation_error(value)
        assert len(value) == 3
        assert isinstance(value[1], str)
        assert isinstance(value[2], dict)
        func_str = value[1]
        bound_values = value[2]
        modules = [importlib.import_module(m_name) for m_name in cls.modules]
        global_ns = {k:v for m in modules
                         for k,v in m.__dict__.items()}
        func = mtbserialize.deserialize_function(func_str, global_ns)
        if isinstance(func, cls):
            raise NotImplementedError(
                "Was a partial function saved from function decorated with "
                "a PureFunction decorator ? I haven't decided how to deal with this.")
        return cls(functools.partial(func, **bound_values))


    @staticmethod
    def json_encoder(v):
        if isinstance(v, PureFunction):
            func = v.func
        elif isinstance(v, Callable):
            func = v
        else:
            raise TypeError("`PartialPureFunction.json_encoder` accepts only "
                            "`PureFunction` or Callable arguments. Received "
                            f"{type(v)}.")
        if not isinstance(func, functools.partial):
            # Make a partial with empty dict of bound arguments
            func = functools.partial(func)
        if isinstance(func.func, functools.partial):
            raise NotImplementedError("`PartialPureFunction.json_encoder` does not "
                                      "support nested partial functions at this time")
        return ("PartialPureFunction",
                mtbserialize.serialize_function(func.func),
                func.keywords)

class CompositePureFunction(PureFunction):
    """
    A lazy operation composed of an operation (+,-,*,/) and one or more terms,
    at least one of which is a PureFunction.
    Non-pure functions are not allowed as arguments.

    Typically obtained after performing operations on PureFunctions:
    >>> f = PureFunction(…)
    >>> g = PureFunction(…)
    >>> h = f + g
    >>> isinstance(h, CompositePureFunction)  # True

    .. important:: Function arithmetic must only be done between functions
       with the same signature. This is NOT checked at present, although it
       may be in the future.
    """
    def __new__(cls, func=None, *terms):
        return super().__new__(cls, func)
    def __init__(self, func, *terms):
        if func not in operator.__dict__.values():
            raise TypeError("CompositePureFunctions can only be created with "
                            "functions defined in " "the 'operator' module.")
        for t in terms:
            if isinstance(t, Callable) and not isinstance(t, PureFunction):
                raise TypeError("CompositePureFunction can only compose "
                                "constants and other PureFunctions. Invalid "
                                f"argument: {t}.")
        self.func = func
        self.terms = terms
        if not getattr(self, '__name__', None):
            self.__name__ = "composite_pure_function"

    # TODO? Use overloading (e.g. functools.singledispatch) to avoid conditionals ?
    def __call__(self, *args):
        return self.func(*(t(*args) if isinstance(t, Callable) else t
                           for t in self.terms))

    @classmethod
    def _validate_serialized(cls, value):
        "Format: ('CompositePureFunction', [op], [terms])"
        if not (isinstance(value, Sequence)
                and len(value) > 0 and value[0] == "CompositePureFunction"):
            cls.raise_validation_error(value)
        assert len(value) == 3
        assert isinstance(value[1], str)
        assert isinstance(value[2], Sequence)
        func = getattr(operator, value[1])
        terms = []
        for t in value[2]:
            if (isinstance(t, str)
                or isinstance(t, Sequence) and len(t) and isinstance(t[0], str)):
                # Nested serializations end up here.
                # First cond. catches PureFunction, second cond. its subclasses.
                terms.append(PureFunction.validate(t))
            elif isinstance(t, PlainArg):
                # Either Number or Array – str is already accounted for
                terms.append(t)
            else:
                raise TypeError("Attempted to deserialize a CompositePureFunction, "
                                "but the following value is neither a PlainArg "
                                f"nor a PureFunction: '{value}'.")
        return cls(func, *terms)

    @staticmethod
    def json_encoder(v):
        if isinstance(v, CompositePureFunction):
            assert v.func in operator.__dict__.values()
            return ("CompositePureFunction",
                    v.func.__name__,
                    v.terms)
        else:
            raise NotImplementedError

class RV:
    __slots__ = 'rv', 'frozen', 'gen', 'module'
        # We use __slots__ to prevent infinite recursion in getattr
    # Definining __slots__ means RV has no __dict__, and we need to define
    # our own pickling/unpickling methods
    def __getstate__(self):
        return {nm: getattr(self, nm) for nm in self.__slots__}
    def __setstate__(self, state):
        for nm in self.__slots__:
            setattr(self, nm, state[nm])
    def __init__(self, rv):
        """
        Implements a wrapper around random variables (aka distributions), adding
        the extra methods required by tasks:
            - `__str__`
            - `__hash__`
            - `desc`
            - `valid_desc`
            - `from_desc`
              (Although `from_desc` will not work with frozen multivariate
              distributions.)
        All unwrapped attributes are redirected to the random variable, so
        you should be able to use this instance as a drop-in replacement to
        the original RV.

        Parameters
        ----------
        rv: scipy.stats random variable
            Either a 'bare' or 'frozen' random variable.
            Examples:
                - 'bare'  : scipy.stats.norm,   scipy.stats.cauchy
                - 'frozen': scipy.stats.norm(), scipy.stats.cauchy(-1,1)
        """
        if isinstance(rv, dict) and self.valid_desc(rv):
            rv = self.from_desc(rv, instantiate=False)
        if not isinstance(rv, RVType):
            raise ValueError("`rv` must be a scipy random variable.")
        self.rv = rv
        self.frozen = isinstance(rv, RVFrozenType)
        # Generators are only supposed to be instantiated once, when the
        # stats module is loaded. So we need to find its identifier in the
        # RV module. We do this by searching through the module's variables to
        # find the already instantiated one.
        # This assumes that the generator is always instantiated inside the
        # module where its type is defined, but this seems safe to me – the
        # whole point of these generators is to hide the actual type.
        T = type(rv)
        if self.frozen:
            gen = T.__qualname__
        else:
            gen = None
            for nm,v in vars(sys.modules[T.__module__]).items():
                if isinstance(v, T):
                    gen = nm
                    break
            if gen is None:
                raise ValueError(
                    "Unable to find a generator for random variables of type "
                    f"{str(T)} in {T.__module__}.\n"
                    "(scipy.stats uses generator instances to create "
                    "random variables like `scipy.stats.norm`.)")
        self.gen = gen
        self.module = T.__module__
    # ----------------------------------
    # Emulate underlying RV
    def __getattr__(self, attr):
        if attr in self.__slots__:
            # This attribute should be defined, but isn't (otherwise we
            # wouldn't be here). This is probably because the class isn't
            # yet initialized
            return AttributeError
        else:
            return getattr(self.rv, attr)
    def __call__(self, *args, **kwargs):
        return self.rv(*args, **kwargs)
    @property
    def args(self):
        # Multivariate RVs don't save args, kwds
        return getattr(self.rv, 'args', None)
    @property
    def kwds(self):
        # Multivariate RVs don't save args, kwds
        return getattr(self.rv, 'kwds', None)
    # ----------------------------------
    def __str__(self):
        return self.gen
    def __repr__(self):
        s = '.'.join((self.__module__, self.gen))
        args = self.args
        kwds = self.kwds
        if None in (args, kwds):
            s += "([unknown args])"
        elif self.frozen:
            s += ('(' + ', '.join(args)
                  + ', '.join([f'{kw}={val}' for kw,val in kwds.items()]))
        return s
    def __hash__(self):
        return int(digest(self.desc), base=16)
    @staticmethod
    def valid_desc(desc):
        raise NotImplementedError
        # return utils.is_valid_desc(
        #     desc,
        #     required_keys=['input type', 'generator', 'module', 'frozen'],
        #     optional_keys=['args', 'kwds'],
        #     expected_types={'input type': str, 'generator': str,
        #                     'module': str, 'frozen': bool,
        #                     'args': (tuple, list), 'kwds': dict}
        #     )
    @property
    def desc(self):
        desc = config.ParameterSet({
            'input type': 'Random variable',
            'generator': self.gen,
            'module': self.module,  # Module where constructor is defined
            'frozen': self.frozen,
        })
        if self.frozen:
            if None in (self.args, self.kwds):
                warn("Cannot produce a valid description for a frozen "
                     "distribution if it doesn't not save `args` and `kwds` "
                     "attributes (this happens for multivariate distributions).")
                desc.frozen = 'invalid'  # Will make valid_desc return False
            else:
                desc.args = self.rv.args
                desc.kwds = self.rv.kwds
        return desc
    @classmethod
    def from_desc(cls, desc, instantiate=True):
        """
        Creates and returns a new instance, unless `instantiate` is False,
        In the latter case the unwrapper random variable is returned instead.

        Parameters
        ----------
        desc: dict
            RV description as returned by desc
        instantiate: bool
            Set to False to return an unwrapped random variable, which can
            be used as argument to __init__.
        """
        assert cls.valid_desc(desc)
        desc = config.ParameterSet(desc)
        m = importlib.import_module(desc.module)
        gen = getattr(m, desc.generator)
        if desc.frozen:
            rv = gen(*desc.args, **desc.kwds)
        else:
            rv = gen
        if instantiate:
            return cls(rv)
        else:
            return rv


json_encoders = {
    # DataFile: lambda filename: describe_datafile(filename),
    PureFunction: PureFunction.json_encoder,
    PartialPureFunction: PartialPureFunction.json_encoder,
    set      : lambda s: sorted(s), # Default serializer has undefined order => inconsistent task digests
    frozenset: lambda s: sorted(s)  # Idem
}
