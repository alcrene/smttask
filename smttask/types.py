import abc
from collections import Iterable
from numbers import Number
from warnings import warn
import importlib
import numpy as np
from sumatra.parameters import NTParameterSet as ParameterSet
from sumatra.datastore.filesystem import DataFile
from . import utils

from scipy.stats._distn_infrastructure import rv_generic, rv_frozen
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
RVScalarType = (rv_generic, rv_frozen)
RVMVType = (multi_rv_generic, multi_rv_frozen)
RVFrozenType = (rv_frozen, multi_rv_frozen)
RVType = RVScalarType + RVMVType

PlainArg = (Number, str, np.ndarray)

InputTypes = (PlainArg, DataFile)
    # Append to this all types that can be used as task inputs
LazyLoadTypes = (DataFile,)
    # Append to this types which aren't immediately loaded
    # Not really used at the moment, other than for some assertions
LazyCastTypes = ()
    # Append to this types which aren't immediately cast to their expected type
    # Not really used at the moment, other than for some assertions
PackedTypes = ()
    # Append to this descriptions of packed arguments, i.e. arguments passed
    # as list or tuples, for which we test (and cast to) the type of the
    # *elements*. Examples: InputTuple, ListOf

_created_types = {}
    # Internal cache of previously created dynamic types
    # This avoid having two distinc InputTuple(int, int) types, for example,
    # which could be confusing.

class InputTuple:
    """
    The purpose of this class is to define type inputs for Tasks returning
    multiple values. If an upstream task returns say a float and either an int
    or a float, then one can specify this by writing
    >>> class Downstream(Task):
    >>>     inputs = {'upstream_data': InputTuple(float, (int, float))}
    The resulting type will cast and validate inputs as expected.
    >>> T = InputTuple(float, (int, float))
    >>> # T(4)
    ValueError: InputTuple_float__int_float_ takes exactly 2 arguments
    >>> T(4,4)
    (4.0, 4)

    The constructor uses a bit of metaclass magic so that `InputTuple`
    creates a *subclass* of InputTuple, rather than an instance, with the
    special attribute `types`:
    >>> T.types
    (float, (int, float))
    """
    def __new__(cls, *types):
        typenames = []
        for T in types:
            if isinstance(T, type):
                typenames.append(T.__qualname__)
            elif (isinstance(T, tuple)
                  and all(isinstance(_T, type) for _T in T)):
                typenames.append('_' + '_'.join(
                    [_T.__qualname__ for _T in T]) + '_')
            else:
                raise ValueError("Arguments to `InputTuple` must be types "
                                 "(or tuples of types).")
        name = 'InputTuple_' + '_'.join(typenames)
        if name in _created_types:
            T = _created_types[name]
        else:
            T = type(name, (InputTuple,), {'types': types})
            T.__new__ = cls.__subclassnew__
            T.__init__ = cls.__subclassinit__
            _created_types[name] = T
        return T
    def __subclassnew__(cls, *args):
        if len(args) != len(cls.types):
            raise ValueError("{} takes exactly {} arguments."
                             .format(cls.__name__, len(cls.types)))
        newargs = (a if isinstance(a, T) else cast(a, T)
                   for a, T in zip(args, cls.types))
        return tuple.__new__(cls, newargs)
    def __subclassinit__(self, *args):
        tuple.__init__(self)
    # TODO: Is it possible to get `isinstance` to work against a tuple ?
InputTypes = InputTypes + (InputTuple,)
PackedTypes = PackedTypes + (InputTuple,)

class ListOf:
    """
    Use this to specify a dependency with is a list.
    Expected element types must be given; use a tuple to specify multiple types.
    The number of elements can optionally be specified; if left blank, any
    number is accepted.
    """
    def __new__(cls, T, n=-1):
        """
        Parameters
        ----------
        T: type, or tuple of types
            The type of expected list elements.
        n: int
            Expected number of list elements. -1 indicates any number.
        """
        # Validate inputs
        if (not isinstance(T, type)
            and not (isinstance(T, tuple)
                     and all(isinstance(_T, type) for _T in T))):
            raise ValueError("`T` arguments to `ListOf` must be a type "
                             "(or tuple of types).")
        if not isinstance(n, int):
            raise ValueError("Argument n to ListOf must be an integer. "
                             f"(is {type(n)})")
        if n == 0 or n < -1:
            warn(f"Argument n={n} to `ListOf` is almost surely a mistake.")
        # Construct name for the new type
        if isinstance(T, tuple):
            typename = '_'.join([_T.__qualname__ in T])
        else:
            typename = T.__qualname__
        name = 'ListOf_' + typename
        if n > -1:
            name += '_' + str(n)
        if name in _created_types:
            listT = _created_types[name]
        else:
            listT = type(name, (ListOf,list), {'type': T, 'n': n})
            _created_types[name] = listT
            listT.__new__ = list.__new__
            listT.__init__ = cls.__subclassinit__
        return listT
    def __subclassinit__(self, arglist):
        if not isinstance(arglist, Iterable):
            raise TypeError("`arglist` is expected to be a list, not "
                            f"{type(arglist)}")
        if self.n != -1 and len(arglists) != self.n:
            raise ValueError(f"Number of arguments ({len(arglist)}) does not "
                             f"match prescribed length ({self.n})")
        castargs = []
        for a in arglist:
           if not isinstance(a, self.type):
               try:
                   castargs.append(cast(a, self.type))
               except TypeError as e:
                   import traceback as tb
                   print("Original error: ")
                   tb.print_tb(e, limit=1)
                   print("Error was re-raised here:")
                   raise TypeError(f"Value {a} is not of, and cannot be casted "
                                   f"to, the prescribed type {self.type}.")
           else:
               castargs.append(a)
        list.__init__(self, castargs)
InputTypes = InputTypes + (ListOf,)
PackedTypes = PackedTypes + (ListOf,)

class File(abc.ABC):
     """Use this to specify a dependency which is a filename."""
     @property
     @abc.abstractmethod
     def root(self):
         raise NotImplementedError("Root depends on whether file is for input "
                                   "or output")
     def __init__(self, filename):
         # outroot = Path(config.project.data_store.root)
         self.filename = Path(filename)
         if self.filename.is_absolute():
             self.filename = utils.relative_path(self.root, filename)
         # self.inputfilename = utils.relative_path(inroot, filename)
     @property
     def full_path(self):
         return self.root.joinpath(self.filename)
    # --------------
    # Standard InputType stuff
     def __str__(self):
         return str(self.filename)
     def __repr__(self):
         return "File({})".format(self.filename)
     def __hash__(self):
         return int(digest(self.desc), base=16)
     @staticmethod
     def valid_desc(desc):
         return utils.is_valid_desc(
            desc,
            required_keys=['input type', 'filename'],
            expected_types={'input type': str, 'filename': str}
            )
     def get_desc(self, filename):
         filename = Path(filename)
         if not filename.is_absolute():
             filename = self.root.joinpath(filename)
         return ParameterSet({
             'input type': 'File',
             'filename': Task.normalize_input_path(
                filename)
         })
     @property
     def desc(self):
         return self.get_desc(self.filename)
     @classmethod
     def from_desc(cls, desc):
         assert cls.valid_desc(desc)
         return cls(desc.filename)

class InputFile(File):
    @property
    def root(self):
        return Path(config.project.input_datastore.root)
class OutputFile(File):
    @property
    def root(self):
        return Path(config.project.data_store.root)
InputTypes = InputTypes + (InputFile,)
LazyLoadTypes = LazyLoadTypes + (InputFile,)

class StatelessFunction:
    """
    Use this to specify a dependency which is a function.
    The function must be stateless, and at some point this may check raise an
    if a class method is specified.
    Please don't do self-defeating undectable things, like using global module
    variables inside your function.

    We use `inspect.getsource` to compute the dependency hash, so any source
    code change will invalidate previous cached results (as it should).
    Changing even the name of the function, even for a lambda function,
    suffices to invalidate the cache.
    """
    def __init__(self, f, name=None):
        """
        If you use a nameless function (i.e. lambda) for `f`, consider setting
        the name attribute after creation to something more human-readable.

        Parameters
        ----------
        f: Callable
            You must ensure that `f` is stateless; the initializer is unabl
            to verify this.
        name: str
            Defaults to `f.__qualname__`, or if that fails, `str(f)`.

        Examples
        -------
        >>> x = StatelessFunction(np.arange)
        >>> x.name
        "arange"
        >>> y = StatelessFunction(lambda t: t**2)
        >>> y.name
        "<function <lambda> at 0x7f9bf42abb00>"
        >>> z = StatelessFunction(lambda t: t**2, name='z')'
        >>> z.name
        "<lambda z>"
        """
        if not isinstance(f, Callable):
            raise ValueError("`f` argument must be callable.")
        self.f = f
        if f.__name__ == "<lambda>":
            if name is None:
                self.name = str(f)
            else:
                self.name = "<lambda {}>".format(name)
            warn("Using a lambda function as a dependency is fragile "
                 "(e.g. the name to which you assign it changes the hash). "
                 "Consider adding the function to a module with a proper "
                 "definition.\nLambda function name: {}".format(self.name))
        else:
            self.name = (name if name is not None
                         else getattr(f, '__qualname__', str(f)))
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    # --------------
    # Standard InputType stuff
    def __str__(self):
        return self.name
    def __repr__(self):
        return '.'.join((self.f.__module__, self.name))
    def __hash__(self):
        return int(digest(self.desc), base=16)
    @staticmethod
    def valid_desc(desc):
        return utils.is_valid_desc(
            desc,
            required_keys=['input type', 'module', 'srcname'],
            optional_keys=['name'],
            expected_types={'input type': str, 'module': str,
                            'srcname': str, 'name': str})
    @property
    def desc(self):
        desc = ParameterSet({
            'input type': 'Function',
            #'source'    : inspect.getsource(self.f),
            'module'    : self.f.__module__,
            'srcname'   : self.f.__qualname__
            })
        if '.' in desc.srcname:
            raise NotImplementedError("It seems the function '{}' is a method, "
                                      "which is currently unsupported."
                                      .format(desc.srcname))
        if self.name != desc.srcname:
            # For display it can be useful to have a shortened name
            # but it's distracting to save redundant info in the desc
            desc['name'] = self.name
    @classmethod
    def from_desc(cls, desc):
        assert cls.valid_desc(desc)
        m = importlib.import_module(desc.module)
        f = getattr(m, desc.srcname)
        name = desc.get('name', None)
        return cls(f, name)
InputTypes = InputTypes + (StatelessFunction,)

import sys
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
        return utils.is_valid_desc(
            desc,
            required_keys=['input type', 'generator', 'module', 'frozen'],
            optional_keys=['args', 'kwds'],
            expected_types={'input type': str, 'generator': str,
                            'module': str, 'frozen': bool,
                            'args': (tuple, list), 'kwds': dict}
            )
    @property
    def desc(self):
        desc = ParameterSet({
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
        desc = ParameterSet(desc)
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
InputTypes = InputTypes + (RV,)

# ============================
# Cast function
# ============================

def cast(value, totype, input_or_output=None):
    """
    For each parameter θ, this method will check whether it matches the
    expected type T (as defined by `self.inputs`). If it doesn't, it tries
    to cast it to that type (i.e. that the T has an attribute
    `castable`, and that `isinstance(θ, T)` is True; in this case
    the value is cast to the appropriate type.
    If the input descriptor defines more than one possible input, left-
    most types in the definition take precedence. All types are checked
    for an exact match before casting is attempted.

    Parameters
    ----------
    value:
        Value to cast
    totype: type | tuple of types
        Type to which to cast. If multiple, left-most takes precedence.
        [NOT TRUE: Casting is only attempted with types with a `castable` attribute;
        `castable` attribute must be a list of types.]
    input_or_output: 'input' | 'output'
        Specify whether we are casting an input or output parameter.
        This is required for File parameters.

    Raises
    ------
    TypeError:
        If unable to cast.
    ValueError:
        If tried to cast to File type without specifying `input_or_output`.
    """
    if isinstance(value, totype):
        # Don't cast if `value` is already of the right type
        return value
    elif isinstance(totype, tuple):
        for T in totype:
            try:
                r = cast(value, T, input_or_output)
            except (ValueError, TypeError):
                pass
            else:
                return r
        raise TypeError("Unable to cast {} to type {}"
                        .format(value, totype))
    else:
        T = totype
        if T is File:
            if input_or_output is None:
                raise ValueError("`input_or_output` must be specified to "
                                 "cast `File` arguments.")
            else:
                T = {'input': InputFile, 'output': OutputFile}[input_or_output]
        elif T is np.ndarray:
            T = np.array
        elif isinstance(value, RVType):
            T = RV
        return T(value)
