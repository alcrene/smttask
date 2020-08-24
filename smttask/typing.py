import sys
import abc
from numbers import Number
from warnings import warn
import importlib
import inspect
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

from typing import Type, TypeVar, Callable, Iterable, Tuple, List
from types import new_class

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

def describe_datafile(datafile: DataFile):
    assert isinstance(datafile, DataFile)
    filename = Path(filename.full_path)
    return {
        'input type': 'File',
        'filename': str(normalize_input_path(filename))
    }

json_encoders = {
    DataFile: lambda filename: describe_datafile(filename)
}

T = TypeVar('T')
# This class derived adapted from pydantic.types.ConstrainedList
class SeparateOutputs(tuple):
    # Needed for pydantic to detect that this is a list
    # __origin__ = tuple
    __args__: Tuple[Type[T], ...]

    item_type: Type[T]
    _get_names: Callable[..., Iterable[str]]
    get_names_args: Tuple[str]

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

# This function is adapted from pydantic.types.conlist
def separate_outputs(item_type: Type[T], get_names: Callable):
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
        value is saved. The arguments of the function must match
        names of the task inputs; those values are passed as arguments.
        Note that at present, Task arguments are not supported for `get_names`.
    """
    sig = inspect.signature(get_names)
    namespace = {'item_type':item_type,
                 '__args__': [item_type, ...],
                 '_get_names': get_names,
                 'get_names_args': tuple(sig.parameters)}
    return new_class('SeparateOutputsValue', (SeparateOutputs,), {},
                     lambda ns: ns.update(namespace))

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
