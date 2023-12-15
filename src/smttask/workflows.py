"""
Assembling Tasks into workflows
===============================

This module provides a pair of functions which can be used to combine multiple
tasks into containerized *workflows*, with a single set of parameters.

It borrows ideas from both `papermill`_ and `NotebookScripter`_. The idea to have
a *workflow* module, which defines global parameters and instantiates multiple
tasks depending on those parameters. A separate module (or interactive session)
may run the workflow module multiple times with different parameters.

The working principle is very similar to NotebookScripter.

Differences with NotebookScripter:

- Works with plain Python modules

  + To execute Jupyter notebooks, pair them first to a Python module with Jupytext.
  
- Less boilerplate

  - Workflow parameters are declared as normal global parameters, rather than
    retrieved with a ``receive_parameter`` function.
    
    + This makes scripts more portable and easier to read.
    + Since this is also the format used by papermill, workflows can also be
      executed with papermill, if desired.
      
- Fewer execution options

  + No option to run in a separate process.
  + No Jupyter-specific functionality.
  
- The workflow module is imported as a normal module, in addition to being
  returned.

Differences with papermill:

- Works with plain Python modules

  + To execute Jupyter notebooks, pair them first to a Python module with Jupytext.
  
- Works will with any arguments – not just plain types like `str` and `float`
  
  + (Papermill converts all arguments to strings before passing them to the
  notebook being executed.)
  
- Workflows are run in the same process as the calling module

- Negligible overhead
  
  + Papermill creates a new anonymous notebook on each call, which in my
  experience can take up to 10 seconds. This can be an issue when attempting
  to do a parameter scan by repeating a workflow multiple. (Note that if the
  workflow only *instantiates* tasks, in most cases it should complete in less
  than a second).

Usage
-----

.. code-block::
   :caption: workflow_module.py

   from smttask.workflows import set_workflow_args
   
   # Default values for workflow parameters
   a1 = 0
   b1 = 1.
   
   # If executed externally, replace the parameters with those passed as args
   # (No effect if workflow_module.py is called directly)
   set_workflow_args(__name__, globals())

   taskA = Task1(a=a1, b=b1)
   taskB = Task2(a=task1, b=...)  # Workflow => Make Task2 depend on Task1
   
.. code-block::
   :caption: runfile.py
   
   from smttask.workflows import run_workflow
   
   wf_list = [run_workflow(a1=a1) for a1 in (-1, 0, 1, 2)]
      # Uses default value for b1

   # Execute all workflows:
   for wf in wf_list:
       wf.taskB.run()

   # Alternatively, save tasks for later execution
   for wf in wf_list:
       wf.taskB.save()

.. Note::
   Only parameters already defined before the call to `set_workflow_args` will
   be replaced.

.. Note::
   The implementation uses a global variable is used to pass parameters between
   modules. The is definitely a hack, but probably no worse than the magic
   that papermill or NotebookScripter themselves use.

.. _papermill: papermill.readthedocs.io/en/latest/
.. _NotebookScripter: https://github.com/breathe/NotebookScripter
"""
from __future__ import annotations

__all__ = ["run_workflow", "set_workflow_args", "SeedGenerator", "_ParamColl"]

# NB: __getattr__ defined in Parameter Collections section

#############################################################################
##                     Running notebooks as scripts                       ###
#############################################################################

script_args = {}
previously_run_workflows = {}  # Used to prevent running a workflow after it has been modified
def run_workflow(module_name: str, package: str=None,
                 exenv: str="workflow", exenv_var="exenv", **parameters
    ) -> module:
    """
    Import (or reload) a module, effectively executing it as a script.
    The imported module can retrieve parameters, which are stored in
    `wcml.utils.script_args`.

    To allow the module to detect when it is being run as a workflow, an
    "execution environment" variable is injected into its global namespace.
    The default name for this variable is ``exenv`` and the default value
    ``"workflow"``; these defaults can be changed with the `exenv_var` and
    `exenv` parameters respectively.
    
    The (re)imported module is returned, allowing to retrieve values defined
    in its namespace.
    
    .. Important::
       For this to work, the script must include a call to `set_workflow_args`.
       
    .. Note::
       Workflow files must not be modified between calls to `run_workflow`:
       Python's introspection is not 100% robust with regards to reloaded modules,
       which may break smttask's reproducibility guarantees. (In particular,
       `inspect.getsource`, which is used to serialize functions, may return
       incorrect results.)

    Parameters
    ----------
    module_name: Name of the module as it appears in sys.modules
    package: If the module has not yet been imported, this is passed
        to `importlib.import_module`.
        It is required when `module_name` is relative.
    exenv: The value to which to set the global execution environment variable.
        If `None`, no variable is injected.
    exenv_var: The name of the global execution environment variable in the
        workflow module.
        If `None`, no variable is injected.
        
    **parameters: Parameters to pass to the script
    
    Returns
    -------
    The (re)imported module.
    
    See also
    --------
    set_workflow_args
    """
    global script_args, previously_run_workflows
    import importlib
    import sys
    import inspect
    from .hashing import stableintdigest
    parameters = parameters.copy()
    if exenv is not None and exenv_var is not None:
        parameters[exenv_var] = exenv
    script_args[module_name] = parameters
    if module_name in sys.modules:
        m_old = sys.modules[module_name]
        # Do the next check before trying to reload, in case a modification causes reload to fail
        previous_hash = previously_run_workflows.get(module_name)
        if previous_hash:
            if stableintdigest(inspect.getsource(m_old)) != previous_hash:
                raise RuntimeError(f"Workflow files (here: '{module_name}') "
                                   "must not be modified between calls to `run_workflow`.")
        m = importlib.reload(m_old)
        new_hash = stableintdigest(inspect.getsource(m))
        if previous_hash is None:
            previously_run_workflows[module_name] = new_hash
        elif new_hash != previous_hash:
            # There may be redundancy between this check and the other one; not sure if one check can catch all cases
            raise RuntimeError(f"Workflow files (here: '{module_name}') "
                               "must not be modified between calls to `run_workflow`.")
    else:
        m = importlib.import_module(module_name, package=package)
        previously_run_workflows[module_name] = stableintdigest(inspect.getsource(m))
    return m

def set_workflow_args(__name__: str, globals: Dict[str,Any], existing_only: bool=False):
    """
    To allow a notebook to be executed with `run_workflow`, place this
    immediately below its parameter block:
    
        set_workflow_args(__name__, globals())
        
    :param:existing_only: If `True`, only variables already defined in the
       module before the call to `set_workflow_args` will be replaced by values
       passed to `run_workflow`.
        
    .. todo:: Make __name__ and globals optional, using the stack trace to
       get values which work in most situations.
       
    See also
    --------
    run_workflow
    """
    if __name__ != "__main__":  # Make this call safe for interactive sessions
        # Running within an import
        #  - if run through `run_workflow`, there will be parameters in `script_args`
        #    which should replace the current values
        if __name__ in script_args:
            for k, v in script_args[__name__].items():
                if not existing_only or k in globals:
                    globals[k] = v

#############################################################################
##                           Seed Generator                               ###
#############################################################################

import dataclasses
import numbers
import numpy as np
from collections import namedtuple
from collections.abc import Sequence as Sequence_
from dataclasses import dataclass, fields, InitVar
from typing import Union, ClassVar, NamedTuple
from typing import Union, List, Tuple  # Types required to serialize SingleSeedGenerator: used in scityping.numpy.SeedSequence
from scityping.numpy import NPValue, Array, SeedSequence
from .hashing import stableintdigest
from .utils import flatten

def _normalize_entropy(key) -> Union[int,Tuple[int,...]]:
    """
    Convert a key to something consumable by `SeedSequence`.
    Key may be a scalar or tuple of scalars
    Accepted types are int, float, str.

    Sequences of length one are converted to scalars.
    This is mostly for consistency in hashes and string outputs: the
    resulting SeedSequence is the same.
    """
    if isinstance(key, numbers.Integral):
        return key
    elif isinstance(key, (list, tuple)):
        return tuple(_normalize_entropy(_key) for _key in key)
        # return tuple(_key if isinstance(key, numbers.Integral)
        #              else stableintdigest(_key)
        #              for _key in key)
    else:
        return stableintdigest(key)
        # raise TypeError("Unrecognized format for `key`")

class SingleSeedGenerator(SeedSequence):
    """
    Make SeedSequence callable, allowing to create multiple high-quality seeds.
    On each call, the arguments are converted to integers (via hashing), then 
    used to produce a unique seed.
    By default, a single integer seed is returned per call.
    To make each call return multiple seeds, use the keyword argument `length`
    when creating (not calling) the seed generator. When `length` is greater
    than one, seeds are returned as NumPy vector.

    .. Important:: Multiple calls with the same arguments will return the same seed.
    .. Note:: Recommended usage is through `SeedGenerator`.
    """
    class Data(SeedSequence.Data):
        length: int
        def encode(seedseq: SingleSeedGenerator) -> SingleSeedGenerator.Data:
            return (*SeedSequence.Data.encode(seedseq), seedseq.length)

    def __init__(self, *args, length=1, **kwargs):
        self.length = length  # The generated state will have this many integers
        super().__init__(*args, **kwargs)
    def __call__(self, *key,
        ) -> Union[NPValue[np.unsignedinteger], Array[np.unsignedinteger, 1]]:
        """
        If length=1, returns an integer. Otherwise returns an array.
        """
        # Convert keys to ints. stableintdigest also works with ints, but
        # returns a different value – it seems that this could be suprising,
        # so if we don't need to, we don't apply it.
        key = flatten(_normalize_entropy(key))
        seed = np.random.SeedSequence(self.entropy,
                                      spawn_key=(*self.spawn_key, *key)
               ).generate_state(self.length)
        if self.length == 1:
            seed = seed[0]
        return seed
    def __str__(self):
        "Returns a shorter string than `SeedSequence`, more appropriate for being "
        "part of an argument list. Entropy is not printed and keywords removed"
        return f"{type(self).__qualname__}({self.spawn_key})"

@dataclass
class SeedGenerator:
    """
    Maintain multiple, mutually independent seed generators.
    
    For each named seed, the length of the required state vector is inferred
    from the annotations. For example,
    
    >>> class SeedGen(SeedGenerator):
    >>>     data: Tuple[int, int]
    >>>     noise: int
            
    will generator length-2 state vectors for ``data`` and scalars for ``noise``.
    (Length 1 arrays are automatically converted to scalars.)
    
    .. Important:: Only the types ``int`` and ``Tuple`` are supported.
       The ellipsis (``...``) argument to ``Tuple`` is not supported.
       Other types may work accidentally.
       
    .. rubric:: Usage
    
    Initialize a seed generator with a base entropy value (for example obtained
    with ``np.random.SeedSequence().entropy``):
    
    >>> seedgen = SeedGen(entropy)
        
    Seeds are generated by providing a key, which can be of any length and contain
    either integers, floats or strings:
    
    >>> seedgen.noise(4)
    # 760562028
    >>> seedgen.noise("tom", 6.4)
    # 3375185240
    
    To get a named tuple storing a seed value for every attribute, pass the key
    to the whole generator
    
    >>> seedgen(4)
    # SeedValues(data=array([2596421399, 1856282581], dtype=uint32), noise=760562028)
    
    Note that the value generated for ``seedgen.noise`` is the same, because
    the same key was used.
    """
    entropy: InitVar[int]
    SeedValues: ClassVar[NamedTuple]
    def __init__(self, entropy):
        # seedseq = np.random.SeedSequence(_normalize_entropy(entropy))
        # seednames = [nm for nm, field in self.__dataclass_fields__.items()
        #              if field._field_type == dataclasses._FIELD]
        # for nm, sseq in zip(seednames, seedseq.spawn(len(seednames))):
        #     setattr(self, nm, sseq.generate_state(1)[0])
        entropy = tuple(flatten(_normalize_entropy(entropy)))
        for i, field in enumerate(fields(self)):
            # Get the length of the required state from the annotations
            # seedT = self.__dataclass_fields__[nm].type
            length = len(getattr(field.type, "__args__", [None]))  # Defaults to length 1 if type does not define length
            # Set the seed attribute
            setattr(self, field.name, SingleSeedGenerator(entropy, spawn_key=(i,), length=length))
    def __init_subclass__(cls):
        # Automatically decorate all subclasses with @dataclasses.dataclass
        # We want to use the __init__ of the parent, so we disable automatic creation of __init__
        dataclasses.dataclass(cls, init=False)
        seednames = [field.name for field in fields(cls)]
        # seednames = [nm for nm, field in cls.__dataclass_fields__.items()
        #              if field._field_type == dataclasses._FIELD]
        cls.SeedValues = namedtuple(f"SeedValues", seednames)
    def __call__(self, key):
        return self.SeedValues(**{nm: getattr(self, nm)(key)
                                  for nm in self.SeedValues._fields})

#############################################################################
##                        Parameter Collections                           ###
#############################################################################


import logging
import numpy as np
from abc import ABC
from collections.abc import Mapping, Sequence as Sequence_, Generator
from functools import partial
from typing import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
try:
    from dataclasses import KW_ONLY
except ImportError:
    # With Python < 3.10, all parameters in subclasses will need to be specified, but at least the code won’t break
    from scityping import NoneType
    KW_ONLY = NoneType
from itertools import chain, product, repeat, islice
from math import prod, inf, nan
from typing import ClassVar, Literal, Union, Any, Dict, List
from scityping import Serializable, Dataclass, NoneType
from scityping.utils import get_type_key

# from scityping import dataclass  # This is the dataclass type scityping uses.
#                                  # It will be serializable if Pydantic is installed
from .hashing import stableintdigest

try:
    from numpy.typing import ArrayLike
except:
    ArrayLike = "ArrayLike"
try:
    from scipy import stats
except ModuleNotFoundError:
    stats = None
try:
    from scityping.holoviews import Dimension
except ModuleNotFoundError:
    Dimension = str

logger = logging.getLogger(__name__)


# NB: One might be tempted to use the ability of ABCs to register abstract subclasses
# and do the following:
#     Sequence_.register(np.ndarray)
# Then tests within expand() and ExpandableSequence would only need to check for
# an instance of Sequence. Indeed, this is the recommended mechanism if other
# libraries want to make their Sequence-compatible expand like sequences.
# 
# However, registering NumPy arrays as sequences causes a *really* confusing bug
# if one is also using JAX, or any other library which uses `pytrees`.
# (`pytree` has special treatment of NumPy arrays, but this is broken if arrays
# register as instances of Sequence)

Seed = Union[int, ArrayLike, np.random.SeedSequence, None]
class NOSEED:  # Default argument; used to differentiate `None`
    pass
# Scipy.stats does not provide a public name for the frozen dist types
if stats:
    RVFrozen = next(C for C in type(stats.norm()).mro()[::-1] if "frozen" in C.__name__.lower())
    MultiRVFrozen = next(C for C in type(stats.multivariate_normal()).mro()[::-1] if "frozen" in C.__name__.lower())
else:
    class RVFrozen:  # These are only used in isinstance() checks, so an empty
        pass         # class suffices to avoid those tests failing and simply return `False`
    class MultiRVFrozen:
        pass

## Pickling of dynamic types requires being able to look them up in the module ##

def __getattr__(attr):
    ParamCollType = {PColl.__qualname__: PColl
                     for PColl in _paramcoll_registry.values()}.get(attr)
    if ParamCollType:
        return ParamCollType
    else:
        raise AttributeError(f"Module '{__name__}' does not define '{attr}'.")

## `expand` function ##

_expandable_types = (Sequence, RVFrozen, MultiRVFrozen)
def expand(obj: Union[_expandable_types]):
    if isinstance(obj, (Sequence_, np.ndarray)):  # See above for why we don’t register ndarray as a subclass of Sequence
        return ExpandableSequence(obj)
    # elif isinstance(obj, Mapping):
    #     return ExpandableMapping(obj)
    elif isinstance(obj, RVFrozen):
        return ExpandableUniRV(obj)
    elif isinstance(obj, MultiRVFrozen):
        return ExpandableMultiRV(obj)
    else:
        raise TypeError(
            f"Argument must be an instance of one of the following types: {_expandable_types}. "
            "Note that Sequence and Mapping are abstract classes, "
            "so if you know your argument type is compatible with them, "
            "you can indicate this by registering it as a virtual subclass:\n"
            "    from collections.abc import Sequence\n"
            "    Sequence.register(MyType)")

def str_rv(rv: RVFrozen):
    # RVFrozen instances all follow a standard pattern, which we can use for better str representation
    # Unfortunately MultiRVFrozen types are not so standardized
    argstrs = [str(a) for a in rv.args]
    argstrs += [f"{subk}={subv}"
                for subk,subv in rv.kwds.items()]
    return f"{rv.dist.name}({', '.join(argstrs)})"

@dataclass
class Expandable(ABC):
    pass

@Expandable.register
@dataclass
class ExpandableSequence(Sequence):
    _seq: tuple
    def __post_init__(self):
        if not isinstance(self._seq, (Sequence_, np.ndarray)):  # See above for why we don’t register ndarray as a subclass of Sequence
            raise TypeError("`seq` must be a Sequence (i.e. a non-consuming iterable).\n"
                            "If you know your argument type is compatible with a Sequence, "
                            "you can indicate this by registering it as a virtual subclass:\n"
                            "    from collections.abc import Sequence\n"
                            "    Sequence.register(MyType)")
    @property
    def length(self):
        return len(self._seq)
    def __len__(self):
        return self._seq.__len__()
    def __getitem__(self, key):
        return self._seq.__getitem__(key)
    def __str__(self):
        return str(self._seq)
    def __repr__(self):
        return f"~({repr(self._seq)})"
    def __eq__(self, other):
        return self._seq == other
    # def __getattr__(self, attr):
    #     return getattr(self._seq, attr)

# @Expandable.register
# class ExpandableMapping(Mapping):
#     def __init__(self):
#         if not isinstance(self.map, Mapping):
#             raise TypeError("`map` must be a Mapping (i.e. a non-consuming iterable).\n"
#                             "If you know your argument type is compatible with a Sequence, "
#                             "you can indicate this by registering it as a virtual subclass:\n"
#                             "    from collections.abc import Mapping\n"
#                             "    Mapping.register(MyType)")
#     def __str__(self):
#         return str(self._map)
#     def __repr__(self):
#         return f"~({repr(self._map)})"
#     def __getattr__(self, attr):
#         return getattr(self._map, attr)

def _make_rng_key(key: Union[int,tuple,str]):
    """Convert a nested tuple of ints and strs to a nested tuple of just ints,
    which can be consumed by SeedSequence.
    """
    if isinstance(key, str):
        # SeedSequence expects a uint32, which is 4 bytes.
        # stableintdigest(*, 4) returns an integer exactly between 0 and 2**32
        # (We don't need to specify 4, because it's the default)
        return stableintdigest(key) 
    elif isinstance(key, tuple):
        return tuple(_make_rng_key(k) for k in key)
    elif isinstance(key, np.random.SeedSequence):
        return key.generate_state(1)
    else:
        return key

@Expandable.register
@dataclass
class ExpandableRV(ABC):
    _rv: Union[RVFrozen, MultiRVFrozen]
    def __getattr__(self, attr):
        if attr == "_rv":  # Prevent infinite recursion
            raise AttributeError(f"No attribute '{attr}'.")
        return getattr(self._rv, attr)
    @property
    def length(self):
        return inf
    def make_iter(self, seed: Union[int,tuple,str], size: Optional[int]=None,
                  max_chunksize: int=1024):
        """
        Return an amortized infinite iterator: each `rvs` call requests twice
        as many samples as the previous call, up to `max_chunksize`.
        """
        rng = np.random.Generator(np.random.PCG64(_make_rng_key(seed)))
        if size is None:
            # Size unknown: Return an amortized infinite iterator
            chunksize = 1
            while True:
                chunksize = min(chunksize, max_chunksize)
                yield from self._rv.rvs(chunksize, random_state=rng)
                chunksize *= 2
        else:
            # Size known: draw that many samples immediately
            k = 0
            while k < size:
                chunksize = min(size-k, max_chunksize)
                yield from self._rv.rvs(chunksize, random_state=rng)
                k += chunksize

class ExpandableUniRV(ExpandableRV):
    _rv: RVFrozen
    def __post_init__(self):
        if not isinstance(self._rv, RVFrozen):
            raise TypeError("`rv` must be a frozen univariate random variable "
                            "(i.e. a distribution from scipy.stats with fixed parameters).\n")
        # self._rv = rv
    def __str__(self):
        return str_rv(self._rv)
    def __repr__(self):
        return f"~{str_rv(self._rv)}"

class ExpandableMultiRV(ExpandableRV):
    _rv: MultiRVFrozen
    def __post_init__(self):
        if not isinstance(rv, MultiRVFrozen):
            raise TypeError("`rv` must be a frozen multivariate random variable "
                            "(i.e. a multivariate rdistribution from scipy.stats with fixed parameters).\n")
        # self._rv = rv

## ParamColl ##
# NOTE also how __getattr__ above allows retrieving dynamically created types

# When we autocreate new subclasses of ParamColl from plain dataclasses, we
# add them to a registry, to avoid recreating the same subclasses multiple times.
# Among other things, this avoids bugs where things should compare equal but don't.
_paramcoll_registry = {}

def _get_paramcoll_type(dataclass_type: type):
    if issubclass(dataclass_type, ParamColl):
        return dataclass_type
    elif dataclass_type in _paramcoll_registry:
        return _paramcoll_registry[dataclass_type]
    else:
        ParamColl_fieldnames = {field.name for field in fields(ParamColl)}
        v_fieldnames = {field.name for field in fields(dataclass_type)}
        if v_fieldnames & ParamColl_fieldnames:
            raise RuntimeError(
                f"Dataclass {type(v)} contains expandable parameters, "
                "but it cannot be converted to ParamColl, because the "
                f"following field names conflict: {v_fieldnames & ParamColl_fieldnames}.")
        dcp = dataclass_type.__dataclass_params__
        dataclass_params = {param: getattr(dcp, param) for param in dcp.__slots__}
        NewParamColl = dataclass(**dataclass_params)(
            type("ParamColl_" + dataclass_type.__qualname__,
                 (ParamColl, dataclass_type),
                 {"__orig_dataclass_type__": dataclass_type}))
        NewParamColl.__module__ = __name__
        _paramcoll_registry[dataclass_type] = NewParamColl
        return NewParamColl

def _create_paramcoll_from_data(data: dict, target_type: type):
    ParamCollType = _get_paramcoll_type(target_type)
    return ParamCollType(**data)

@dataclass(frozen=True)
class ParamColl(Mapping, Dataclass):
    """
    A container for parameter sets, which allows expanding lists parameters.
    Implemented as a dataclass, with an added Mapping API to facilitate use for
    keyword arguments.

    Parameters
    ----------
    dims: Optional dictionary of Holoviews dimensions; these are used only
       for the `kdims` property.
    paramseed: ("collection seed") Seed to use when expanding random variables.
    inner_len: Set the length of the collection, when it cannot otherwise be
       determined. Only has an effect when all expandable parameters are
       random variables.

    .. rubric:: Nested parameter collections

       Nested `ParamColl` instances are permitted, and will expand as expected.
       As a convenience, if a `ParamColl` contains plain dataclasses with
       `Expandable` parameters, those dataclasses are converted to `ParamColl`
       instances with the same dataclass arguments (init, frozen, unsafe_hash, etc.)

       .. NOTE:: An autocreated dataclass may inherit the `frozen` arguments,
          in which case `dims`, `paramseed` and `inner_len` can no longer be
          modified once the class is created.

    .. rubric:: Expandable parameters

    - `outer()` will expand every expandable parameter separately and
      return a new `ParamColl` for each possible combination.
      This is akin to itertools’s `product`, or a mathematical outer product.
    - `inner()` will expand every expandable parameter simultaneously and
      return a new `ParamColl` for each combination.
      This is akin to `zip`, or a mathematical inner product.
    Parameters are made expandable by wrapping an iterable with `smttask.workflows.expand`.

    .. rubric:: Random parameters

    So called “frozen” random variables from `scipy.stats` may be used as
    parameters. To ensure reproducibility, in this case a seed *must* be
    specified. (If you really want different values on each call, pass
    `None` as the seed.)
    If there are only random (and possibly scalar parameters), the `ParamColl`
    is of infinite size. To make it a finite iterator, set the `inner_len`
    attribute.
    If there are also expandable parameters, the `ParamColl` has inner/outer
    size determined by the expandable parameters.

    .. rubric:: Use as keyword arguments

    The primary use case for `ParamColl` instances is as keyword arguments.
    To make this easier, instances provide a mapping interface:
    if ``params`` is a `ParamColl` instance, then ``f(**params)`` will pass
    all its public attributes as keyword arguments.

    .. rubric:: Private attributes

    Attributes whose name start with an underscore ``_`` are private:
    they are excluded from the values returned by
    `.keys()`, `.values()`, `.items()`, and `.kdims`.

    .. rubric:: Parameter dimensions

    Dimension instances (such as those created with `holoviews.Dimension`)
    can be assigned to parameters by updating the class’s `dims` dictionary.
    Keys in `dims` should correspond to parameter names.
    `kdims` will preferentially return the values in `dims` when a dimension
    matching a parameter name is found, otherwise it returns the parameter name.

    .. rubric:: Reserved names

    The following names are used for either attributes or methods of the
    base class, and therefore should not be used as parameter names:
    - dims
    - kdims
    - paramseed
    - inner_len
    - outer_len
    - inner
    - outer
    - keys
    - values
    - items

    .. rubric:: Hashing

    Parameter colls can be made hashable by specifying ``unsafe_hash=True``
    to the dataclass

    .. code::python
       @dataclass(unsafe_hash=True)
       class MyParams(ParamColl):
         ...

    Hashes are then computed according to the parameter names, parameter
    values and their seed. (Internally managed attributes are excluded from
    the hash). One reason to make parameter colls hashable is to use them as
    dictionary keys. Note however, that since they are also mutable, this
    breaks the Python convention that only immutable objects are hashable.
    In short, if you choose to modify a `ParamColl` instance after creation,
    avoid also using its hash; hashing its children (for example the param
    colls produced with `inner` or `outer`) is fine. Note that is always
    possibly to create a new parameter collection instead of mutating an old
    one.
    """
    # Users can optionally expand `dims` with hv.Dimension instances
    # Missing dimensions will use the default ``hv.Dimension(θname)``
    dims    : ClassVar[Dict[str, Dimension]] = {}
    _       : KW_ONLY = None  # kw_only required, otherwise subclasses need to define defaults for all of their values. Assigning `= None` allows this to work for <3.10
    paramseed  : Union[Seed,Literal[NOSEED]] = field(default=NOSEED, repr=None)  # NB: kw_only arg here would be cleaner, but would break for Python <3.10
    # inner_len : InitVar[Optional[int]]  = field(default=None, repr=False)                              # A default value has no effect because this is a @property. Note that because this is included in the hash, and cannot be changed if we subclass a frozen dataclass
    # _inner_len: Optional[int] = field(default=None, init=False, repr=False, compare=False)  # Default is used in place of `inner_len` default on instantiation – see inner_len.setter

    _lengths: List[Union[Literal[inf], int]] = field(init=False, repr=False, compare=False)
    _initialized: bool = field(default=False, init=False, repr=False, compare=False)
    
    class Data(Dataclass.Data):
        def encode(dc):
            T, kwds = Dataclass.Data.encode(dc)
            T = getattr(T, "__orig_dataclass_type__", T)  # For dynamically created classes, we need to save the non-dynamic type
            kwds.pop("_", None)  # For Python <3.10, we need to remove the _ field when encoding
            return (T, kwds)
        def decode(data):
            T = _get_paramcoll_type(data.type)
            return T(**data.data)

    @classmethod
    def validate(cls, value, field=None):  # `field` not currently used: only there for consistency
        try:
            return super().validate(value, field)
        except TypeError as e:
            if cls is not ParamColl and str(e).startswith("Serialized data does not match any of the registered"):
                # `value` has the form of serialized data, but indicates a type which is not a subclass of `cls`.
                # It may be a serialized dynamic type, in which case we can deserialize it with ParamColl
                newval = ParamColl.validate(value, field)  # Let ParamColl.validate raise an error if a problem occurs
                # The new value must still be an instance of `cls`
                if not isinstance(newval, cls):
                    raise TypeError(f"Field expects a value of type `{cls.__qualname__}`, but the "
                                    f"provided value deserialized to a object o type `{type(newval).__qualname__}`")
                return newval
            else:
                raise e

    @classmethod
    def reduce(cls, dc, **kwargs):  # **kwargs required for cooperative signature
        enc_data = cls.Data.encode(dc)
        T, _ = enc_data
        if not isinstance(T, Serializable):  # In most cases it would be fine to always
            T = ParamColl            # replace T by ParamColl, but that would prevent subclasses of ParamColl for using normal scityping hooks.
        # Dataclass.reduce uses `get_type_key(cls)`, which does not work with dynamically created ParamColl subclasses
        return (get_type_key(T), enc_data)

    def __post_init__(self):
        # OK to bypass frozen=True: we are still in initialization

        # object.__setattr__(self, "_inner_len", inner_len)
        dc_replacements = self._dataclass_to_paramcoll()
        for name, val in dc_replacements.items():
            object.__setattr__(self, name, val)
        self._update_lengths()
        # self._validate_inner_len()

        object.__setattr__(self, "_initialized", True)

    def __init_subclass__(cls):
        # Make Expandable types valid for all parameters
        for nm, T in cls.__annotations__.items():
            if nm in ParamColl.__annotations__:
                continue
            cls.__annotations__[nm] = Union[T, Expandable]
        super().__init_subclass__()

    @classmethod
    def __get_validators__(cls):
        yield Dataclass.validate

    # # Define __reduce__ so we can also pickle ParamColls created dynamically in _dataclass_to_paramcoll
    # def __reduce__(self):
    #     # NB: joblib.memory hardcodes use of pickle protocol 3, so we should make sure the returned value is compatible with that protocol
    #     dataclass_type = getattr(self, "__orig_dataclass_type__", type(self))  # __orig_dataclass_type__ is used when a ParamColl was created dynamically to wrap a plain dataclass
    #     return (_create_paramcoll_from_data, 
    #             ({field.name: getattr(self, field.name)
    #               for field in dataclasses.fields(self)
    #               if not field.name.startswith("_")},
    #              dataclass_type)
    #             )

    def _dataclass_to_paramcoll(self):
        """Return a replacement dictionary which converts nested dataclasses
        which have expandable params to nested ParamColls.

        The returned dictionary can be passed to `dataclasses.replace` to create
        the new container ParamColl which expands properly.

        This allows to use `expand` within any dataclass, but currently only
        works one layer deep. So if ``ModelA.Params`` and ``InnerModelA.Params``
        are plain dataclasses, this expands as expected::

            @dataclass
            class BigModel(ParamColl):
                θa = ModelA.Params(
                    λ=expand([1,2,3]))

        But not this::

            @dataclass
            class BigModel(ParamColl):
                θa = ModelA.Params(
                    ηa=InnerModelA.Params(
                        λ=expand([1,2,3])))
        """
        new_vals = {}
        for name, v in self.items():
            if ( is_dataclass(v)
                 and not isinstance(v, ParamColl)
                 and any(isinstance(getattr(v, field.name), Expandable) for field in fields(v))
                 ):
                v_fieldnames = {field.name for field in fields(v)}
                NewParamColl = _get_paramcoll_type(type(v))
                # NewParamColl = _paramcoll_registry.get(type(v))
                # if NewParamColl is None:
                #     ParamColl_fieldnames = {field.name for field in fields(ParamColl)}
                #     if v_fieldnames & ParamColl_fieldnames:
                #         raise RuntimeError(
                #             f"Dataclass {type(v)} contains expandable parameters, "
                #             "but it cannot be converted to ParamColl, because the "
                #             f"following field names conflict: {v_fieldnames & ParamColl_fieldnames}.")
                #     dcp = v.__dataclass_params__
                #     dataclass_params = {param: getattr(dcp, param) for param in dcp.__slots__}
                #     NewParamColl = dataclass(**dataclass_params)(
                #         type("ParamColl", (type(v), ParamColl), {"__orig_dataclass_type__": type(v)}))
                #     _paramcoll_registry[type(v)] = NewParamColl
                kwargs = {_name: getattr(v, _name) for _name in v_fieldnames}
                seed = NOSEED if self.paramseed is NOSEED else (name, self.paramseed)
                new_vals[name] = NewParamColl(paramseed=seed, **kwargs)
                # setattr(self, name, new_v)
        return new_vals

    def _update_lengths(self):
        # NB: We use object.__setattr__ to avoid triggering `self.__setattr__` (and thus recursion errors and other nasties)
        object.__setattr__(self, "_lengths",
            list(chain.from_iterable(
                (v.length,) if isinstance(v, Expandable)
                else v._lengths if isinstance(v, ParamColl)
                else (1,)
                for k, v in self.items()
            ))
        )

        if inf in self._lengths:
            if self.paramseed is NOSEED:
                raise TypeError("A param collection seed is required when some of the "
                                "parameters are specified as random variables.")
            if len(set(_make_rng_key(self.keys()))) != len(self):  # pragma: no cover
                key_seeds = {name: self._name_to_seed(name) for name in self.keys()}
                logger.warning("By some extremely unlikely coincidence, two of your "
                               "parameter names hash to the same integer value. "
                               "To ensure reproducible draws, you may want to change "
                               "one of your argument names. Hash values are:\n"
                               f"{key_seeds}")

    def _validate_inner_len(self):
        if self._inner_len is not None:
            if (set(self._lengths) - {1, inf}):
                logger.warning("Setting the inner length when there are finitely "
                               "expandable parameters has no effect.")
            elif set(self._lengths) == {1}:
                logger.warning("Setting the inner length when there are no "
                               "expandable or random parameters has no effect.")

    # # Rerun __post_init__ every time a parameter is changed
    # def __setattr__(self, attr, val):
    #     super().__setattr__(attr, val)
    #     if self._initialized and attr not in ParamColl.__dataclass_fields__:
    #         self._dataclass_to_paramcoll()
    #         self._update_lengths()
    #         self._validate_inner_len()

    def __str__(self):
        """
        Compared to dataclass’ default __str__:
        - Shorter, more informative display of scipy distributions.
        - Only show `paramseed` if it is set; also, place seed argument at the end.
        """
        argstr = []
        for name, v in self.items():
            argstr.append(f"{name}={v}")
        if self.paramseed is not NOSEED:
            argstr.append(f", paramseed={self.paramseed}")
        return f"{type(self).__qualname__}({', '.join(argstr)})"

    ## Mapping API ##

    def __len__(self):
        return len(self.keys())
                   
    def __iter__(self):
        yield from self.keys()
    
    @classmethod
    def keys(cls):  # TODO: Return something nicer, like a KeysView ?
        return [k for k in cls.__dataclass_fields__
                if not k.startswith("_") and k not in ParamColl.__dataclass_fields__]  # Exclude private fields and those in the base class
    def __getitem__(self, key):
        return getattr(self, key)

    ## Other descriptors ##

    @classmethod
    @property
    def kdims(cls):
        return [cls.dims.get(θname, θname) for θname in cls.keys()]

    # Expansion API
                       
    @property
    def outer_len(self):
        """
        The length of the iterable created by `outer()`.
        
        If outer products are not supported the outer length is `None`.
        (This happens if the parameter iterables are all infinite)
        """
        L = prod(self._lengths)
        if L != inf:
            return L 
        else:
            # If there are only lengths 1 or inf, we return inf
            # Otherwise, the iterator will terminate once the finite expansians are done
            L = prod(l for l in self._lengths if l < inf)
            return L if L != 1 else None
    
    @property
    def inner_len(self):
        """
        The length of the iterable created by `inner()`.

        If inner products are not supported, the inner length is `None`.
        This happens when expandable parameters don't all have the same length.
        """
        diff_lengths = set(self._lengths) - {1}
        if len(diff_lengths - {inf}) > 1:
            return None
        elif diff_lengths == {inf}:
            return inf #if self._inner_len is None else self._inner_len
        elif len(diff_lengths) == 0:
            # There are no parameters to exand
            return 1
        else:
            return next(iter(diff_lengths - {inf}))  # Length is that of the non-infinite iterator

    @inner_len.setter
    def inner_len(self, value):
        # NB: We need to use object.__setattr__ because frozen=True prevents setting attributes directly.
        #     This is safe because even with this setter, one cannot assign to the `inner_len` attribute.
        #     This can only be done with `dataclasses.replace`, or by calling `object.__setattr__` directly.
        if isinstance(value, property):  # NB: The "inner_len" default is actually the `property` object, even when we provide a default in the annotations
            # This is the initial instantiation, and no value was passed for `inner_len`
            object.__setattr__(self, "_inner_len",
                 self.__dataclass_fields__["_inner_len"].default)
        else:
            # Recursively update the length of nested ParamColls
            for nm, v in self.items():
                if isinstance(v, ParamColl) and v.inner_len != value:
                    object.__setattr__(self, nm, replace(v, inner_len=value))
            # Update our own length
            object.__setattr__(self, "_inner_len", value)
            # self._inner_len = value
            # self._validate_inner_len()
    
    def inner(self, start=None, stop=None, step=None):
        """
        If only `start` is provided, it sets the maximum length of the iterator.
        """
        if start is not None and stop is None:
            start, stop = 0, start
        if start is not None or stop is not None or step is not None:
            yield from islice(self.inner(), start, stop, step)
        else:
            for kw in self._get_kw_lst_inner():
                yield type(self)(**kw)
            
    def outer(self, start=None, stop=None, step=None):
        """
        If only `start` is provided, it sets the maximum length of the iterator.
        """
        if start is not None and stop is None:
            start, stop = 0, start
        if start is not None or stop is not None or step is not None:
            yield from islice(self.outer(), start, stop, step)
        else:
            for kw in self._get_kw_lst_outer():
                yield type(self)(**kw)

    ## Private methods ##

    def _get_kw_lst_inner(self):
        inner_len = self.inner_len
        if inner_len is None:
            diff_lengths = set(self._lengths) - {1}
            raise ValueError("Expandable parameters do not all have the same lengths."
                 "`expand` parameters with the following lengths were found:\n"
                 f"{diff_lengths}")
        elif inner_len == 1:
            # There are no parameters to expand  (this implies in particular that there are no random parameters)
            return [{k: v[0] if isinstance(v, Expandable)
                        else v for k, v in self.items()}]
        else:
            kw = {k: v.inner() if isinstance(v, ParamColl)
                     else v.make_iter(seed=(self.paramseed, k), size=inner_len)
                        if isinstance(v, ExpandableRV)
                     else v if isinstance(v, Expandable)
                     else repeat(v)
                  for k,v in self.items()}
            for vlst in zip(*kw.values()):
                yield {k: v for k, v in zip(kw.keys(), vlst)}
    
    def _get_kw_lst_outer(self):
        outer_len = self.outer_len
        if outer_len is None:
            raise ValueError("An 'outer' product of only infinite iterators "
                             "does not really make sense. Use 'inner' to "
                             "create an infinite parameter iterator.")
        kw = {k: list(v.outer()) if isinstance(v, ParamColl)  # outer() returns a generator, product() needs a list
                 else [v.make_iter(seed=(self.paramseed, k), size=outer_len)]  # NB: We don’t want `product`
                    if isinstance(v, ExpandableRV)                        #     to expand the RV iterator
                 else v if isinstance(v, Expandable)
                 else [v]
              for k,v in self.items()}
        for vlst in product(*kw.values()):
            yield {k: next(v) if isinstance(v, Generator) else v   # `Generator` is for the RV iterator
                   for k, v in zip(kw.keys(), vlst)}               # Ostensibly we could support other generators ?


