"""
Basic convenience functions for computing hash digests (representations of 
hash objects).

Note that this is NOT achieved by using Python’s builtin `hash`, which is purposefully
not stable across sessions for security reasons. Instead we use SHA1 hashes, with
the option ``usedforsecurity=False`` when possible.

.. Tip:: The following function can be used to calculate the likelihood
   of a hash collision based on the length of digests::

       def p_coll(N, M):
         '''
         :param N: Number of distinct hashes. For a 6 character hex digest,
            this would be 16**6.
         :param M: Number of hashes we expect to create.
         '''
         logp = np.sum(np.log(N-np.arange(M))) - M*np.log(N)
         return 1-np.exp(logp)

"""

import sys
import hashlib
from functools import partial
from typing import Union

if sys.version_info >= (3, 9):
    sha1 = partial(hashlib.sha1, usedforsecurity=False)
else:
    sha1 = hashlib.sha1
def stablehash(o: Union[bytes,str]) -> '_hashlib.HASH':
    """
    The builtin `hash` is not stable across sessions for security reasons.
    This `stablehash` can be used when consistency of a hash is required, e.g.
    for on-disk caches.

    For obtaining a usable digest, see the convenience functions
    `stablehexdigest`, `stablebytesdigest` and `stableintdigest`.

    .. Note:: These functions are not meant for cryptographic use; indeed when
       possible we pass the ``usedforsecurity=False``

    Returns
    -------
    HASH object
    """
    if isinstance(o, str): o = o.encode('utf8')
    return sha1(o)
def stablehexdigest(o) -> str:
    """
    Returns
    -------
    str
    """
    return stablehash(o).hexdigest()
def stablebytesdigest(o) -> bytes:
    """
    Returns
    -------
    bytes
    """
    return stablehash(o).digest()
def stableintdigest(o, byte_len=4) -> int:
    """
    Suitable as the return value of a `__hash__` method.

    .. Note:: Although this method is provided, note that the purpose of a
       digest (a unique fingerprint) is not the same as the intended use of
       the `__hash__` magic method (fast hash tables, in particular for
       dictionaries). In the latter case, a certain degree of hash collisions
       is in fact desired, since that is required for the most efficient tables.
       Because this function uses SHA1 to obtain almost surely unique digests,
       it is much slower than typical `__hash__` implementations. This can
       become noticeable if it is involved in a lot of dictionary lookups.

    Parameters
    ----------
    o : object to hash (see `stablehash`)
    byte_len : int, Optional (default: 4)
        Number of bytes to keep from the hash. A value of `b` provides at most
        `8**b` bits of entropy. With `b=4`, this is 4096 bits and 10 digit
        integers.

    Returns
    -------
    int
    """
    return int.from_bytes(stablebytesdigest(o)[:byte_len], 'little')


######################## Normalization to bytes ################################
# The code below allows implements a generic _tobytes function, which tries    #
# its best to convert a wide variety of inputs into a bytes sequence which can #
# be hashed. This functionality is currently only really used by               #
# workflows.SeedGenerator (and there, limited to `_normalize_entropy);         #
# the rest of smttask only needs support for hashing  str and int.             #
# There is also the possibility that we could reimplement the                  #
# functionality below using Serializable.reduce; this would avoid the need for #
# all the type cases. OTOH, Serializable.reduce needs to allow bi-directional  #
# conversion, whereas this is not an issue, making the logic ultimately        #
# simpler and faster.                                                          #
#                                                                              #
# All this to say that whether we keep _tobytes in this module going forward   #
# is still undetermined.                                                       #
######################## ###################### ################################

from enum import Enum
from typing import Any

def universal_stablehash(o: Any) -> '_hashlib.HASH':
    """
    Like `stablehash`, but with much more flexibility in the type of `o`.
    The value is passed through `_tobytes` to normalize it to bytes before
    hashing, allowing most values to be used to produce a unique, stable hash.
    See `_tobytes` for details.
    Additional value->bytes rules for different types can be added to the
    `_byte_converters` dictionary.
    """
    return stablehash(_tobytes(o))
def universal_stableintdigest(o: Any, byte_len=4) -> int:
    return stableintdigest(_tobytes(o), byte_len)

# Extra functions to converting values to bytes specialized to specific types
# This is the mechanism to use to add support for types outside your own control
# (i.e. for which it is not possible to add a __bytes__ method).
class TypeDict:
    """
    A dictionary using types as keys. A key will match any of its subclasses;
    if there are multiple possible matches, the earliest in the dictionary
    takes precedence.
    """
    def __getitem__(self, key):
        if not isinstance(key, type):
            raise TypeError("TypeDict keys must be types")
        for k, v in self.items():
            if issubclass(key, k):
                return v
        raise KeyError(f"Type {key} is not a subclass of any of this "
                       "TypeDict's type keys.")

    def __setitem__(self, key, value):
        if not isinstance(key, type):
            raise TypeError("TypeDict keys must be types")
        return super().__setitem__(key, value)

    def __contains__(self, key):
        return (isinstance(key, type) and
                any(issubclass(key, k) for k in self))

_byte_converters = TypeDict()

def _tobytes(o) -> bytes:
    """
    Utility function for converting an object to bytes. This is used to
    generate unique, stable digests for arbitrary sequences of objects:

    1. Different inputs should, with high probability, return different byte
       sequences.
    2. The same inputs should always return the same byte sequence, even when
       executed in a new session (in order to satisfy the 'stable' description).
       Note that this precludes using an object's `id`, which is sometimes
       how `hash` is implemented.
       It also precludes using `hash` or `__hash__`, since that function is
       randomly salted for each new Python session.
    3. It is NOT necessary for the input to be reconstructable from
       the returned bytes.

    ..Note:: To avoid overly complicated byte sequences, the distinction
       guarantee is not preserved across types. So `_tobytes(b"A")`,
       `_tobytes("A")` and `_tobytes(65)` all return `b'A'`.
       So multiple inputs can return the same byte sequence, as long as they
       are unlikely to be used in the same location to mean different things.

    **Supported types**
    - None
    - bytes
    - str
    - int
    - float
    - Enum
    - type
    - dataclasses, as long as their fields are supported types.
      + NOTE: At present we allow hashing both frozen and non-frozen dataclasses
    - Any object implementing a ``__bytes__`` method
    - Mapping
    - Sequence
    - Collection
    - Any object for which `bytes(o)` does not raise an exception

    TODO?: Restrict to immutable objects ?

    Raises
    ------
    TypeError:
        - If `o` is a consumable Iterable.
        - If `o` is of a type for which `_to_bytes` is not implemented.
    """
    # byte converters for specific types
    if o is None:
        # TODO: Would another value more appropriately represent None ? E.g. \x00 ?
        return b""
    elif isinstance(o, bytes):
        return o
    elif isinstance(o, str):
        return o.encode('utf8')
    elif isinstance(o, int):
        l = ((o + (o<0)).bit_length() + 8) // 8  # Based on https://stackoverflow.com/a/54141411
        return o.to_bytes(length=l, byteorder='little', signed=True)
    elif isinstance(o, float):
        return o.hex().encode('utf8')
    elif isinstance(o, Enum):
        return _tobytes(o.value)
    elif isinstance(o, type):
        return _tobytes(f"{o.__module__}.{o.__qualname__}")
    elif is_dataclass(o):
        # DEVNOTE: To restrict this to immutable dataclasses, check `o.__dataclass_params__.frozen`
        return _tobytes(tuple((f.name, getattr(o, f.name)) for f in fields(o)))
    # Generic byte encoders. These methods may not be ideal for each type, or
    # even work at all, so we first check if the type provides a __bytes__ method.
    elif hasattr(o, '__bytes__'):
        return bytes(o)
    elif type(o) in _byte_converters:
        return _byte_converters[type(o)](o)
    elif isinstance(o, Mapping) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(k) + _tobytes(v) for k,v in o.items())
    elif isinstance(o, Sequence) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in o)
    elif isinstance(o, Collection) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in sorted(o))
    elif isinstance(o, Iterable) and not isinstance(o, terminating_types):
        raise ValueError("Cannot compute a stable hash for a consumable Iterable.")
    else:
        try:
            return bytes(o)
        except TypeError:
            # As an ultimate fallback, attempt to use the same decomposition
            # that pickle would
            try:
                state = o.__getstate__()
            except Exception:
                raise TypeError("smttask.hashing._tobytes does not know how "
                                f"to convert values of type {type(o)} to bytes. "
                                "One way to solve this is may be to add a "
                                "`__bytes__` method to that type. If that is "
                                "not possible, you may also add a converter to "
                                "smttask.hashing._byte_converters.")
            else:
                return _tobytes(state)