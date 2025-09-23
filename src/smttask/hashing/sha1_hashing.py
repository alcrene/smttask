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

if sys.version_info >= (3, 9):
    sha1 = partial(hashlib.sha1, usedforsecurity=False)
else:
    sha1 = hashlib.sha1
def stablehash(o: bytes|str) -> '_hashlib.HASH':
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

