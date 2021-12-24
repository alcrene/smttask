"""
This module provides a few minimal Task types, which can be used as
placeholders for testing. By design, the use only builtin types, so they
can be imported into any project.
"""

import math
from typing import List
from smttask import MemoizedTask

@MemoizedTask
def MacSine(x: List[float], order: int=8) -> List[float]:
    "Compute sin(x) using a MacLaurin series of the given order."
    #     n : 1  2  3  4  …
    # order : 1  3  5  7  …
    pmax = order - 1 + order%2  # Subtract 1 if order is even
    an = pmax*(pmax-1)
    res = [xi**2/an for xi in x]
    # A modification of Horner's algorithm which evaluates the polynomial and the factorial simultaneously
    for p in range(pmax-2, 1, -2):
        an = p*(p-1)
        res[:] = (xi**2/an * (1-r) for xi, r in zip(x, res))
    res[:] = (xi*(1-r) for xi, r in zip(x, res))
    return res
    
