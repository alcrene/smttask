"""
Each time pytest is run, this file is copied to test_project/tasks.py.
"""

import os
import smttask
from smttask import TaskOutput, RecordedTask, RecordedIterativeTask, MemoizedTask
from smttask.typing import separate_outputs
from pydantic import confloat
import math
try:
    from tqdm.auto import tqdm
except (NameError, ImportError):
    def tqdm(it, **kwargs):
        return it
else:
    import tqdm.notebook
    if tqdm.notebook.IProgress is None:
        # Jupyter and/or ipywidgets need to be updated. Fall back to plain tqdm
        from tqdm import tqdm

@RecordedTask
def Square_x(x: float) -> float:
    return x**2

from pydantic.types import conlist
class SquareCubeOutput(TaskOutput):
    sqr : float
    cube: float
    morepower: separate_outputs(
        float, lambda pmax: [str(p) for p in range(4, pmax+1)])
@RecordedTask
def SquareAndCube_x(x: float, pmax: int) -> SquareCubeOutput:
    return x**2, x**3, [x**p for p in range(4, pmax+1)]

# PowSeq goes from 'instantaneous' to 'breaks the kernel' around n=24, a=3, p=2,
# so maybe isn't the best example
class PowSeqOutput(TaskOutput):
    n: int  # Required at least for now: no way to refer to the iteration step read from file name
    a: int
@RecordedIterativeTask('n', map={'n': 'start_n',
                                 'a': 'a'})
def PowSeq(start_n: int, n:int, a: int, p: int) -> PowSeqOutput:
    for n in tqdm(range(start_n+1, n+1), position=smttask.config.process_number):
        a = a**p
    return n, a

# Perhaps a better example is a sequence which it doesn't explode: approximate orbits
class OrbitOutput(TaskOutput):
    n: int
    x: float
    y: float
@RecordedIterativeTask('n', map={'n': 'start_n',
                                 'x': 'x',
                                 'y': 'y'})
def Orbit(start_n: int, n: int, x: float, y: float) -> OrbitOutput:
    r = (x**2 + y**2)**(0.5)
    for n in tqdm(range(start_n+1, n+1), position=smttask.config.process_number):
        x, y = x-y, y+x   # Add 90° rotated vector
        r2 = (x**2 + y**2)**(0.5)
        x = x / r2*r      # Normalize so that it stays on the original circle
        y = y / r2*r
    return n, x, y

from smttask.typing import PureFunction, PartialPureFunction
@MemoizedTask
def AddPureFunctions(
    f1: PureFunction,
    f2: PureFunction[[int], float],
    g1: PartialPureFunction,
    g2: PartialPureFunction[[int], float],
    f3: PureFunction  # For testing CompositePureFunction
) -> PureFunction:
    def h(x, p):
        # original g2 was g2(x, p); x was bound by keyword, so we need to pass p as kwarg as well
        return f1(x) + f2(p) + g1(x) + g2(p=p) + f3(x)
    return h

@RecordedTask
def Failing(x: float) -> float:
    return x / 0

# A fragile task which may fail if x=y=0
class PolarOutput(TaskOutput):
    r: confloat(ge=0)
    θ: confloat(ge=-math.pi, le=math.pi)
@RecordedTask
def Polar(x: float, y: float) -> PolarOutput:
    if x == y == 0:
        outcome = "The mapping of (0,0) to polar coordinates is undefined."
    else:
        outcome = "Returned polar coordinates"
    return {'r':math.sqrt(x**2 + y**2), 'θ':math.atan2(y, x),
            'outcome':outcome}
