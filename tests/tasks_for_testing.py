"""
Each time pytest is run, this file is copied to test_project/tasks.py.
"""

from smttask import TaskOutput, RecordedTask, RecordedIterativeTask, MemoizedTask
from smttask.typing import separate_outputs

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

class PowSeqOutput(TaskOutput):
    n: int  # Required at least for now: no way to refer to the iteration step read from file name
    a: int
@RecordedIterativeTask('n', map={'n': 'start_n',
                                 'a': 'a'})
def PowSeq(start_n: int, n:int, a: int, p: int) -> PowSeqOutput:
    for n in range(start_n+1, n+1):
        a = a**p
    return n, a


from smttask.typing import PureFunction, PurePartialFunction
@MemoizedTask
def AddPureFunctions(
    f1: PureFunction,
    f2: PureFunction[[int], float],
    g1: PurePartialFunction,
    g2: PurePartialFunction[[int], float]
) -> PureFunction:
    def h(x, p):
        return f1(x) + f2(p) + g1(x) + g2(p)
    return h
