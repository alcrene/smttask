from smttask import TaskOutput, RecordedTask
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
    step_n: int
    a: int
    p: int
@RecordedIterativeTask
def PowSeq(step_n:int, a: int, p: int) -> int:
    return step_n+1, a**p, p
