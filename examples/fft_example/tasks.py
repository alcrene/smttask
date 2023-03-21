import sys
import numpy as np

from scityping.numpy import Array

from smttask import RecordedTask, MemoizedTask, TaskOutput

@MemoizedTask
def GenerateData(τ: float, σ: float, seed: int) -> Array[float, 1]:
    np.random.seed(seed)
    x = [0]
    for i in range(1000):
        x.append( (1-1/τ)*x[i-1] + np.random.normal(0,σ) )
    return x

# Constructing the output type explicitely allows to define multiple results,
# and assign a name to each
# NOTE: At present, it is assumed that the TaskOutput subclass is defined
# in the same module as the task function.
class ProcessDataOutput(TaskOutput):
    y: Array[complex, 1]
    S: Array[float, 1]
@RecordedTask
def ProcessData(x: Array[float, 1]) -> ProcessDataOutput:
    y = np.fft.fft(x)
    S = abs(y*y.conj())
    return y, S
