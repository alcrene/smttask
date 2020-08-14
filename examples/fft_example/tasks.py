import sys
import numpy as np
from sumatra.parameters import build_parameters, NTParameterSet
ParameterSet = NTParameterSet

from mackelab_toolbox.typing import Array

from smttask import RecordedTask, InMemoryTask, TaskOutputs

@InMemoryTask
def GenerateData(τ: float, σ: float, seed: int) -> Array[float, 1]:
    np.random.seed(seed)
    x = [0]
    for i in range(1000):
        x.append( (1-1/τ)*x[i-1] + np.random.normal(0,σ) )
    return x

# Constructing the output type explicitely allows to define multiple results,
# and assign a name to each
# NOTE: At present, it is assumed that the TaskOutputs subclass is defined
# in the same module as the task function.
class ProcessDataOutput(TaskOutputs):
    y: Array[complex, 1]
    S: Array[float, 1]
@RecordedTask
def ProcessData(x: Array[float, 1]) -> ProcessDataOutput:
    y = np.fft.fft(x)
    S = abs(y*y.conj())
    return y, S