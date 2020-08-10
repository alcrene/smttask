import sys
import numpy as np
from sumatra.parameters import build_parameters, NTParameterSet
ParameterSet = NTParameterSet

from mackelab_toolbox.typing import Array

from smttask import RecordedTask, InMemoryTask

@InMemoryTask
def GenerateData(self, τ: float, σ: float, seed: int) -> Array[float, 1]:
    np.random.seed(seed)
    x = [0]
    for i in range(1000):
        x.append( (1-1/τ)*x[i-1] + np.random.normal(0,σ) )
    return x

@RecordedTask
def ProcessData(self, x: Array[float, 1]) -> Array[complex, 1]:
    y = np.fft.fft(x)
    return y
