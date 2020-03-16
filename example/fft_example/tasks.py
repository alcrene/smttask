import sys
import numpy as np
from sumatra.parameters import build_parameters, NTParameterSet
ParameterSet = NTParameterSet

from smttask import RecordedTask, InMemoryTask

class GenerateData(InMemoryTask):
    inputs = {'τ': float, 'σ': float, 'seed': int}
    outputs = {'x': np.ndarray}
    def _run(self, τ, σ, seed):
        np.random.seed(seed)
        x = [0]
        for i in range(1000):
            x.append( (1-1/τ)*x[i-1] + np.random.normal(0,σ) )
        return x,

class ProcessData(RecordedTask):
    inputs = {'x': np.ndarray}
    outputs = {'y': np.ndarray}
    def _run(self, x):
        y = np.fft.fft(x)
        return y,
