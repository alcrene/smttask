import sys
import numpy as np
from sumatra.parameters import build_parameters, NTParameterSet
ParameterSet = NTParameterSet

# from . import config
from smttask import Task

class GenerateTask(Task):
    inputs = {'τ': float, 'σ': float, 'seed': int}
    outputs = ['x']
    def _run(self, τ, σ, seed):
        np.random.seed(seed)
        x = [0]
        for i in range(1000):
            x.append( (1-1/τ)*x[0] + np.random.normal(0,σ) )
        return x

class ProcessTask(Task):
    inputs = {'x': GenerateTask}
    outputs = ['y']
    def _run(self, x):
        y = np.fft.fft(x)
        return y
