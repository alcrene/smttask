"""
The tasks here are the same as in run.py, with the action of decorators done
explicitely.

The purpose of this module is two-fold:

- To allow testing the Task class without the decorators.
- To document what the decorators does.
"""


import sys
import numpy as np
from sumatra.parameters import build_parameters, NTParameterSet
ParameterSet = NTParameterSet

from smttask import RecordedTask, InMemoryTask

from mackelab_toolbox.typing import Array

from smttask.base import Task, TaskInputs, TaskOutputs
from typing import Union

class GenerateData(InMemoryTask):
    class Inputs(TaskInputs):
        τ: Union[Task,float]
        σ: Union[Task,float]
        seed: int
    class Outputs(TaskOutputs):
        "": Array[float,1]
    @staticmethod
    def _run(τ, σ, seed):
        np.random.seed(seed)
        x = [0]
        for i in range(1000):
            x.append( (1-1/τ)*x[i-1] + np.random.normal(0,σ) )
        return x

class ProcessData(RecordedTask):
    class Inputs(TaskInputs):
        x: Union[Task,Array[float,1]]
    class Outputs(TaskOutputs):
        y: Array[complex,1]
        S: Array[float, 1]
    @staticmethod
    def _run(x):
        y = np.fft.fft(x)
        S = abs(y*y.conj())
        return y, S
