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
        x: Array[float,1]
    def _run(self, τ, σ, seed):
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
    def _run(self, x):
        y = np.fft.fft(x)
        return y
