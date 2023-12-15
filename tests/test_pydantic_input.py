import os
import sys
from pathlib import Path
import re
import logging
from smttask.hashing import stablehexdigest
import smttask

os.chdir(Path(__file__).parent)
from utils_for_testing import clean_project

def wip_test_pydantic_input():
    """
    A failing test for Issue#2 :
    "When serializing a Pydantic model, use its own json encoders."
    """
    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    # Define some dummy tasks
    from tasks import Counter, CountingWithSerializableObject, CountingWithDataclass
    py_count = Counter(counter=3)

    task = CountingWithSerializableObject(n=10, pobj=py_count)
    task2 = smttask.Task.from_desc(task.desc.json())
    assert task2.run() == 13

    task = CountingWithDataclass(n=10, pobj=py_count)
    task2 = smttask.Task.from_desc(task.desc.json())
    assert task2.run() == 13
