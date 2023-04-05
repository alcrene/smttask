import os
import shutil
import logging
from pathlib import Path
from typing import List
from smttask import UnpureMemoizedTask

import pytest

projectroot = Path(__file__).parent/"test_project"
datadir = projectroot/"data/ListDir"

os.chdir(Path(__file__).parent)
from utils_for_testing import clean_project

def test_unpure_digest(caplog):
    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(datadir, exist_ok=True)
    os.chdir(projectroot)

    # Create some dummy files for the task to find
    for c in "abc":
        with open(datadir/f"{c}.dat", 'w') as f:
            f.write(c*10)

    @UnpureMemoizedTask
    def ListDir(root: str) -> List[str]:
        return sorted(os.listdir(root))

    # Create three tasks pointing the the same directory but with different
    # str arguments (otherwise smttask recognizes that they are the identical,
    # and creates only one Task)
    task1 = ListDir(root=str(datadir))
    task2 = ListDir(root=str(datadir/'..') + '/' + '/'.join(datadir.parts[-1:]))
    task3 = ListDir(root=str(datadir/'../..') + '/' + '/'.join(datadir.parts[-2:]))
    assert task1.run() == [f"{c}.dat" for c in "abc"]
    assert task1.digest == task2.digest
    # At this point task3.digest is still undetermined because it has not been,
    # run, but task1 and task2 have ran and fixed their digest

    # Add some dummy files – this should change the output of the
    # UnpureTask, and therefore its digest
    for c in "de":
        with open(datadir/f"{c}.dat", 'w') as f:
            f.write(c*10)

    assert task1.run() == [f"{c}.dat" for c in "abc"]    # Result unchanged
    assert task2.run() == [f"{c}.dat" for c in "abc"]    # Result unchanged
    assert task3.run() == [f"{c}.dat" for c in "abcde"]  # Reflects updated data

    # Tasks created before the new files locked in the old list in their digest
    assert task1.digest != task3.digest
    assert task1.digest == task2.digest  # Still the same

    # Forcing a task to recompute will update its digest
    with caplog.at_level(logging.WARNING):
        task2.run(recompute=True)  # User warning that digest has changed
        assert caplog.records[0].msg.startswith("Digest has changed")
    assert task2.digest == task3.digest
