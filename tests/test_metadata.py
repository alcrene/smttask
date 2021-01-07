"""
Tests relating to record metadata.

- Setting the 'outcome' attribute.
"""
import os
import sys
from pathlib import Path
os.chdir(Path(__file__).parent)
from utils_for_testing import clean_project

def test_outcome():

    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    # Define a task which takes different outcomes
    from tasks import Polar
    task_succeed = Polar(x=1, y=0, reason="pytest")
    task_undefined = Polar(x=0, y=0, reason="pytest")

    task_succeed.run()
    task_undefined.run()

    from smttask.view import RecordStoreView
    RecordStoreView.default_project_dir = projectpath
    recordlist = RecordStoreView().list
    # Most recent records come first
    assert "undefined" in recordlist[0].outcome
    assert "undefined" not in recordlist[1].outcome
