import os
import sys
import shutil
import pytest
from pathlib import Path
from subprocess import run
import sumatra.commands

# os.chdir("tests")
# project_root = Path(os.getcwd())/"test_project"
def pytest_sessionstart(session):
    """
    Delete the old Sumatra project and create a new one at the start of a
    test session.
    """
    os.chdir(Path(__file__).parent)
    project_root = Path(__file__).parent/"test_project"
    if os.path.exists(project_root):
        shutil.rmtree(project_root)
    os.mkdir(project_root)
    shutil.copy("tasks_for_testing.py", project_root/"tasks.py")
    shutil.copy("types_for_testing.py", project_root/"data_types.py")
    os.chdir(project_root)
    run(["git", "init"])
    with open(".gitignore", 'w') as f:
        f.write("run\n")
    run(["git", "add", "tasks.py"])
    run(["git", "commit", "-m", "Add tasks."])

    argv = ["smt", "init", "test_project",
            "--datapath", str(project_root/"data/run_dump"),
            "--input", str(project_root/"data"),
            "--repository", str(project_root)]
    # The arguments above are meant for `run()`. However, because `run`
    # doesn't execute in a virtual environment, we call the sumatra command
    # directly (hence why we drop the first to elements)
    sumatra.commands.init(argv[2:])

# TODO: If we could move more setup into `create_project` a fixture which is
#    scoped to run only once for the whole session, we could speed up execution

@pytest.fixture(autouse=True)
def projectroot() -> Path:
    """Add the test project to sys.path and cd into it.
    Returns the path to the runtime directory (project root)."""
    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if projectpath not in sys.path:
        sys.path.insert(0, projectpath)
    os.chdir(projectroot)
    return projectroot

@pytest.fixture()
def clean(projectroot) -> Path:
    """Same as `projectroot`, but also ensure that the contents is clean.
    This is done by deleting everything under `projectroot`, so the overhead
    for this fixture is larger.
    """
    os.chdir(projectroot.parent)
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

## clean_project ##

import os
import shutil

ignored_names = ['__pycache__', '.smt', '.git', '.gitignore',
                 'tasks.py', 'data_types.py', 'run_dump']

def clean_project(projectroot):
    """
    All test projects store their input data at 'projectroot/data', and
    their output data at 'projectroot/data/run_dump'.
    """
    # Remove everything in the data directory, including directories
    with os.scandir(projectroot) as it:
        for entry in it:
            if entry.is_file() and entry.name not in ignored_names:
                os.remove(entry.path)
            elif entry.is_dir() and entry.name not in ignored_names:
                shutil.rmtree(entry.path)
    # os.makedirs(projectroot/"data", exist_ok=True)
