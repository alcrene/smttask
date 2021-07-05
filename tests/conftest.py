import os
import shutil
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
