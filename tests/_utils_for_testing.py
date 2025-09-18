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
