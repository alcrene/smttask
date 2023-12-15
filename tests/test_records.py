import os
import sys
from pathlib import Path
import re
import logging
from smttask.hashing import stablehexdigest

os.chdir(Path(__file__).parent)
from utils_for_testing import clean_project

def test_rebuild_input_datastore():
    # TODO: Test that more recent tasks overwrite older ones
    # TODO: Explicitely test behaviour when record output no longer exists (e.g. was deleted)
    from smttask import config
    from smttask.view import RecordStoreView
    from smttask.utils import compute_input_symlinks
    import shutil

    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)
    from tasks import Square_x, SquareAndCube_x

    # Create and run some tasks. Include tasks with both single & multiple outputs
    tasks = [Square_x(x=x, reason="pytest") for x in (1.1, 2.1, 5)]
    tasks += [SquareAndCube_x(x=x, pmax=5, reason="pytest") for x in (1.1, 2.1, 5)]
    for task in tasks:
        task.run()

    # Assert that the outputs were produced at the expected locations
    outroot = Path(config.project.data_store.root)
    inroot = Path(config.project.input_datastore.root)
    for task in tasks:
        for relpath in task.relative_outputpaths.values():
            assert (outroot/relpath).exists()
            assert (outroot/relpath).is_file()
            assert not (outroot/relpath).is_symlink()
            assert (inroot/relpath).exists()
            assert (inroot/relpath).is_symlink()
            assert (outroot/relpath).resolve() == (inroot/relpath).resolve()

    # Delete the input data store, but leave the output data store intact
    for path in inroot.iterdir():
        # NOTE: With Python 3.9+, we could use Path.is_relative_to
        #       This would avoid this clunky pattern where raising an exception
        #       is the normal code path
        try:
            outroot.relative_to(path)
        except ValueError:
            # path not part of path to outroot - delete
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    # Assert that the links in the input datastore no longer exist
    for task in tasks:
        for relpath in task.relative_outputpaths.values():
            assert (outroot/relpath).exists()
            assert (outroot/relpath).is_file()
            assert not (inroot/relpath).exists()

    # Rebuild the input data store
    recordlist = RecordStoreView()
    recordlist.rebuild_input_datastore(compute_input_symlinks)

    # Assert that the correct links were added back to the input data store
    outroot = Path(config.project.data_store.root)
    inroot = Path(config.project.input_datastore.root)
    for task in tasks:
        for relpath in task.relative_outputpaths.values():
            assert (outroot/relpath).exists()
            assert (outroot/relpath).is_file()
            assert (inroot/relpath).exists()
            assert (inroot/relpath).is_symlink()
            assert (outroot/relpath).resolve() == (inroot/relpath).resolve()

if __name__ == "__main__":
    test_rebuild_input_datastore()
