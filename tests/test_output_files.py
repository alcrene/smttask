import os
import sys
from pathlib import Path
import re
import logging
from mackelab_toolbox.utils import stablehexdigest

os.chdir(Path(__file__).parent)
from utils_for_testing import clean_project

def wip_test_parse_unhashed_params():
    hashdigest = stablehexdigest(91)[:10]
    pname = "stepi"
    ps = [f"{hsh}.json"]
    nm = ""
    for i in [100, 3, 541]:
        ps.append(f"{hashdigest}__{pname}_{i}_{nm}.json")
    re_outfile = f"{re.escape(hashdigest)}__{re.escape(pname)}_(\d*)_([a-zA-Z0-9]*).json$"

    outfiles = {}
    for p in ps:
        m = re.match(re_outfile, p)
        if m is not None:
            assert len(m.groups()) == 2
            itervalue, varname = m.groups()
            outfiles[itervalue] = varname

# Test: Multiple outputs
# Test: IterativeTask

def test_recorded_task(caplog):

    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    # Define some dummy tasks
    from tasks import Square_x
    tasks = [Square_x(x=x, reason="pytest") for x in (1.1, 2.1, 5)]
    task_digests = ["fddae04540", "69443b7735", "84d1f9f300"]

    # Run the tasks
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            task.run(cache=False)  # cache=False to test reloading from disk below
            assert caplog.records[-1].msg == "Square_x: No cached result was found; running task."

    # Assert that the outputs were produced at the expected locations
    assert set(os.listdir(projectroot/"data")) == set(
        ["run_dump", "Square_x"])
    for task, digest in zip(tasks, task_digests):
        assert task.hashed_digest == digest
        assert task.unhashed_digests == {}
        assert task.digest == digest
        assert os.path.exists(projectroot/f"data/Square_x/{digest}_.json")
        assert os.path.islink(projectroot/f"data/Square_x/{digest}_.json")
        assert os.path.exists(projectroot/f"data/run_dump/Square_x/{digest}_.json")
        assert os.path.isfile(projectroot/f"data/run_dump/Square_x/{digest}_.json")

    # Run the tasks again
    # They should be reloaded from disk
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            task.run(cache=True)  # cache=False to test
            assert caplog.records[-1].msg == "Square_x: loading result of previous run from disk."

    # Run the tasks a 3rd time
    # They should be reloaded from memory
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            task.run()  # cache=False to test
            assert caplog.records[-1].msg == "Square_x: loading from in-memory cache"

# def test_iterative_task(caplog):
