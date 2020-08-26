import os
import sys
from pathlib import Path
import re
import logging
from mackelab_toolbox.utils import stablehexdigest

from smttask.base import NotComputed

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
    task_digests = ["82a2e90928", "e66f05eb1c", "ac423adb96"]

    # Delete any leftover cache
    for task in tasks:
        task._run_result = NotComputed

    # Run the tasks
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            task.run(cache=False)  # cache=False to test reloading from disk below
            assert caplog.records[-1].msg == "Square_x: no previously saved result was found; running task."

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
            task.run(cache=True)  # cache=True => now saved in memory
            assert caplog.records[-1].msg == "Square_x: loading result of previous run from disk."

    # Run the tasks a 3rd time
    # They should be reloaded from memory
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            task.run()  # cache=False to test
            assert caplog.records[-1].msg == "Square_x: loading memoized result"

def test_multiple_output_task(caplog):

    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    # Define some dummy tasks
    from tasks import SquareAndCube_x
    tasks = [SquareAndCube_x(reason="pytest", x=x, pmax=5) for x in (1.1, 2.1, 5)]
    task_digests = ["495b69f958", "7cdd71a07c", "ebd2b26edb"]

    # Delete any leftover cache
    for task in tasks:
        task._run_result = NotComputed

    # Run the tasks
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            result = task.run(cache=False)  # cache=False to test reloading from disk below
            assert caplog.records[-1].msg == "SquareAndCube_x: no previously saved result was found; running task."
    x=5.
    assert result == (x**2, x**3, (x**4, x**5))
    assert isinstance(result[2], tuple)

    # Assert that the outputs were produced at the expected locations
    assert set(os.listdir(projectroot/"data")) == set(
        ["run_dump", "SquareAndCube_x"])
    for task, digest in zip(tasks, task_digests):
        assert task.hashed_digest == digest
        assert task.unhashed_digests == {}
        assert task.digest == digest
        assert os.path.exists(projectroot/f"data/SquareAndCube_x/{digest}_sqr.json")
        assert os.path.islink(projectroot/f"data/SquareAndCube_x/{digest}_sqr.json")
        assert os.path.exists(projectroot/f"data/SquareAndCube_x/{digest}_cube.json")
        assert os.path.islink(projectroot/f"data/SquareAndCube_x/{digest}_cube.json")
        assert os.path.exists(projectroot/f"data/SquareAndCube_x/{digest}_4.json")
        assert os.path.islink(projectroot/f"data/SquareAndCube_x/{digest}_4.json")
        assert os.path.exists(projectroot/f"data/SquareAndCube_x/{digest}_5.json")
        assert os.path.islink(projectroot/f"data/SquareAndCube_x/{digest}_5.json")
        assert os.path.exists(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_sqr.json")
        assert os.path.isfile(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_sqr.json")
        assert os.path.exists(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_cube.json")
        assert os.path.isfile(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_cube.json")
        assert os.path.exists(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_4.json")
        assert os.path.exists(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_4.json")
        assert os.path.isfile(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_5.json")
        assert os.path.isfile(projectroot/f"data/run_dump/SquareAndCube_x/{digest}_5.json")

    # Run the tasks again
    # They should be reloaded from disk
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            result = task.run(cache=True)  # cache=True => now saved in memory
            assert caplog.records[-1].msg == "SquareAndCube_x: loading result of previous run from disk."
    assert result == (x**2, x**3, (x**4, x**5))

    # Run the tasks a 3rd time
    # They should be reloaded from memory
    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        for task in tasks:
            task.run()  # cache=False to test
            assert caplog.records[-1].msg == "SquareAndCube_x: loading memoized result"

def test_iterative_task(caplog):

    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    # Define some dummy tasks
    from tasks import PowSeq
    tasks = {1: PowSeq(start_n=1, n=1, a=3, p=3, reason="pytest"),
             2: PowSeq(start_n=1, n=2, a=3, p=3, reason="pytest"),
             3: PowSeq(start_n=1, n=3, a=3, p=3, reason="pytest"),
             4: PowSeq(start_n=1, n=4, a=3, p=3, reason="pytest")
             }
    hashed_digest = "a9ecc7a846"

    # Delete any leftover cache
    for task in tasks.values():
        task._run_result = NotComputed

    with caplog.at_level(logging.DEBUG, logger='smttask.smttask'):
        # Compute n=2 from scratch
        n = 2
        result = tasks[n].run(cache=False)
        assert caplog.records[-1].msg == "PowSeq: no previously saved result was found; running task."
        assert result[0] == n
        assert result[1] == 3**3
        for nm in ['a', 'n']:
            assert os.path.exists(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.islink(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.exists(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.isfile(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
        with open(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_a.json") as f:
            a = int(f.read())
        assert a == 3**3

        # Reload n=2 from disk
        n = 2
        result = tasks[n].run(cache=False)
        assert caplog.records[-2].msg == "Found output from a previous run of task 'PowSeq' matching these parameters."
        assert caplog.records[-1].msg == "PowSeq: loading result of previous run from disk."

        # Compute n=4, starting from n=2 reloaded from disk
        n = 4
        result = tasks[n].run(cache=False)
        assert caplog.records[-3].msg == "Found output from a previous run of task 'PowSeq' matching these parameters but with only 2 iterations."
        assert caplog.records[-2].msg == "PowSeq: loading result of previous run from disk."
        assert caplog.records[-1].msg == "PowSeq: continuing from a previous partial result."
        assert result[0] == n
        assert result[1] == ((3**3)**3)**3
        for nm in ['a', 'n']:
            assert os.path.exists(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.islink(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.exists(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.isfile(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
        with open(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_a.json") as f:
            a = int(f.read())
        assert a == ((3**3)**3)**3

        # Reload n=4 from disk
        n = 4
        result = tasks[n].run(cache=False)
        assert caplog.records[-2].msg == "Found output from a previous run of task 'PowSeq' matching these parameters."
        assert caplog.records[-1].msg == "PowSeq: loading result of previous run from disk."

        # Compute n=1 from scratch
        n = 1
        result = tasks[n].run(cache=False)
        assert caplog.records[-1].msg == "PowSeq: no previously saved result was found; running task."
        assert result[0] == n
        assert result[1] == 3
        for nm in ['a', 'n']:
            assert os.path.exists(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.islink(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.exists(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.isfile(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
        with open(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_a.json") as f:
            a = int(f.read())
        assert a == 3

        # Compute n=3, starting from n=2 reloaded from disk
        n = 3
        result = tasks[n].run(cache=False)
        assert caplog.records[-3].msg == "Found output from a previous run of task 'PowSeq' matching these parameters but with only 2 iterations."
        assert caplog.records[-2].msg == "PowSeq: loading result of previous run from disk."
        assert caplog.records[-1].msg == "PowSeq: continuing from a previous partial result."
        assert result[0] == n
        assert result[1] == (3**3)**3
        for nm in ['a', 'n']:
            assert os.path.exists(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.islink(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.exists(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
            assert os.path.isfile(projectroot/f"data/run_dump/PowSeq/{hashed_digest}__n_{n}_{nm}.json")
        with open(projectroot/f"data/PowSeq/{hashed_digest}__n_{n}_a.json") as f:
            a = int(f.read())
        assert a == (3**3)**3
