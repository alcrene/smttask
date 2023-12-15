import os
import sys
from pathlib import Path
import re
import logging
# from sumatra.projects import load_project
import smttask
from smttask.hashing import stablehexdigest

os.chdir(Path(__file__).parent)
from utils_for_testing import clean_project

def wip_test_parse_unhashed_params():
    hashdigest = stablehexdigest(91)[:10]
    pname = "stepi"
    ps = [f"{hsh}.json"]
    nm = ""
    for i in [100, 3, 541]:
        ps.append(f"{hashdigest}__{pname}_{i}_{nm}.json")
    re_outfile = f"{re.escape(hashdigest)}__{re.escape(pname)}_(\\d*)_([a-zA-Z0-9]*).json$"

    outfiles = {}
    for p in ps:
        m = re.match(re_outfile, p)
        if m is not None:
            assert len(m.groups()) == 2
            itervalue, varname = m.groups()
            outfiles[itervalue] = varname

def test_recorded_task(caplog):
    # OPTIMIZATION/TIMING: Running 3 tasks takes ~30 seconds
    #   (everything else in this test takes < 100ms)

    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    # Define some dummy tasks
    from smttask import Task
    from tasks import Square_x
    tasks = [Square_x(x=x, reason="pytest") for x in (1.1, 2.1, 5)]
    task_digests = ['7ad6c9eb99', '2eb601a664', '1a247b2f98']

    # Delete any leftover cache
    for task in tasks:
        task.clear()

    # Run the tasks
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        caplog.clear()
        for task in tasks:
            task.run(cache=False)  # cache=False to test reloading from disk below
            assert caplog.records[0].msg == "No previously saved result was found; running task."

    # Assert that the outputs were produced at the expected locations
    assert set(os.listdir(projectroot/"data")) == {"run_dump", "Square_x"}
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
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        for task in tasks:
            caplog.clear()
            task.run(cache=True)  # cache=True => now saved in memory
            assert caplog.records[0].msg == "Loading result of previous run from disk."

    # Run the tasks a 3rd time
    # They should be reloaded from memory
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        for task in tasks:
            caplog.clear()
            task.run()  # cache=False to test
            assert caplog.records[0].msg == "Loading memoized result."

    # Assert that status tags are saved
    # TODO: Test every possible tag value. Will require tasks which fail after each update of `status`
    # project = load_project()   # NB: If this line fails, it may indicate that the recordstore is not being closed after access
    project = smttask.config.project
    for label in project.get_labels():
        record = project.get_record(label)
        assert record.tags == {'_finished_'}

    # Test deserialization
    new_task = Task.from_desc(task.desc.json())
    # Task recognizes that it is being constructed with the same arguments, and simply returns the preexisting instance
    assert new_task is task

def test_class_task(caplog):
    "Test that decorator can also be applied to callable classes."

    # TODO: With a class and fixture, we should be able to reuse the code
    #       from test_recorded_task

    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Define some dummy tasks
    from smttask import Task
    from tasks import ClassTask
    tasks = [ClassTask(x=x, a=a, reason="pytest")
             for x, a in [(1.1, -2), (2.1, 0), (5, 3)]]
    print([t.digest for t in tasks])
    task_digests = ['be99ccb486', '08373073fb', 'e3727e4d17']

    # Delete any leftover cache
    for task in tasks:
        task.clear()

    # Run the tasks
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        caplog.clear()
        for task in tasks:
            task.run(cache=False)  # cache=False to test reloading from disk below
            assert caplog.records[0].msg == "No previously saved result was found; running task."

    # Assert that the outputs were produced at the expected locations
    # NB: We could test for equality if we used `clean_project` to remove task results for other tests
    assert set(os.listdir(projectroot/"data")) >= {"run_dump", "ClassTask"}
    for task, digest in zip(tasks, task_digests):
        assert task.hashed_digest == digest
        assert task.unhashed_digests == {}
        assert task.digest == digest
        assert os.path.exists(projectroot/f"data/ClassTask/{digest}_.json")
        assert os.path.islink(projectroot/f"data/ClassTask/{digest}_.json")
        assert os.path.exists(projectroot/f"data/run_dump/ClassTask/{digest}_.json")
        assert os.path.isfile(projectroot/f"data/run_dump/ClassTask/{digest}_.json")

    # Run the tasks again
    # They should be reloaded from disk
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        for task in tasks:
            caplog.clear()
            task.run(cache=True)  # cache=True => now saved in memory
            assert caplog.records[0].msg == "Loading result of previous run from disk."

    # Run the tasks a 3rd time
    # They should be reloaded from memory
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        for task in tasks:
            caplog.clear()
            task.run()  # cache=False to test
            assert caplog.records[0].msg == "Loading memoized result."

    # Test deserialization
    new_task = Task.from_desc(task.desc.json())
    # Task recognizes that it is being constructed with the same arguments, and simply returns the preexisting instance
    assert new_task is task

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
    task_digests = ['860feb44ee', '4b754dd53d', 'fcde864238']

    # Delete any leftover cache
    for task in tasks:
        task.clear()

    # Run the tasks
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        for task in tasks:
            caplog.clear()
            result = task.run(cache=False)  # cache=False to test reloading from disk below
            assert caplog.records[0].msg == "No previously saved result was found; running task."
    x=5.
    assert result == (x**2, x**3, (x**4, x**5))
    assert isinstance(result[2], tuple)

    # Assert that the outputs were produced at the expected locations
    assert set(os.listdir(projectroot/"data")) == {"run_dump", "SquareAndCube_x"}
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
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        for task in tasks:
            caplog.clear()
            result = task.run(cache=True)  # cache=True => now saved in memory
            assert caplog.records[0].msg == "Loading result of previous run from disk."
    assert result == (x**2, x**3, (x**4, x**5))

    # Run the tasks a 3rd time
    # They should be reloaded from memory
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        for task in tasks:
            caplog.clear()
            task.run()  # cache=False to test
            assert caplog.records[0].msg == "Loading memoized result."

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
    from smttask import Task
    from tasks import PowSeq
    tasks = {1: PowSeq(start_n=1, n=1, a=3, p=3, reason="pytest"),
             2: PowSeq(start_n=1, n=2, a=3, p=3, reason="pytest"),
             3: PowSeq(start_n=1, n=3, a=3, p=3, reason="pytest"),
             4: PowSeq(start_n=1, n=4, a=3, p=3, reason="pytest")
             }
    hashed_digest = "b2c7aa835f"

    # Delete any leftover cache
    for task in tasks.values():
        task.clear()

    with caplog.at_level(logging.DEBUG, logger=tasks[1].logger.name):
        caplog.clear()
        # Compute n=2 from scratch
        n = 2
        result = tasks[n].run(cache=False)
        assert caplog.records[0].msg == "No previously saved result was found; running task."
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
        caplog.clear()
        n = 2
        result = tasks[n].run(cache=False)
        assert caplog.records[0].msg == "Found output from a previous run matching these parameters."
        assert caplog.records[1].msg == "Loading result of previous run from disk."

        # Compute n=4, starting from n=2 reloaded from disk
        caplog.clear()
        n = 4
        result = tasks[n].run(cache=False)
        assert caplog.records[0].msg == "Found output from a previous run matching these parameters but with only 2 iterations."
        assert caplog.records[1].msg == "Loading result of previous run from disk."
        assert caplog.records[2].msg == "Continuing from a previous partial result."
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
        caplog.clear()
        n = 4
        result = tasks[n].run(cache=False)
        assert caplog.records[0].msg == "Found output from a previous run matching these parameters."
        assert caplog.records[1].msg == "Loading result of previous run from disk."

        # Compute n=1 from scratch
        caplog.clear()
        n = 1
        result = tasks[n].run(cache=False)
        assert caplog.records[0].msg == "No previously saved result was found; running task."
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
        caplog.clear()
        n = 3
        result = tasks[n].run(cache=False)
        assert caplog.records[0].msg == "Found output from a previous run matching these parameters but with only 2 iterations."
        assert caplog.records[1].msg == "Loading result of previous run from disk."
        assert caplog.records[2].msg == "Continuing from a previous partial result."
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
            
    # Test deserialization
    new_task = Task.from_desc(task.desc.json())
    # Task recognizes that it is being constructed with the same arguments, and simply returns the preexisting instance
    assert new_task is task

def test_create_task(caplog):
    
    projectroot = Path(__file__).parent/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)

    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    from smttask import Create, Task, NotComputed, config
    from data_types import Point
    import scityping
    scityping.config.safe_packages.add("data_types")
    
    # Define some dummy tasks
    # Note that we can create `Create` tasks directly in the run file
    tasks = [Create(Point)(x=i*0.3, y=1-i*0.3) for i in range(3)]
    task_digests = ['7cfede3723', '41b458fd1a', '54b726f509']#['ed85744d5a', '127d88b74d', '97fa31904f']
    
    # Delete any leftover cache
    for task in tasks:
        task.clear()

    # Run the tasks
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        caplog.clear()
        for i, (task, digest) in enumerate(zip(tasks, task_digests)):
            point = task.run(cache=False)  # cache=False to test reloading from disk below
            assert task.hashed_digest == digest
            assert task.unhashed_digests == {}
            assert task.digest == digest
            assert caplog.records[0].msg == "Running task in memory."
            assert point.x == i*0.3
            assert point.y == 1-i*0.3
            assert task._run_result is NotComputed  # Not cached
            
    # Run tasks again
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        caplog.clear()
        for i, task in enumerate(tasks):
            point = task.run(cache=True)
            assert caplog.records[0].msg == "Running task in memory."
            assert len(task._run_result) == 1
            assert task._run_result.obj is point
            
    # Run tasks a 3rd time
    # They should be reloaded from memory
    with caplog.at_level(logging.DEBUG, logger=tasks[0].logger.name):
        caplog.clear()
        for i, task in enumerate(tasks):
            point = task.run(cache=True)
            assert caplog.records[0].msg == "Loading memoized result."
            assert len(task._run_result) == 1
            assert task._run_result.obj is point
            
    # Test deserialization
    new_task = Task.from_desc(task.desc.json())
    # Task recognizes that it is being constructed with the same arguments, and simply returns the preexisting instance
    assert new_task is task
