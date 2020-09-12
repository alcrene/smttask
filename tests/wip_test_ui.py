
from pathlib import Path
import functools
import sys
import os
import smttask
testroot = Path(smttask.__file__).parent.parent/"tests"
os.chdir(testroot)
from utils_for_testing import clean_project

def test_ui_run_mp():
    # Add test directory to import search path
    # Use PYTHONPATH environment variable, because that will be seen by
    # subprocesses spawned by multiprocessing.Pool
    projectroot = testroot/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        # sys.path.insert(0, projectpath)
        PYTHONPATH = os.getenv("PYTHONPATH", "")
        if PYTHONPATH:
            PYTHONPATH = os.pathsep + PYTHONPATH
        os.environ["PYTHONPATH"] = projectpath + PYTHONPATH
    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    from tasks import Square_x, Failing, Orbit
    os.makedirs('tasklist', exist_ok=True)

    Square_x(x=3, reason="smttask run test").save('tasklist')
    Square_x(x=4, reason="smttask run test").save('tasklist')
    # %bash
    # Running with --no-record does not create a record and does not delete the taskdesc file
    PYTHONPATH=f"{str(projectpath)}" smttask run tasklist/Square_x__661eaf10bc.taskdesc.json --no-record
    assert [no saved record]
    assert os.path.exists(projectpath/"tasklist/Square_x__661eaf10bc.taskdesc.json")
    # Running with leave creates a record and does not delete the taskdesc file
    PYTHONPATH=f"{str(projectpath)}" smttask run tasklist/Square_x__661eaf10bc.taskdesc.json --leave
    assert [new record]
    assert os.path.exists(projectpath/"tasklist/Square_x__661eaf10bc.taskdesc.json")
    # Running with defaults creates a record and deletes the taskdesc file
    PYTHONPATH=f"{str(projectpath)}" smttask run tasklist/Square_x__661eaf10bc.taskdesc.json
    assert [new record]
    assert not os.path.exists(projectpath/"tasklist/Square_x__661eaf10bc.taskdesc.json")

    # Add tasks back
    Square_x(x=3, reason="smttask run test").save('tasklist')
    Square_x(x=4, reason="smttask run test").save('tasklist')

    # Both tasks are executed
    PYTHONPATH="$HOME/usr/local/python/smttask/tests/test_project" smttask run tasklist/Square_x__*.taskdesc.json -v
    assert [2 new records]
    assert not os.path.exists(projectpath/"tasklist/Square_x__661eaf10bc.taskdesc.json")
    assert not os.path.exists(projectpath/"tasklist/Square_x__e8d9eedb55.taskdesc.json")

    # Tasks using tqdm work and use SMTTASK_PROCESS_NUM to distinguish progress bars
    # => values around n=30000000 take about 15s to run, appropriate for testing KeyboardInterrupt
    Orbit(start_n=0, n=30000213, x=1.2, y=2, reason="smttask run test (tqdm)").save('tasklist')
    Orbit(start_n=0, n=30000215, x=3.3, y=40.1, reason="smttask run test (tqdm)").save('tasklist')
    Orbit(start_n=0, n=30000217, x=3.3, y=40.1, reason="smttask run test (tqdm)").save('tasklist')
    Orbit(start_n=0, n=30000219, x=3.3, y=40.1, reason="smttask run test (tqdm)").save('tasklist')
    PYTHONPATH="$HOME/usr/local/python/smttask/tests/test_project" smttask run tasklist/Orbit__*.taskdesc.json -v --cores 2
    assert [2 new records]

    # Failures
    Failing(x=3, reason="smttask run test (failure)").save('tasklist')

    from tqdm import tqdm
    a = 3
    for n in tqdm(range(3, 20)):
        a = a**2

    from time import sleep
