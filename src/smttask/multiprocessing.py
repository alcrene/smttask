"""
Defines two context which are used together when executing tasks.

- `unique_process_num` sets a unique number across all smttask processes,
  including those started in other shells. This is the number to use e.g.
  for unique cache files.
  The number is set as the environment variable SMTTASK_PROCESS_NUM.
  TODO?: Use a module variable instead, like worker_index ?
- `unique_worker_index` sets a unique number among the smttask proceses within
  the same multiprocessing pool. This is the number to use e.g. for progress
  bar offsets.
  The number is stored in this module, as the global variable `worker_idx`.
  It can be retrieved by other functions (e.g. the function defining the task)
  with `get_worker_index`, which defaults to returning 0 when the value is unset.

TODO?: Combine into a single context manager which sets both ?
"""

# FIXME: The current implementation has some race conditions:
#   - We assume that if a second call to `smttask run` is made, it is made
#     with enough delay that the first call has created all of its lock files.
#   - There is a window of time, between two runs, when the lock file is deleted.
#     If a new batch run is made at this time, it may allocate for itself a
#     process number which is already in the first call's `process_numbers`
#     array.

import os
import re
import logging
import multiprocessing
from pathlib import Path
from .config import config

logger = logging.getLogger(__name__)

lockfilename = "smttask_process-{}.lock"

## Shared variables for communication between processes ##
# (These variables must be initialized by calling `init_synchronized_vars`.)
# Termination flag, so processes which receive SIGINT can tell others to abort
# FIXME?: Pool only supports shared variables as global arguments, but this
#         will only work on Unix (see https://stackoverflow.com/a/1721911)
#         The link gives ideas of solutions that should work on Windows
#         (and Mac, which since 3.8 no longer uses fork: https://bugs.python.org/issue33725)
stop_workers = None
# Boolean array, of same length as the number of workers. When a worker wants
# to assign itself a unique index, it looks for an entry which is True (= free),
# and reserves it by setting to false.
free_worker_idcs = None
# Integer array, of same length as the number of workers. When a worker wants
# to create a lock file, it looks for the first free file of the form
# lockfilename.format(i), where i is an integer from this list.
process_numbers = None

## Per-process variables ##
worker_idx = None

def init_synchronized_vars(n_workers):
    """:param n_workers: Same parameter which would have been passed to `Pool`."""
    global stop_workers, free_worker_idcs, process_numbers
    stop_workers = multiprocessing.Value('b', False)
    free_worker_idcs = multiprocessing.Array('b', [True]*n_workers)
    n0 = get_highest_assigned_process_num() + 1
    process_numbers = multiprocessing.Array('I', range(n0, n0+n_workers))

class unique_worker_index:
    """
    Sets `worker_idx` to a value which is unique among simultaneously running processes.
    Uses 0 if the multiprocessing messaging variables are not initialized.
    """
    def __init__(self):
        global free_worker_idcs, worker_idx
        try:
            with free_worker_idcs.get_lock():
                for i, free in enumerate(free_worker_idcs):
                    if free:
                        free_worker_idcs[i] = False
                        logger.debug(f"Worker index {i}.")
                        worker_idx = i
                        return
                # This shouldn't happen, but just in case do something reasonable
                logger.debug(f"No free worker index found. Worker index set to {len(free_worker_idcs)}.")
                worker_idx = len(free_worker_idcs)
        except AttributeError:  # Happens if free_worker_idcs is None
            worker_idx = 0

    def __enter__(self):
        global worker_idx
        return worker_idx

    def __exit__(self, exc_type, exc_value, traceback):
        global free_worker_idcs, worker_idx
        if worker_idx is not None and free_worker_idcs is not None:
            try:
                free_worker_idcs[worker_idx] = True
            except IndexError:
                logger.error("IndexError when smttask tried to release a "
                             "multiprocessing worker.\n"
                             f"Worker index we attempted to release: {worker_idx}\n"
                             f"Dict of all free worker idcs: {free_worker_idcs}")
        worker_idx = None

def get_worker_index():
    global worker_idx
    return worker_idx or 0

def abort(value=None):
    """
    Use without argument to return the current 'abort' status.
    Pass `True` to send an abort signal to other processes.
    """
    global stop_workers
    if value is None:
        return stop_workers and stop_workers.value
    elif stop_workers:
        stop_workers.value = value

import tempfile
class unique_process_num:
    """
    Sets the environment variables SMTTASK_PROCESS_NUM to the smallest integer
    not already assigned to an smttask process.
    Assigned numbers are tracked by files
    """
    def __enter__(self):
        global process_number
        
        tmpdir = Path(tempfile.gettempdir())
        file = None
        for n in process_numbers:
            try:
                fpath = tmpdir/lockfilename.format(n)
                file = open(fpath, 'x')
            except OSError:
                pass
            else:
                break
        if file is None:
            raise RuntimeError("Smttask: The maximum number of processes have "
                               "already been assigned. If there are stale "
                               "lock files, they can be removed from "
                               f"{tmpdir}/{lockfilename.format('N')}")
        self.file = file
        self.fpath = fpath
        os.environ["SMTTASK_PROCESS_NUM"] = str(n)
        logger.debug(f"Assigned number {n} to this smttask process. "
                     f"Lock file created at location '{self.fpath}'.")
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        try:
            os.remove(self.fpath)
            logger.debug("Removed the smttask process number lock file at "
                         f"location '{self.fpath}'.")
        except (OSError, FileNotFoundError):
            logger.debug("The smttask process number lock file at location "
                         f"'{self.fpath}' was already removed.")

def get_highest_assigned_process_num():
    return max((-1, 
                *(int(m[1])
                  for m in (re.match(lockfilename.format("(\d+)"), f) 
                            for f in (f for f in os.listdir("/tmp")
                                      if f.startswith("smttask_process")))
                  if m)
               ))
