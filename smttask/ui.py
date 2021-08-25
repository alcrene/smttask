import click
import logging
import os
import io
import time
from datetime import datetime
from warnings import warn
import traceback
import multiprocessing
import functools
from pathlib import Path
import pdb as pdb_module
from tqdm.contrib.logging import logging_redirect_tqdm
import sumatra.commands
from .base import Task, EmptyOutput
from .config import config
from . import utils
from .tqdm import tqdm
from .multiprocessing import unique_process_num, unique_worker_index
import smttask.multiprocessing as smttask_mp
from .view import RecordStoreView

logger = logging.getLogger(__name__)

# https://stackoverflow.com/a/8146857
class NeverMatch(Exception):
    "An exception class that is never raised by any code anywhere"

@click.group()
def cli():
    pass

@cli.command()
def init():
    """Run an initialization wizard for SumatraTask.

    If you want to include this in a script, the underlying `smt init`
    command provided by Sumatra may be more appropriate.
    """
    # See https://stackoverflow.com/questions/37340049/how-do-i-print-colored-output-to-the-terminal-in-python
    BOLD = '\033[1m'
    END = '\033[0m'

    print(f"\n{BOLD}Overview{END}")
    print(
        "In order to ensure task results are never overwritten, while still "
        "providing consistent file locations for inputs, SumatraTask requires "
        "separate paths for output and input datastores (files in the 'input' directory "
        "are symlinked to their most recent equivalent in the 'output' "
        "directory). By default these are the subfolders './data' and "
        "'./data/run_dump'."
        )

    cwd = Path(os.getcwd())
    path_repo = cwd

    print(f"\n{BOLD}Select Git repository{END}")
    print("The project directory should be a git repository containing "
          "the code and settings used for the project.\n"
          "(Note: providing a different value will clone the repository at "
          "that location into the current directory.)")
    r = input(f"Project directory (default: '{path_repo}'): ")
    if r != "":
        path_repo = Path(r).expanduser()
        print(f"{path_repo} will be cloned to {cwd}.")

    run_file_pattern = "run"
    no_pattern = "-"
    print("\nYou can add a pattern to your .gitignore to exclude run files "
          "(files used to tie Tasks together and execute them). The default "
          "is to place these files in a directory named 'run'.")
    print(f"\n(Note: Pattern will not be added to .gitignore if already present; type '{no_pattern}' to avoid adding any pattern): ")
    r = input(f"Runfile exclude pattern to include in .gitignore (default: '{run_file_pattern}'): ")
    if r != "" and r != no_pattern:
        run_file_pattern = r
    elif r != no_pattern:
        already_present = False
        try:
            with open(path_repo/".gitignore", 'r') as f:
                for line in f:
                    line = line[:line.find('#')].strip()  # Remove comments and whitespace
                    if (line == run_file_pattern
                        or line.rsplit("/", maxsplit=1)[-1] == run_file_pattern):  # Absolute paths count as well (e.g. 'run' matches '/run')
                        already_present = True
                        break
        except FileNotFoundError:
            pass
        if already_present:
            print("Run file pattern already present in .gitignore.")
        else:
            print("Appending run file pattern to .gitignore.")
            with open(path_repo/".gitignore", 'a') as f:
                f.write(f"\n{run_file_pattern}\n")

    print(f"\n{BOLD}Setup Sumatra project{END}")

    project_name = cwd.stem
    r = input(f"Project name (default {project_name}): ")
    if r != "":
        project_name = r

    path_inputs = cwd/"data"
    r = input(f"\nInput datastore (default: {path_inputs}): ")
    if r != "":
        path_inputs = Path(r).expanduser()
    path_outputs = path_inputs/"run_dump"
    r = input(f"\nOutput datastore (default: {path_outputs}): ")
    if r != "":
        path_outputs = Path(r).expanduser()

    fail = False
    if path_inputs.exists() and not path_inputs.is_dir():
        print("Input datastore is not a directory.")
        fail = True
    if path_outputs.exists() and not path_outputs.is_dir():
        print("Output datastore is not a directory.")
        fail = True
    if path_inputs == path_outputs:
        print("Input and output datastore must be different.")
        fail = True
    # Check that input/output dirs are read-writable.
    # Dirs may not be created yet, so we go up the path until we find the
    # first already existing directory
    for path in (path_inputs, path_outputs):
        latest_parent = path
        while not latest_parent.exists():
            latest_parent = latest_parent.parent
        if not os.access(latest_parent, os.W_OK | os.R_OK):
            print(f"{latest_parent} is not read-writable")
            fail = True
    if fail:
        return

    # Construct the argument list as it would be passed on the CLI, and
    # call Sumatra's init
    argv_str = f"{project_name} --datapath {path_outputs} --input {path_inputs} " \
               f"--repository {path_repo}"
    print("This will initialize a Sumatra project with the following command:")
    print("smt init " + argv_str)
    r = input("Continue ? [Y/n] ")
    if r.lower() != "n":
        sumatra.commands.init(argv_str.split())
    print("Done initializing Sumatra.")

    print(f"\n{BOLD}Smttask initialization complete.{END}\n")

@cli.command()
@click.argument('taskdesc', nargs=-1,
                type=click.Path(exists=True, path_type=Path, resolve_path=True))
@click.option("-n", "--cores", default=-1,
    help="The number of parallel processes to use, if more than one TASKDESC "
         "is given. This parameter is ignored if there is only one TASKDESC. "
         "Zero or negative values indicate to use all available cores, minus "
         "the specified value.\n"
         "(Example: on a 4 core machine, the value -1 indicates to use 3 cores.)\n"
         "Default value value is -1.")
@click.option('--record/--no-record', default=True,
    help="Use `--no-record` to disable recording (and thereby also the check "
         "that the version control repository is clean).")
@click.option('--keep/--remove', default=False,
    help="By default, after successfully running a task, the taskdesc file is "
         "removed from disk. This can be disabled with the '--keep' option."
         "Cleaning is done on a best-effort basis: if the file cannot be found "
         "(e.g. because it was moved), no error is raised.")
@click.option('--recompute/--no-recompute', default=False,
    help="Add the flag '--recompute' to force computation of a task, even if "
         "a previous run is found.")
@click.option('--reason', default=None,
    help="Override the reason provided in the TASKDESC.")
@click.option('-v', '--verbose', count=True,
    help="Specify up to 2 times (`-vv`) to increase the logging level which "
         "is printed. Default is to print info, warning and error messages.\n"
         "-v: debug and up\n-vv: everything.")
    # TODO: -v debug (only my packages | or just exclude django), -vv debug (all packages)
@click.option('-q', '--quiet', count=True,
    help="Turn off info messages. Specifying multiple times will also "
         "turn off warning (-qq), error (-qqq) and critical (-qqqq) messages.")
@click.option('--progress-interval', default=0., type=float,
    help="Set the interval, in *minutes*, at which the progress bar will be "
         "updated. Especially useful when the output will be sent to a log "
         "file, to avoid logging thousands of progress bar updates.\n"
         "The default is to let tqdm dynamically adjust the update rate, for "
         "a smooth but CPU-friendly progress bar appropriate for console output.")
@click.option('--pdb/--no-pdb', default=False,
    help="Launch the pdb post-mortem debugger if an exception is raised while "
         "running the task. If there is more than one task, this option is "
         "mostly ignored since "
         "only errors in the root process will trigger a debugging session.")
@click.option('--wait', default=None,
    help="Specify an amount of time to wait before starting the task(s).\n"
         "Formats: '1h30m', '1hour 30min'.")
def run(taskdesc, cores, record, keep, recompute, reason, verbose, quiet,
        progress_interval, pdb, wait):
    """
    Execute the Task(s) defined by TASKDESC. If multiple TASKDESC files are
    passed, these are executed in parallel, with the number of parallel
    processes determined by CORE.

    A taskdesc can be obtained by calling `.save()` on an instantiated task.

    This commands creates the environment variable SMTTASK_PROCESS_NUM, which
    stores an integer >= 0. This number is unique between concurrent calls
    to `smttask run`, and can be used to assign unique path names to each
    concurrent process. This is useful if for example tasks require a
    compilation directory, and one wants to avoid simultaneous tasks attempting
    to use the same directory.
    """
    cwd = Path(os.getcwd())

    # Concatenate taskdesc files, recursing into directories
    taskdesc_files = []
    for taskdesc_path in taskdesc:
        if taskdesc_path.is_dir():
            for dirpath, dirnames, filenames in os.walk(taskdesc_path):
                for filename in filenames:
                    taskdesc_files.append(Path(dirpath)/filename)
        else:
            taskdesc_files.append(taskdesc_path)

    n_tasks = len(taskdesc_files)
    cores = min(n_tasks, cores)
    config.max_processes = min(n_tasks, cores)
    cores = config.max_processes  # max_processes converts cores<0 to cpu_count-cores
    verbose *= 10; quiet *= 10  # Logging levels are in steps of 10
    default = logging.INFO
    loglevel = max(min(default+quiet-verbose,
                       logging.CRITICAL),
                   logging.DEBUG)
    logging.basicConfig(level=loglevel, force=True)
        # force=True to reset the root logger in case it was already created
    def task_loader(taskdesc_paths: 'Sequence[Path]'):
        for taskpath in taskdesc_paths:
            try:
                with open(taskpath) as file:
                    taskdesc = file.read()
            except (Exception if pdb else NeverMatch) as e:
                pdb_module.post_mortem()
            else:
                if not os.path.exists(taskpath):
                    taskpath = None
                yield taskdesc, taskpath

    if wait:
        amount = utils.parse_duration_str(wait)
        # We could just `wait(amount)`, but then the user would have no progress indicator
        orig_time = datetime.now()
        t = tqdm(desc=f"Waiting {amount} seconds",
                 total=float(amount),
                 # Remove iteration counters – it's 1/s by construction
                 bar_format='{l_bar}{bar}| [{elapsed}<{remaining}]'
                 )
            # Update the indicator ourselves since `sleep(1)` is only approx 1s
        Δ = 0
        while Δ < amount:
            time.sleep(1)
            Δ = (datetime.now() - orig_time).seconds
            t.n = Δ
            t.update(0)  # Trigger UI update of progress bar
        t.close()
        print("")  # Add a new line so we can see where the real output starts

    if n_tasks <= 1:
        for taskinfo in tqdm(task_loader(taskdesc_files),
                             desc="Tasks",
                             total=n_tasks,
                             position=0
                            ):
            _run_task(taskinfo, record, keep, recompute, reason, loglevel,
                      progress_interval, pdb=pdb)
    else:
        if pdb:
            warn("The '--pdb' option is mostly ignored when there is more than one task.")
        smttask_mp.init_synchronized_vars(cores)
        # We use maxtaskperchild because some tasks can only be run once (e.g. RNG creators)
        # QUESTION: What is the cost to this ? There solutions for RNGs that don't require this – is this cost justified ?
        with multiprocessing.Pool(cores, maxtasksperchild=1) as pool:
            # NOTE: try-catch must be INSIDE the Pool context, otherwise when
            # an exception occurs, we crash out and don't execute clean-up code
            worker = functools.partial(
                _run_task, record=record, keep=keep, recompute=recompute,
                reason=reason, loglevel=loglevel,
                progress_interval=progress_interval, pdb=False, subprocess=True)
            worklist = pool.imap(worker, task_loader(taskdesc_files))
            pool.close()
            try:
                for task in tqdm(worklist,
                                 desc="Tasks",
                                 total=n_tasks,
                                 position=0
                                ):
                    if smttask_mp.stop_workers.value:
                        break
                # NOTE: When a process returns, the next work unit is started
                #       IMMEDIATELY. This means that even if we catch an Interrupt
                #       exception here, and set stop_workers to False, it is
                #       already too late for stopping the next task (although
                #       all subsequence ones would see the flag). Ergo why we
                #       set the flag in the exception within the subprocess,
                #       which is necessarily completed before that process completes.
            except (KeyboardInterrupt, SystemExit):
                # The keyboard interrupt is sent to each process simultaneously
                # That means that even if the tasks catch it, it must ALSO be
                # caught by the mother process (this one)
                smttask_mp.abort(True)  # Redundant with subprocess; <=> assertion
                logger.info("`run` was forcefully terminated.")
                # If we simply continue, the Pool context manager will call
                # `terminate()` on the subprocesses immediately, preventing
                # them from completing their exception handlers.
                # Calling `join` forces `map` to wait until all work units are
                # complete. This will still execute all work units, but they
                # will exit immediately since `stop_workers` is now True.
                # TODO: Add timeout. This may require using map_async
                pool.close()
                pool.join()

            except (Exception if pdb else NeverMatch) as e:
                pdb_module.post_mortem()


def _run_task(taskinfo, record, keep, recompute, reason, loglevel,
              progress_interval=0., pdb=False, subprocess=False):
    if smttask_mp.abort():
        logger.debug("Received termination signal before starting task. Aborting task execution.")
        return

    logging.basicConfig(level=loglevel)
    # Require an extra -v to see the 'debug' level of noisy dependencies
    # TODO: Any way to do this for all dependencies, but leave the project logger untouched ?
    # TODO: Single function called both here and in mother process
    if loglevel <= logging.DEBUG:
        logging.getLogger('django').setLevel(loglevel+10)
        logging.getLogger('git').setLevel(loglevel+10)
        logging.getLogger('matplotlib').setLevel(loglevel+10)
    logging.captureWarnings(True)

    with logging_redirect_tqdm(tqdm_class=tqdm):
        taskdesc, taskpath = taskinfo
        config.record = record

        with unique_process_num(), unique_worker_index():
            tqdm.defaults.position = smttask_mp.get_worker_index()
            if progress_interval > 0:
                tqdm.defaults.miniters = 1  # Deactivate dynamic miniter
                tqdm.defaults.mininterval = progress_interval*60.  # Convert to minutes
            try:
                task = Task.from_desc(taskdesc)
                result = task.run(recompute=recompute, reason=reason)
                # TODO: If the `outputs.write()` call fails, outputs will
                #       not be empty, but we still should detect that as an
                #       unsuccessful run.
                if isinstance(result, EmptyOutput):
                    if result.status == 'killed':
                        logger.info("Task was killed.")
                    else:
                        logger.info("Task terminated abnormally.")
                elif record and not keep:
                    # Task was completed successfully: clean the file
                    try:
                        os.remove(taskpath)
                    except (OSError, FileNotFoundError):
                        pass

            except (KeyboardInterrupt, SystemExit):
                # By catching the exception here, we allow context to run its cleanup
                smttask_mp.abort(True)
                logger.debug("Caught KeyboardInterrupt. Sending termination "
                             "signal to other processes.")

            except (Exception if pdb else NeverMatch) as e:
                pdb_module.post_mortem()

            except (Exception if subprocess else NeverMatch) as e:
                # In a subprocess, we need to print the traceback ourselves to see it
                # https://jichu4n.com/posts/python-multiprocessing-and-exceptions/
                exc_buffer = io.StringIO()
                traceback.print_exc(file=exc_buffer)
                logging.error("Uncaught exception in worker process:\n"
                              f"{exc_buffer.getvalue()}")
                raise e

@cli.group()
def rebuild():
    pass

@rebuild.command()
def datastore():
    """Rebuild the input datastore.

    For each Sumatra record, instantiate the task, recompute the expected
    output name, and recreate the links in the input datastore, pointing to
    the recorded data location.

    This is useful if e.g. an update has caused all of the Task digests to
    change, in order for those computations to be found by future Tasks.

    This function is safe in the sense that it never deletes or moves the
    original data, and will ignore records for which it is unable to
    reconstruct the corresponding Task. It may however replace links.
    """
    logging.basicConfig(level=logging.INFO)
    recordlist = RecordStoreView()
    recordlist.rebuild_input_datastore(utils.compute_input_symlinks)

if __name__ == "__main__":
    cli()
