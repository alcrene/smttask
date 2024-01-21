import click
import contextlib
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
from . import utils
from . import multiprocessing as smttask_mp
from .base import Task, EmptyOutput
from .config import config
from .tqdm import tqdm
from .multiprocessing import unique_process_num, unique_worker_index
from .view import RecordStoreView

logger = logging.getLogger(__name__)

class NeverError(Exception):
    "An exception class that is never raised by any code anywhere"

@click.group()
def cli():
    pass

@cli.group()
def project():
    """Manage SumatraTask projects."""
    pass

@project.command()
def init():
    """Run the initialization wizard for SumatraTask.

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
    path_repo = Path(r).expanduser()
    if not (path_repo/".git").exists():
        print(f"\nERROR: {path_repo.absolute()} is not a git repository.\n")
        return  # FAILURE - EARLY EXIT
    if r != "":
        print(f"{path_repo} will be cloned to {cwd}.")


    no_pattern = ""
    # run_file_pattern = "run"
    # run_file_pattern = no_pattern
    print(f"\n{BOLD}Add a run file pattern to .gitignore{END}")
    print("\nYou can add a pattern to your .gitignore to exclude run files "
          "(files used to tie Tasks together and execute them). For example, "
          "if you want to place these files in a directory named 'run', "
          "add the pattern 'run'.")
    print("If you intend to create or run tasks from a separate project "
          "directory (e.g. the directory with labnotes/reports/manuscript), "
          "you don't need to add a pattern to .gitignore.")
    print(f"\n(Note: Pattern will not be added to .gitignore if already present; "
          "leave blank to avoid adding any pattern): ")
    # r = input(f"Runfile exclude pattern to include in .gitignore (default: '{run_file_pattern}'): ")
    r = input(f"Runfile exclude pattern to include in .gitignore (default: '{no_pattern}'): ")
    # if r != no_pattern:
    #     run_file_pattern = r
    if r != no_pattern:
        already_present = False
        run_file_pattern = r
        with contextlib.suppress(FileNotFoundError):
            with open(path_repo/".gitignore", 'r') as f:
                for line in f:
                    line = line[:line.find('#')].strip()  # Remove comments and whitespace
                    if (line == run_file_pattern
                        or line.rsplit("/", maxsplit=1)[-1] == run_file_pattern):  # Absolute paths count as well (e.g. 'run' matches '/run')
                        already_present = True
                        break
        if already_present:
            print("Run file pattern already present in .gitignore.")
        else:
            print("Appending run file pattern to .gitignore.")
            with open(path_repo/".gitignore", 'a') as f:
                f.write(f"\n{run_file_pattern}\n")

    print(f"\n{BOLD}Setup Sumatra project{END}")

    project_name = cwd.stem
    r = input(f"Project name (default '{project_name}'): ")
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
    # NB: This works also when paths contain spaces
    init_args = [project_name, "--datapath", path_outputs,
                 "--input", path_inputs, "--repository", path_repo]
    init_args = [str(a) for a in init_args]  # Sumatra does not support Path at present
    print("This will initialize a Sumatra project with the following command:")
    print("smt init " + " ".join(init_args))
    r = input("Continue ? [Y/n] ")
    if r.lower() != "n":
        sumatra.commands.init(init_args)
    print("Done initializing Sumatra.")

    print(f"\n{BOLD}Smttask initialization complete.{END}\n")
    
import inspect
envs_dir_default = inspect.signature(utils.clone_conda_project).parameters["envs_dir"].default
@project.command()
@click.argument("src", type=click.Path(exists=True), nargs=-1)
@click.argument("dest", type=click.Path(exists=False))
@click.option("--src-env", type=str, default="",
    help="The name of the conda environment used in the source project. "
         "If not specified, the last component of the SRC path is assumed to "
         "be the source environment name.")
@click.option("--dest-env", type=str, default="",
    help="The name of the conda environment to create for the destination project."
         "If not specified, the last component of the DEST path is assumed to "
         "be the destination environment name.")
@click.option("--envs-dir", type=click.Path(exists=False),
              default=envs_dir_default,
    help="The directory in which to create the new environment for the cloned "
         f"environment. Default: {envs_dir_default}.")
@click.option("--config", type=click.Path(exists=False), multiple=True, default=(),
    help="Additonal configuration file to copy over to the clone project. "
    "Configuration files SHOULD be located in the project's top level directory, "
    "SHOULD by compatible with Python's `configparser` module, "
    "and SHOULD NOT be tracked with version control. "
    "Provided path may point to a template file of the same name outside the project, "
    "whose values are substituted for those of the source project's config file."
    )
@click.option("--install-kernel/--no-kernel", default=True,
    help="By default, an IPython kernel is installed to make the cloned "
         "environment available from within Jupyter. This can be prevented "
         "with the --no-kernel option.")
def clone(src, dest, src_env, dest_env, envs_dir, config, install_kernel):
    """
    Clone a project from SRC to DEST.
    If only one value is supplied, SRC is taken to be the current directory.
    As with other Sumatra commands, can be called anywhere within a Sumatra
    projects folder hierarchy.
    
    Currently only projects using Conda environments are supported.
    """
    if len(src) == 0:
        src = Path.cwd()
    elif len(src) > 1:
        raise ValueError("Only one SRC project may be specified.")
    else:
        src = src[0]
    src = Path(src)
    dest = Path(dest)
    envs_dir = Path(envs_dir)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exists")
    src = src.expanduser().resolve()
    # dest = dest.expanduser().resolve()
    # Emulate Sumatra: allow calling from anywhere below the root project folder
    orig_src = src
    while src:
        if ".smt" in os.listdir(src):
            break
        src = src.parent
    if not src:
        raise FileNotFoundError(f"{orig_src} is not contained within a Sumatra "
                                "project.")
    # Infer additional arguments required for clone
    if not src_env:
        src_env = src.stem
    if not dest_env:
        dest_env = dest.stem
    if not envs_dir.expanduser().resolve().exists():
        envs_dir.expanduser().resolve().mkdir(parents=True)
    
    # Call utility function
    smttask_logger = logging.getLogger("smttask")
    old_level = smttask_logger.level
    smttask_logger.setLevel(logging.INFO)  # TODO?: Verbosity option ?
    utils.clone_conda_project(src, dest, src_env, dest_env, envs_dir, config, install_kernel)
    smttask_logger.setLevel(old_level)

@cli.command()
@click.argument('taskdesc', nargs=-1,
                type=click.Path(exists=True, resolve_path=True))
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
@click.option('--keep/--clean', default=False,
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
@click.option('-r', '--reverse', default=False,
    help="If multiple tasks are passed, iterate over them in reversed order. "
         "Possible use case: smttask is already running throw a list of tasks, "
         "and while we have some extra compute time, we want to knock off some "
         "tasks from the end of the list.")
@click.option('--start-spacing', default=1.5, type=float,
    help="When running with more than one core, the amount of time (in seconds) "
         "to wait before starting the next job. This can help avoid lock "
         "conflicts (all recorded tasks need to access the .smt record store "
         "when they start and when they end).\nDefault: 1.5 s")
@click.option('--progress-interval', default=0., type=float,
    help="Set the interval, in *minutes*, at which the progress bar will be "
         "updated. Especially useful when the output will be sent to a log "
         "file, to avoid logging thousands of progress bar updates.\n"
         "The default is to let tqdm dynamically adjust the update rate, for "
         "a smooth but CPU-friendly progress bar appropriate for console output.")
@click.option('--record-store', default=None,
              type=click.Path(exists=False, dir_okay=False),
    help="Specify the Sumatra record store to which execution records should "
         "be saved; if no file exists at the specified location, a new record "
         "store is created. This option is only required to use a different "
         "store that the one specified in project file. The imagined use case "
         "is when launching multiple simultaneous runs – the default SQLite "
         "backend is not very robust against concurrent access, and can suffer "
         "corruption in such cases. Using separate record store avoids this "
         "problem, at the cost of having to merge record stores afterwards. "
         "Warning: this feature relies on possibly unsafe manipulations of "
         "Sumatra objects. If you need this feature, consider switching to a "
         "PostgreSQL backend instead.\nSee also the `smttask store merge` "
         "command for combining record stores.")
@click.option('--pdb/--no-pdb', default=False,
    help="Launch the pdb post-mortem debugger if an exception is raised while "
         "running the task. If there is more than one task, this option is "
         "mostly ignored since "
         "only errors in the root process will trigger a debugging session.")
@click.option('--wait', default=None,
    help="Specify an amount of time to wait before starting the task(s).\n"
         "Formats: '1h30m', '1hour 30min'.")
@click.option('--import', "pkg_imports", multiple=True,
    help="Import these packages before running the task. Also adds them to the "
         "list of safe packages. Intended for importing the base project package.")
def run(taskdesc, cores, record, keep, recompute, reason, verbose, quiet, reverse,
        start_spacing, progress_interval, record_store, pdb, wait, pkg_imports):
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
    taskdesc = tuple(Path(p) for p in taskdesc)  # With v8, we could do this by passing a 'path_type' argument to click.Path

    cwd = Path(os.getcwd())
    if record_store:
        record_store = os.path.abspath(record_store)

    # Concatenate taskdesc files, recursing into directories
    taskdesc_files = []
    for taskdesc_path in taskdesc:
        if taskdesc_path.is_dir():
            for dirpath, dirnames, filenames in os.walk(taskdesc_path):
                # At present, the cost of sorting `filenames` seems to be worth
                # it to have predictable execution and easier to read output.
                # TODO: Sorting is imperfect: 'task-9' sorts after 'task-100'
                taskdesc_files.extend(
                    sorted(Path(dirpath)/filename for filename in filenames))
        else:
            taskdesc_files.append(taskdesc_path)
    if reverse:
        taskdesc_files = taskdesc_files[::-1]

    n_tasks = len(taskdesc_files)
    config.max_processes = cores  # NB: `cores` can be negative. max_processes converts cores<0 to cpu_count-cores
    config.max_processes = min(n_tasks, config.max_processes)  # config.max_processes is strictly positive
    cores = config.max_processes  
    verbose *= 10; quiet *= 10  # Logging levels are in steps of 10
    default = logging.INFO
    loglevel = max(min(default+quiet-verbose,
                       logging.CRITICAL),
                   logging.DEBUG)
    logging.basicConfig(level=loglevel, force=True)
        # force=True to reset the root logger in case it was already created
    def task_loader(taskdesc_paths: 'Sequence[Path]', start_spacing=start_spacing):
        for taskpath in taskdesc_paths:
            try:
                with open(taskpath) as file:
                    taskdesc = file.read()
            except (Exception if pdb else NeverError) as e:
                pdb_module.post_mortem()
            else:
                if not os.path.exists(taskpath):
                    taskpath = None
                yield taskdesc, taskpath
            time.sleep(start_spacing)  # Space out start times to avoid lock conflicts

    if wait:
        amount = utils.parse_duration_str(wait)
        # We could just `wait(amount)`, but then the user would have no progress indicator
        orig_time = datetime.now()
        t = tqdm(desc=f"Waiting {amount} seconds",
                 total=float(amount),
                 # Remove iteration counters – it's 1/s by construction
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

    start_time = datetime.now()
    
    if cores <= 1:
        smttask_mp.init_synchronized_vars(n_tasks)
        for taskinfo in tqdm(task_loader(taskdesc_files),
                             desc="Tasks",
                             total=n_tasks,
                             position=0
                            ):
            _run_task(taskinfo, record, keep, recompute, reason, loglevel,
                      pkg_imports, progress_interval,
                      record_store=record_store, pdb=pdb)
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
                reason=reason, pkg_imports=pkg_imports, loglevel=loglevel,
                progress_interval=progress_interval, record_store=record_store,
                pdb=False, subprocess=True)
            worklist = pool.imap(worker, task_loader(taskdesc_files))
            pool.close()
            PdbExc = pdb and Exception or NeverError
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

            except PdbExc as e:
                pdb_module.post_mortem()
                
    end_time = datetime.now()
    
    print("\nRuns completed")
    print(f"Start time: {start_time:%c}")
    print(f"End time  : {end_time:%c}")

def _run_task(taskinfo, record, keep, recompute, reason, loglevel, pkg_imports,
              progress_interval=0., record_store=None, pdb=False, subprocess=False):
    if smttask_mp.abort():
        logger.debug("Received termination signal before starting task. Aborting task execution.")
        return

    # Add the CWD to path
    import sys
    sys.path.insert(0, "")

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
                for pkg in pkg_imports:
                    from importlib import import_module
                    config.safe_packages.add(pkg)
                    import_module(pkg)
                task = Task.from_desc(taskdesc)
                result = task.run(recompute=recompute, reason=reason,
                                  record_store=record_store)
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

            except (Exception if pdb else NeverError) as e:
                pdb_module.post_mortem()

            except (Exception if subprocess else NeverError) as e:
                # In a subprocess, we need to print the traceback ourselves to see it
                # https://jichu4n.com/posts/python-multiprocessing-and-exceptions/
                exc_buffer = io.StringIO()
                traceback.print_exc(file=exc_buffer)
                logging.error("Uncaught exception in worker process:\n"
                              f"{exc_buffer.getvalue()}")
                raise e

@cli.group()
def store():
    """Inspect or manipulate the records and data stores."""
    pass
# TODO?: Alias datastore, recordstore group commands ?

@store.command()
@click.argument('taskdesc', nargs=1, type=click.File())
def find_output(taskdesc):  # NB: Use `find-output` on CLI
    """
    Find output files for a previously run TASKDESC.

    If the specified task was not already run, or not recorded, prints
    a message stating that no task output was found.
    """
    task = Task.from_desc(taskdesc.read())
    try:
        found_files = task.find_saved_outputs()
    except FileNotFoundError:
        print(f"No task output found for task {task.name}.")
    else:
        print(f"Outputs for task {task.name}:")
        if found_files.is_partial:
            print("(Only partial task outputs were found.)")
        varwidth = max(len(varnm) for varnm in found_files.outputpaths)
        for varnm, path in found_files.outputpaths.items():
            print(f"  {varnm:>{varwidth}} -> {path}")
            
@store.command()
@click.argument("record", nargs=1, type=str)
@click.option("--out", default='.', type=click.Path(exists=False),
    help="The location at which to save the task file, the default name is a "
         "combination of task name and digest. If `out` is a directory, the "
         "file is saved in that directory with the default name; the default "
         "is to save in the current directory.")
@click.option("--force/--no-force", "-f", default=False,
    help="Allow overwriting an existing file.")
def recreate(record, out, force):
    """
    Recreate the task file used to create RECORD and save it at location OUT.
    """
    taskdesc = utils.task_from_record(record, format="taskdesc")
    try:
        outpath = taskdesc.save(out, allow_overwrite=force)
    except FileExistsError as e:
        print(e)
        print("To overwrite the existing file, use the --force (-f) option.")
    else:
        print(f"Task file was saved at: {outpath}")

@store.command()
def rebuild():
    """
    Rebuild the input datastore.

    For each Sumatra record, instantiate the task, recompute the expected
    output name, and recreate the links in the input datastore, pointing to
    the recorded data location.

    This is useful if e.g. an update has caused all of the Task digests to
    change, in order for those computations to be found by future Tasks.

    This function is safe in the sense that it never deletes nor moves the
    original data, and will ignore records for which it is unable to
    reconstruct the corresponding Task. It may however replace links.
    """
    logging.basicConfig(level=logging.INFO)
    rsview = RecordStoreView()
    rsview.rebuild_input_datastore(utils.compute_input_symlinks)

@store.command()
@click.argument('taskdesc', nargs=-1,
                type=click.Path(exists=False, resolve_path=False))
@click.option('--keep/--clean', default=False,
    help="By default, if the task's output data files are found, the taskdesc "
         "file is removed from disk. This can be disabled with the '--keep' "
         "option. Cleaning is done on a best-effort basis: if the file cannot "
         "be found (e.g. because it was moved), no error is raised.")
@click.option('--dry-run/--actual-run', default=False,
    help="Don't make any modifications to the record store. Instead, print "
         "the list of records which would be added, and the list of task "
         "files which would be deleted.")
@click.option('-v', '--verbose', count=True,
    help="Specify up to 2 times (`-vv`) to increase the logging level which "
         "is printed. Default is to print info, warning and error messages.\n"
         "-v: debug and up\n-vv: everything.")
    # TODO: -v debug (only my packages | or just exclude django), -vv debug (all packages)
@click.option('-q', '--quiet', count=True,
    help="Turn off info messages. Specifying multiple times will also "
         "turn off warning (-qq), error (-qqq) and critical (-qqqq) messages.")
def create_surrogates(taskdesc, keep, dry_run, verbose, quiet):
    """
    Create surrogate records for outputs without records.

    For each provided TASKDESC file, check

    1) If outputs for that task are stored on disk, indicating that it was
       already run.
    2) If there is a matching entry in the record store.

    If (1) is true but (2) is false, then a new surrogate record is created,
    to associate the task desc with the output.
    Any number of TASKDESC files may be provided, and directories will be
    recursed into.

    This allows routines which query the record store for outputs to work as
    expected, but of course statistics like run time for surrogate records are
    undefined.

    The "surrogate" tag is added to all surrogate records.

    Reasons for having task outputs without associate record store entries
    include executing a task without recording, merging data stores without
    merging the associated record stores, and write conflicts when multiple
    processes attempt to access the record store simultaneously.

    It may be easier to understand this function with a sample of its output;
    such an example can be found in the smttask docs at this location:
    :doc:`smttask/docs/user-api/example_output_smttask_store_create-surrogates.md </user-api/example_output_smttask_store_create-surrogates.md>`.
    """
    taskdesc = tuple(Path(p) for p in taskdesc)  # With v8, we could do this by passing a 'path_type' argument to click.Path

    import sys
    import shutil
    from sumatra.core import TIMESTAMP_FORMAT, STATUS_FORMAT
    from sumatra.programs import PythonExecutable
    from sumatra.datastore.filesystem import DataFile

    # Set up logging
    verbose *= 10; quiet *= 10  # Logging levels are in steps of 10
    default = logging.INFO
    loglevel = max(min(default+quiet-verbose,
                       logging.CRITICAL),
                   logging.DEBUG)
    logging.basicConfig(level=loglevel, force=True)
        # force=True to reset the root logger in case it was already created

    rsview = RecordStoreView()

    record_outputpaths = {frozenset(str(Path(p).resolve()) for p in rec.outputpaths)
                          for rec in tqdm(rsview, desc="Scanning record store")}

    # Concatenate taskdesc files, recursing into directories
    taskdesc_files = []
    for taskdesc_path in taskdesc:
        if taskdesc_path.is_dir():
            for dirpath, dirnames, filenames in os.walk(taskdesc_path):
                # At present, the cost of sorting `filenames` seems to be worth
                # it to have predictable execution and easier to read output.
                # TODO: Sorting is imperfect: 'task-9' sorts after 'task-100'
                taskdesc_files.extend(
                    sorted(Path(dirpath)/filename for filename in filenames))
        else:
            taskdesc_files.append(Path(taskdesc_path))

    if len(taskdesc_files) == 0:
        print("No task files were specified. Exiting.")
        return

    taskfiles_to_delete = []
    taskfiles_to_add_as_records = []
    taskfiles_untouched = []
    for taskpath in tqdm(taskdesc_files, desc="Iterating over task files"):
        if not taskpath.exists():
            taskfiles_to_delete.append((taskpath, "(∄ task desc)"))
            continue
        task = Task.from_desc(taskpath)
        # Set the `outroot` after loading `task`: loading the task may change the project directory
        outroot = Path(config.project.data_store.root)
        outputpaths = frozenset(str((outroot/p).resolve()) for p in task.outputpaths.values())
        if task.saved_to_input_datastore:
            # The task's output files have been found on disk
            if outputpaths in record_outputpaths:
                # There is at least one record pointing to these output files
                taskfiles_to_delete.append((taskpath, "(∃ output, ∃ record)"))
            else:
                # The outputs exist, but no record points to them
                # => Add a surrogate record
                taskfiles_to_add_as_records.append(taskpath)
                taskfiles_to_delete.append((taskpath, "(∃ output, ∄ record)"))
                # TODO: DRY with task_types._run_and_record()
                input_data = [input.generate_key()
                              for input in task.input_files]
                module_name = getattr(task, '_module_name', type(task).__module__)
                module = sys.modules[module_name]
                parameter_str = task.desc.json(indent=2)
                try:
                    parameters = config.ParameterSet(parameter_str)
                except Exception as e:
                    parameters = parameter_str
                label = datetime.now().strftime(TIMESTAMP_FORMAT) + '_' + task.digest[:6]

                if not dry_run:
                    smtrecord = config.project.new_record(
                        parameters=parameters,
                        input_data=input_data,
                        script_args=type(task).__name__,
                        executable=PythonExecutable(sys.executable),
                        main_file=module.__file__,
                        reason=task.reason,
                        label=label
                    )
                    smtrecord.version = "<unknown>"
                        # Set version after creating record, otherwise the
                        # no modifications check will fail
                        # TODO: Can we disable to modification check completely ?
                        # It's not relevant anyway.
                    smtrecord.add_tag(STATUS_FORMAT % "finished")
                        # NB: status must be one of those in sumatra.web.templatetags.filters:labelize_tag:style_map
                    smtrecord.add_tag("surrogate")
                    # TODO: Again, DRY with task_types._run_and_record()
                    smtrecord.output_data = [
                        DataFile(Path(path).relative_to(outroot.resolve()),
                                 config.project.data_store).generate_key()
                        for path in outputpaths]

                    config.project.save_record(smtrecord)
                    config.project.save()
        else:
             taskfiles_untouched.append(taskpath)

    have_been = "would be" if dry_run else "have been"
    will = "would" if dry_run else "will"

    if taskfiles_to_add_as_records:
        print(f"Surrogate records {have_been} added for the following tasks:")
        for taskpath in taskfiles_to_add_as_records:
            print(f"  {taskpath}")

    if not keep and taskfiles_to_delete:
        print(f"\nThe following task files {will} be removed:")
        termcols = shutil.get_terminal_size().columns
        w = max(len(str(t[0])) for t in taskfiles_to_delete)
        w = min(w, termcols - 25)  # Max columns to use for the path before truncating
            # 20: max width of `reason`  |  3: spacing used in formatted string
        for taskpath, reason in taskfiles_to_delete:
            pathstr = str(taskpath)
            if len(pathstr) > w: pathstr = "…" + pathstr[-w-1:]
            print(f"  {pathstr:<{w}}   {reason}")

    if taskfiles_untouched:
        print(f"\nThe following task files {will} be kept since no "
                    "corresponding output files were found:")
        for taskpath in taskfiles_untouched:
            print(f"  {taskpath}")

    if not keep and not dry_run:
        for taskpath, _ in taskfiles_to_delete:
            try:
                os.remove(taskpath)
            except (OSError, FileNotFoundError):
                pass
        print("Aforementioned task files have been removed.")

@store.command()
@click.argument('sources', nargs=-1,
                type=click.Path(exists=True, resolve_path=False))
@click.option('--target', default=None,
              type=click.Path(exists=True, dir_okay=False),
    help="The record store into which to merge the entries from the source "
         "record stores. The merge is one way: only the target store is "
         "modified. The default is to merge into the current ")
@click.option('--keep/--clean', default=False,
    help="By default, source record stores are not deleted after being merged "
         "into the target. Specifying '--clean' indicates to remove them. "
         "Stores are never removed if they caused a record name collision.")
@click.option('--backup/--no-backup', default=True,
    help="By default, backups of the record stores is made. (For the target "
         "store, the backup gets the suffix '.backup', while for the source "
         "stores, they are placed in a '.backup' subdirectory.) With the "
         "'--no-backup' option, the target store is simply overwritten, and "
         "source stores are permanently removed (unless '--keep' is also passed).")
@click.option('-v', '--verbose/--quiet', default=False,
    help="Print more verbose output (each store which is deleted).")
def merge(sources, target, keep, backup, verbose):
    """
    Merge entries from multiple record stores.
    
    SOURCES may be either record store files or directories; directories are
    recursed into. If directories, they should only contain record store files.
    Hidden files and directories (those starting with '.') are skipped.

    Intended usage is for combining run data that was recorded in separate
    record stores with the --record-store option of `smttask run`.
    E.g., if multiple runs used all different stores and placed them under
    the directory 'run/tmp_stores', they can be merged into the current project:

        smttask store merge run/tmp_stores

    To merge into a record store at a different location:

        smttask store merge run/tmp_stores --target path/to/record_store
    """
    sources = tuple(Path(p) for p in sources)  # With v8, we could do this by passing a 'path_type' argument to click.Path

    # Reference: sumatra.commands:sync()
    import shutil
    import textwrap
    from sumatra.recordstore import get_record_store
    from sumatra.recordstore import get_record_store, have_django
    from .utils import sync_one_way
    if have_django:
        from sumatra.recordstore import DjangoRecordStore
        from django.db import connections
    else:
        class DjangoRecordStore:  # Dummy type which will also return False in `isinstance` checks
            pass

    # Concatenate source files, recursing into directories
    source_files = []
    for store_path in sources:
        if store_path.is_dir():
            for dirpath, dirnames, filenames in os.walk(store_path):
                # Skip hidden files and directories
                for dirname in dirnames[:]:
                    if dirname.startswith('.'):
                        dirnames.remove(dirname)
                for filename in filenames[:]:
                    if filename.startswith('.'):
                        filenames.remove(filename)
                # Add non-hidden files to the list of sources
                source_files.extend(
                    sorted(Path(dirpath)/filename for filename in filenames))
        else:
            source_files.append(store_path)

    if len(source_files) == 0:
        print("No files were found at the given location. Exiting.")
        return

    if target is None:
        target_store = config.project.record_store
    else:
        target_store = get_record_store(str(target))

    if backup:
        target_store.backup()

    all_collisions = {}
    # NB: Django requires that all record stores be loaded before using any of them
    src_stores = [get_record_store(str(src_path))
                  for src_path in tqdm(source_files, desc="Loading record stores")]
    for src_path, src_store in tqdm(zip(source_files, src_stores),
                                    desc="Merging record stores",
                                    total=len(src_stores)):
        collisions = sync_one_way(src_store, target_store, config.project.name)
        # Before moving or deleting the file, we need to close the DB connection
        if isinstance(src_store, DjangoRecordStore):
            connections[src_store._db_label].close()
        # If the sync worked without collisions, now clean up the store file
        # Otherwise, add to the list of collisions to be printed once all stores are merged
        if collisions:
            all_collisions[src_path] = collisions
        elif not keep and backup:
            backupdir = Path(src_path).parent/".backup"
            backupdir.mkdir(parents=True, exist_ok=True)
            backuppath = backupdir/Path(src_path).name
            src_path.rename(backuppath)
            if verbose:
                tqdm.write(f"Moved record store to backup location {backuppath}")
        elif not keep and not backup:
            os.remove(src_path)
            if verbose:
                tqdm.write(f"Removed record store at location {src_path}.")
        elif verbose:
            tqdm.write(f"The record store at location {src_path} can be removed. "
                       "(use --clean to do this automatically).")

    if all_collisions:
        print()
        print("Merge incomplete: the record names listed below occur in both "
              "the indicated source store and the target store, and the "
              "corresponding records in each store differ.")
        termcols = shutil.get_terminal_size().columns
        for src_path, collisions in all_collisions.items():
            print(src_path)
            print("  " + "\n  ".join(textwrap.wrap(", ".join(collisions), termcols-5)))
            print()

@cli.group()
def debug():
    """Debugging utilities."""
    pass

import json
import rich.console

@debug.command()
@click.argument('task1', nargs=1,
                type=click.Path(exists=True, resolve_path=False))
@click.argument('task2', nargs=1,
                type=click.Path(exists=True, resolve_path=False))
def compare(task1, task2):
    """
    Compare two task files that should be the same but aren't.

    Common reasons for this to happen:
    - Values in the serialization with undefined order.
    - A random seed is not set.
    - Input parameters which contain state.
    """

    console = rich.console.Console()

    with open(task1) as f:
        json1 = f.read()
    with open(task2) as f:
        json2 = f.read()

    # Find characters that differ and print the context around them.

    # %%
    quiet = False
    with console.pager():
        for i, (c1, c2) in enumerate(zip(json1, json2)):
            if c1 != c2 and not quiet:
                console.print("")
                console.rule(f"char {i}")
                console.print(json1[i-10:i+50])
                console.print(json2[i-10:i+50])
                quiet = True
            elif quiet and json1[i-5:i+1] == json2[i-5:i+1]:
                quiet = False

def print_parents(d: dict, s: str, parents=[], print=print):
    # print all entries at the same nested level together
    # to keep the output easier to read
    for k in d:
        if s in k:
            print(parents + [k])
    for k, v in d.items():
        if isinstance(v, dict):
            print_parents(v, s, parents+[k])
        elif s in str(v):
            print(parents + [k], "->", v)

@debug.command()
@click.argument('pattern', nargs=1, type=str)
@click.argument('task', nargs=1,
                type=click.Path(exists=True, resolve_path=False))
def key_parents(pattern, task):
    """
    Recurse into a task's fields and print the sequence(s) of nesting levels
    that lead to a key or value matching PATTERN.
    """
    with open(task) as f:
        data = json.load(f)

    console = rich.console.Console()
    with console.pager():
        print_parents(data, pattern, print=console.print)


@debug.command()
@click.option("--create/--inplace",
              help="Whether to overwrite (inplace) the task or create a new "
                   "one with the '.human' suffix.")
@click.argument("task", nargs=1,
                type=click.Path(exists=True, resolve_path=False))
def humanize(task, create: bool):
    """
    Reformat a task file to make it more human readable.
    """
    task = Path(task)
    with open(task) as f:
        data = json.load(f)
    if create:
        output_file = str(task.parent/task.stem) + ".human" + task.suffix
    else:
        output_file = task
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=3)

if __name__ == "__main__":
    cli()
