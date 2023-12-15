
#################################################
# Manifest                                      #
# --------                                      #
# Special classes:                              #
#   + Singleton                                 #
#                                               #
# Constants:                                    #
#   + NO_VALUE                                  #
#                                               #
# Iteration utilites:                           #
#   + terminating_types                         #
#   + flatten                                   #
#                                               #
# Misc:                                         #
#   + lenient_issubclass                        #
#   + relative_path                             #
#   + parse_duration_str                        #
#                                               #
# Operating with ParameterSet:                  #
#   + full_param_desc                           #
#   + fold_task_inputs                          #
#   + get_task_param                            #
#   + dfdiff                                    #
#   + ParameterComparison                       #
#                                               #
# Operating with records:                       #
#   + compute_input_symlinks                    #
#   + tasks_have_run                            #
#   + task_from_record                          #
#                                               #
# Debugging tasks                               #
#   + compare_task_serializations               #
#   + clear_task_cache                          #
#################################################


from __future__ import annotations

import logging
from collections.abc import Sequence, Generator
# For type hints only:
from sumatra.records import Record
from pathlib import Path
from typing import Union, Optional, Type, Literal, List, Tuple

from scityping.pydantic import BaseModel

# DEVELOPER WARNING: In contrast to smttask._utils, this module imports
# other smttask modules. Do not import it in another module
# WITHIN smttask, in order to avoid introducing an import cycle.
# Importing smttask.utils OUTSIDE smttask is perfectly safe, and is the recommended
# means by which to access methods defined both here AND in smttask._utils

# In contrast to smttask._utils, this module is not imported within smttask
# and therefore can make use of other smttask modules
from .config import config
from .base import Task, TaskInput, TaskDesc, instantiated_tasks
from .view.recordview import RecordView
from .view.recordstoreview import RecordStoreView

logger = logging.getLogger(__name__)

################################################################
## Combine all utils modules into a single public facing one ##
# utils is split into two modules to avoid import cycles
from ._utils import *
    # Imports: - NO_VALUE
    #          - lenient_issubclass
    #          - relative_path
    #          - parse_duration_str
    #          - sync_one_way

from .param_utils import dfdiff, ParameterComparison

#################
# Constants

# from ._utils import NO_VALUE  # Default value in function signatures

#################
# Operating with ParameterSets

def full_param_desc(obj, exclude_digests=False, *args, **kwargs) -> dict:
    """Call .dict recursively through task descriptions and Pydantic models.
    *args, **kwargs are passed on to `pydantic.BaseModel.dict`.

    This function was originally intended for the `task_types._run_and_record`
    method, to create a complete dictionary of parameters, but is no longer
    used.

    Recurses through:
        - Task (via .desc)
        - BaseModel
        - dict
        - Sequence
        - Generator

    Excludes:
        - 'reason' (Task): Not a parameter; recorded separately in Sumatra records
        - digest attributes (TaskInputs): (optional; by default these ARE included)
          Digest attributes are usually redundant with parameters.
          However, if parameters change, digest attributes store what is actually
          used to find precomputed tasks. Even if if there are no parameter
          changes, it is much easier to reconstruct task dependencies when the
          digest is readily available.

    Returns
    -------
    dict
    """

    if isinstance(obj, Task):
        desc = obj.desc
        return {'taskname': desc.taskname,
                'module':   desc.module,
                # 'reason':   desc.reason,
                'inputs': full_param_desc(desc.inputs)
                }
    elif exclude_digests and isinstance(obj, TaskInput):
        # Although a BaseModel, TaskInput has a special iterator to skip
        # digest entries
        return {k: full_param_desc(v) for k,v in obj}
    elif isinstance(obj, BaseModel):
        return {k: full_param_desc(v)
                for k,v in obj.dict(*args, **kwargs).items()}
    elif isinstance(obj, dict):
        return {k: full_param_desc(v) for k,v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(full_param_desc(v) for v in obj)
    elif (isinstance(obj, (Sequence, Generator))   # Iterable is too generic here
          and not isinstance(obj, (str, bytes))):
        return [full_param_desc(v) for v in obj]
    else:
        return obj

taskdesc_fields = set(k for k, v in TaskDesc.__fields__.items() if v.required)
assert 'inputs' in taskdesc_fields
def fold_task_inputs(pset):
    """
    In a hierarchical dictionary, such as the one created by `full_param_desc`,
    replace nested 'taskdesc' structures by their input dictionaries.

    Parameters
    ----------
    pset: dict | ParameterSet initializer
        Hierarchical set of parameters. Any valid initializer for `ParameterSet`
        is accepted.

    Returns
    -------
    ParameterSet
    """
    pset = config.ParameterSet(pset)  # Makes a shallow copy
    if isinstance(pset, dict) and taskdesc_fields <= set(pset.keys()):
        pset = pset['inputs']
    if isinstance(pset, dict):
        for k, v in pset.items():
            if isinstance(v, dict):
                pset[k] = fold_task_inputs(v)
    return pset

from .view.recordview import get_task_param

########
# Operating with records

def compute_input_symlinks(record: Record) -> List[Tuple[Path, Path]]:
    """
    Parameters
    ----------
    record: RecordView | Record

    Returns
    -------
    A dictionary of *(link location, link target)* pairs. Both *link location*
    and *link target* are `Path` objects and relative to the roots of the
    input and output data stores respectively.
    """
    try:
        task = Task.from_desc(record.parameters)
    except Exception:  # Tasks might raise any kind of exception
        raise ValueError("Task could not be recreated.")

    relpaths = task.relative_outputpaths
    rel_symlinks = {}  # A dictionary of (link location, link target) tuples

    # Get the recorded file path associated to each output name
    # Abort if any of the names are missing, or if they cannot be unambiguously resolved
    abort = False
    for nm, new_relpath in relpaths.items():
        # Although the computed output paths may differ from the
        # recorded ones, the variable names should still be the same
        # Get the output path associated to this name
        paths = [outdata.path for outdata in record.output_data
                      if nm in Path(outdata.path).stem.split(
                          '_', task.digest.count('_') + 1 )[-1]  # 'split' removes digest(s); +1 because there is always at least one '_' separating digest & name
                ]
        if len(paths) == 0:
            logger.debug(f"No output file containing {nm} is associated to record {record.label}.")
            abort = True
            break
        elif len(paths) >= 2:
            logger.debug(f"Record {record.label} has multiple output files containing {nm}.")
            abort = True
            break
        target_path = Path(paths[0])
        rel_symlinks[nm] = (new_relpath.with_suffix(target_path.suffix),
                            target_path)
    if abort:
        raise ValueError("Unable to determine output file names associated "
                         "with this task; see preceding debugging message.")

    return rel_symlinks.values()

    # # Compute the new symlinks
    # for nm, relpath in task.relative_outputpaths.items():
    #     outpath = output_paths[nm].resolve()  # Raises error if the path does not exist
    #     inpath = inroot/relpath.with_suffix(outpath.suffix)
    #     symlinks[inpath] = _utils.relative_path(inpath.parent, outpath)

def tasks_have_run(tasklist, warn=True):
    if not all([task.has_run for task in tasklist]):
        if warn:
            logger.warn(
                "Some of the tasks have either not be executed yet, or their "
                "output could not be found. If you have updated `smttask` or made "
                "changes to your code, it may be that digests have changed; if that "
                "is the case, run `smttask rebuild datastore` from the command line "
                "to recompute digests and rebuild the input data store.")
        return False
    else:
        return True

def task_from_record(record: Union[Record, str],
                     format: Literal["task", "taskdesc"]="task",
                     rsview: Optional[RecordStoreView]=None):
    """
    :param:record: Identifier allowing to define a unique record.  May be:
      - A `Record` instance
      - A record label
        The record label may be partial, in which case the latest record is used.
        Note however that selecting with partial labels is slower.
    :param:format: Whether to return a Task or TaskDesc. If only the latter is
       needed, skipping the instantiation of the Task can bring substantial
       speedup.
    :param:rsview: A `RecordStoreView` instance. If not provided, one is
       retrieved with `RecordStoreView()`, which will look for the first Sumatra
       project containing the current directory.
        
    .. warning:: No error is raised if a partial label is used and would match
       multiple records. This may be desired (e.g. to get the latest record
       matching a digest) but it may also hide logic errors.
    """
    if not isinstance(format, str) or format not in {"task", "taskdesc"}:
        raise ValueError("`format` must be one of 'task', 'taskdesc'. "
                         f"Received '{format}'.")
    if isinstance(record, (Record, RecordView)):
        if rsview is not None:
            logger.warning("With `record` provided, `rsview` is ignored.")
    elif isinstance(record, str):
        if rsview is None:
            rsview = RecordStoreView()
        try:
            record = rsview.get(record)
        except KeyError:
            record = next(iter(rsview.filter.label(record)))
    else:
        raise TypeError("`record` should be either a Record or a string label. "
                        f"Received: {record}")
    # We provide the option to return TaskDesc because instantiating an entire
    # Task just to reserialize it can be very wasteful.
    taskdesc = TaskDesc.load(record.parameters)
    if format == "task":
        return Task.from_desc(taskdesc)
    else:
        return taskdesc

########
# Debugging tasks

def compare_task_serializations(task1: Union['path-like', Task],
                                task2: Union['path-like', Task]) -> "DataFrame":
    """
    This function is especially useful for tracking down why a task doesn't
    serialize consistently (and therefore recomputes instead of reusing a
    previous stored result).
    Standard approach is to run the task-producing script twice, writing the
    output to two files, and then call this function passing the file names.
    It will return a pandas DataFrame showing which parameters are different
    between the two serialized tasks.

    :param:task1: Task, absolute path or relative path (from CWD)
        Paths should point to a file created with `Task.save()`.
    :param:task2: Same as `task1`
    """
    from pathlib import Path
    import json

    def parse_task_json(task):
        if isinstance(task, Task):
            # Ensure we compare against a serialized/deserialized version
            return json.loads(task.desc.json())
        else:
            with open(task) as f:
                s = json.load(f)
            return s
    def get_task_name(task):
        if isinstance(task, Task):
            return f"{task.name} <{id(task)}>"
        else:
            return Path(task).stem

    paramset1 = parse_task_json(task1)
    paramset2 = parse_task_json(task2)
    name1 = get_task_name(task1)
    name2 = get_task_name(task2)

    return dfdiff(paramset1, paramset2, name1=name1, name2=name2)

def clear_task_cache(*task_types: Union[Type[Task],Task,str]):
    """
    Clear the Task cache. This forces tasks to be reinitialized.
    
    Normally, attempting to reinitialize a task with the same parameters as
    previously simply returns the previously created instance. Even with the
    option `recompute=True`, changes to the task's code are still ignored.This
    is undesirable during an interactive debugging session, where we may want to
    change and test the Task definition.
    
    One or many task types may be specified, in which case only those types
    are cleared; otherwise, all Tasks are cleared.
    """
    if len(task_types) == 0:
        logger.debug("Clearing all tasks.")
        instantiated_tasks.clear()
    else:
        # Task cache matches by Task name, so we do the same
        task_names = {tt.taskname() if isinstance(tt, Task) else
                      tt.taskname() if lenient_issubclass(tt, Task) else
                      tt
                      for tt in task_types}
        if not all(isinstance(tn, str) for tn in task_names):
            raise TypeError("`task_types` must all be Tasks or strings; "
                            f"received: {', '.join(task_types)}.")
        logger.debug(f"Clearing tasks {','.join(task_names)}.")
        for inst_tasks in instantiated_tasks.values():
            for k in [k for k, v in inst_tasks.items()
                      if v.name in task_names]:
                del inst_tasks[k]
                
# def matching_serializers(task, value) -> List[Type]:
#     """
#     Return all JSON-serializable types associated to `task` that can serialize
#     `value`. Types are returned in order, so the first type is the one that
#     would be used.
#     """
#     return [T for T in task.__config__.json_encoders if isinstance(value, T)]
    
