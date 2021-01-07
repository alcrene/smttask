import logging
from collections.abc import Sequence, Generator
# For type hints only:
from sumatra.records import Record
from pathlib import Path
from typing import Any, Union, Sequence, List, Tuple

from pydantic import BaseModel
import mackelab_toolbox as mtb
import mackelab_toolbox.utils

# DEVELOPER WARNING: In contrast to smttask._utils, this module imports
# some of the base smttask types. If you import it in another module
# WITHIN smttask, make sure you are not introducing an import cycle.
# Importing smttask.utils OUTSIDE smttask is perfectly safe, and is the
# recommended means by which to access methods defined here and in smttask._utils

# In contrast to smttask._utils, this module is not imported within smttask
# and therefore can make use of other smttask modules
from .config import config
from .base import Task, TaskInput, TaskDesc
from .view.recordview import RecordView

logger = logging.getLogger(__name__)

################################################################
## Combine both utils modules into a single public facing one ##
# utils is split into two modules to avoid import cycles
from ._utils import *  # Imports: - lenient_issubclass
                       #          - relative_path

#################
# Constants

NO_VALUE = mtb.utils.sentinel("value not provided")  # Default value in function signatures

#################
# Operating with ParameterSets

def full_param_desc(obj, exclude_digests=False, *args, **kwargs) -> dict:
    """Call .dict recursively through task descriptions and Pydantic models.
    *args, **kwargs are passed on to `pydantic.BaseModel.dict`.

    This function is mainly intended for the `task_types._run_and_record`
    method, to create a complete dictionary of parameters.

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
    neplace nested 'taskdesc' structures by their input dictionaries.

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

def get_task_param(obj, name: Union[str, Sequence], default: Any=NO_VALUE):
    """
    A convenience function for retrieving values from nested parameter sets
    or tasks. Attributes of object types are accessed with slightly syntax,
    and this gets especially cumbersome with nested parameters. This function
    is applied recursively, at each level selecting the appropriate syntax
    depending on the value's type.

    Parameters
    ----------
    obj: dict | Task | RecordView | serialized ParameterSet | task desc | namespace
        The object from which we want to retrieve the value of a particular
        key / attribute.

        dict
            Return `obj[name]`.

        Task
            Return `obj.name`

        RecordView
            Return `ParameterSet(obj.parameters)[name]`

        serialized ParameterSet
            Return `ParameterSet(obj)[name]`

        task desc
            Return `obj['inputs'][name]`   (unless `obj[name]` exists)

        namespace (e.g. `~types.SimpleNamespace`)
            Return `obj.name`

    name: str | Sequence
        The key or attribute name to retrieve. Nested attributes can be
        specified by listing each level separated by a period.
        Multiple names can be specified by wrapping them in a list or tuple;
        they are tried in sequence and the first attribute found is returned.
        This can be used to deal with tasks that may have differently named
        equivalent arguments.
    default: Any
        If the attributed is not found, return this value.
        If not specified, a KeyError is raised.

    Returns
    -------
    The value matching the attribute, or otherwise the value of `default`.

    Raises
    ------
    KeyError:
        If the key `name` is not found and `default` is not set.
    """
    if not isinstance(name, (str, bytes)) and isinstance(name, Sequence):
        for name_ in name:
            try:
                val = get_task_param(obj, name_, default=default)
            except KeyError:
                pass
            else:
                return val
        # If we made it here, none of the names succeeded
        raise KeyError(f"None of the names among {name} are parameters of "
                       f"the {type(obj)} object.")
    if "." in name:
        name, subname = name.split(".", 1)
    else:
        subname = None
    if isinstance(obj, RecordView):
        obj = obj.parameters
    if isinstance(obj, str):
        obj = config.ParameterSet(obj)
        # TODO?: Fall back to Task.from_desc if ParameterSet fails ?
    if isinstance(obj, Task):
        try:
            val = getattr(obj, name)
        except AttributeError as e:
            if default is not NO_VALUE:
                val = default
            else:
                raise KeyError from e
    elif isinstance(obj, dict):
        try:
            if "taskname" in obj and name != "inputs":
                assert "inputs" in obj
                val = obj["inputs"][name]
            else:
                val = obj[name]
        except KeyError as e:
            if default is not NO_VALUE:
                val = default
            else:
                raise KeyError from e
    else:
        # SimpleNamespace ends up here.
        # As good a fallback as any to ensure something is assigned to `val`
        val = getattr(obj, name)
    if subname is not None:
        val = get_task_param(val, subname, default)
    return val

from mackelab_toolbox.parameters import dfdiff, ParameterComparison

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

    relpaths = task.outputpaths
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
    # for nm, relpath in task.outputpaths.items():
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
