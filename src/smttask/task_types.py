"""
This module defines the different types of Tasks, of which there are currently
four:

    - RecordedTask
    - RecordedIterativeTask
    - MemoizedTask
    - UnpureMemoizedTask

The purpose of each type, and their interface, are documented here. However, to
construct them, it is highly recommended to use the identically named
decorators in `smttask.decorators`.
"""

import sys
import os
import re
# from warnings import warn
import logging
import time
import json
from copy import deepcopy
from collections import deque, namedtuple
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Union, Callable, Dict, Tuple

import pydantic.parse

from sumatra.core import TIMESTAMP_FORMAT, STATUS_FORMAT
from sumatra.datastore.filesystem import DataFile
from sumatra.programs import PythonExecutable

from .base import Task, NotComputed, EmptyOutput, TaskExecutionError
from .config import config
# from .typing import PlainArg
# from . import utils

from numbers import Number
from numpy import ndarray
PlainArg = (Number, str, ndarray)

# project = config.project

# TODO: Include run label in project.datastore.root

logger = logging.getLogger(__name__)

__all__ = ['RecordedTask', 'RecordedIterativeTask',
           'MemoizedTask', 'UnpureMemoizedTask']

FoundFiles = namedtuple('FoundFiles', ['outputpaths', 'is_partial', 'param_update'])

class RecordedTask(Task):

    def __init__(self, arg0=None, *, reason=None, **taskinputs):
        super().__init__(arg0, reason=reason, **taskinputs)
        self.outext = ""  # If not empty, should start with period
        if reason is None and config.record:
            logger.warning(f"Task {self.name} was not given a 'reason'.")

    # TODO: How to merge this with _outputpaths_gen ?
    def find_saved_outputs(self):
        """
        Return the list of paths where one would find the output from a
        previous run. Files are not guaranteed to exist at those locations,
        so opening the return paths should be guarded by a try...except clause.

        Basic RecordedTasks have no mechanism for continuing partial
        calculations, so they always return ``is_partial=False`` and
        ``param_update=None``.

        Returns
        -------
        FoundFiles:
            - dict of {output name: path}
            - is_partial: False
            - param_update: None
                `is_partial` and `param_update` are included for API consistency.

        Raises
        ------
        FileNotFoundError:
            If not saved outputs are found
        """
        inroot = Path(config.project.input_datastore.root)
        searchdir = inroot/self.name

        ## Create a regex that will identify matching outputs, and extract their
        #  variable name
        hashed_digest = self.hashed_digest
        re_outfile = f"{re.escape(hashed_digest)}_([a-zA-Z0-9]*).json$"
        ## Loop over on-disk file names, find matching files and extract iteration variable name
        outfiles = {}
        for fname in os.listdir(searchdir):
            m = re.match(re_outfile, fname)
            if m is not None:
                assert fname == m[0]
                varname = m[1]
                outfiles[varname] = searchdir/fname
        ## If there is a file for each output variable, return the paths:
        if all(attr in outfiles for attr in self.Outputs._outputnames_gen(self)):
            return FoundFiles(outputpaths=outfiles,
                              is_partial=False,
                              param_update=None)
        ## Otherwise return None
        else:
            raise FileNotFoundError

    def run(self, cache=None, recompute=False, record=None, reason=None, record_store=None):
        """
        To completely disable recording, use `config.disable_recording = True`.

        Parameters
        ----------
        cache: bool
            Set to True to enable in-memory caching. If unspecified, read from
            class' default, and if that is also not set, from `config`.
        recompute: bool
            Force task to execute, even if it is cached.
            (default: False)
        record: bool
            Set to False to disable recording to Sumatra database. If unspecified, read from
            `config` (default config: True).
        reason: str (Optional)
            Override the reason recorded in the task description.
        """
        if cache is None:
            cache = self.cache if self.cache is not None else config.cache_runs
        if reason is not None:
            self.reason = reason
        inroot = Path(config.project.input_datastore.root)
        outputs = None

        # First try to load pre-computed result
        continue_previous_run = False
        if self._run_result is NotComputed and not recompute:
            # First check if output has already been produced
            _outputs = {}
            try:
                found_files = self.find_saved_outputs()
                for varnm, path in found_files.outputpaths.items():
                    _outputs[varnm] = self._parse_output_file(path)
            except FileNotFoundError:
                pass
            else:
                self.logger.info("Loading result of previous run from disk.")
                # Only assign to `outputs` once all outputs are loaded successfully
                # outputs = tuple(_outputs)
                outputs = self.Outputs(**_outputs, _task=self)
                if found_files.is_partial:
                    # We still need to run the task, but we can start from a
                    # partial computation
                    # We use `update_params` to change `taskinputs`
                    orig_digest = self.digest  # For the assert
                    new_inputs = found_files.param_update(outputs)
                    assert set(new_inputs) <= set(self.taskinputs.__fields__)
                    self.taskinputs = self.Inputs.parse_obj(
                        {**self.dict(), **new_inputs})
                    # for k,v in new_inputs.items():
                    #     setattr(self.taskinputs, k, v)
                    assert self.digest == orig_digest
                    # Now that the info from the previous run has been used to
                    # update `taskinputs`, we delete `outputs` to indicate that
                    # they still need to be computed
                    outputs = None
                    continue_previous_run = True

        elif not recompute:
            self.logger.info("Loading memoized result.")
            outputs = self._run_result

        if outputs is None:
            # We did not find a previously computed result, so run the task
            if recompute:
                self.logger.info(f"Recomputing task.")
            elif continue_previous_run:
                self.logger.info("Continuing from a previous partial result.")
            else:
                self.logger.info(
                    "No previously saved result was found; running task.")
            outputs = self._run_and_record(record, record_store)

        if cache and self._run_result is NotComputed:
            self._run_result = outputs
            self.logger.debug(f"Memoized task result.")

        return outputs.result

    def _run_and_record(self, record: bool=None, record_store=None):
        # Remark: Refer to sumatra.decorators.capture for a similar pattern
        # DEVNOTE: status values should be limited to those defined in the
        #    `style_map` variable of sumatra.web.templatetags.filters:labelize_tag
        #    Otherwise the smt web interface returns an exception
        if record is None:
            record = config.record
        input_data = [input.generate_key()
                      for input in self.input_files]
        # Module where task is defined
        # Decorators set the _module_name attribute explicitely, because with the
        # dynamically created class, the `type(self)` method gets the module wrong
        module_name = getattr(self, '_module_name', type(self).__module__)
        module = sys.modules[module_name]
        old_status='pre_run'
        status='running'
        self.logger.debug(f"Status: '{old_status}' → '{status}'.")
        if record:
            if record_store:
                # Update config to use a different record store
                import tempfile
                import shutil
                from sumatra.projects import _get_project_file
                from sumatra.recordstore import DefaultRecordStore
                # `record_store` may specify a new location – in this case,
                # ensure that parent directories exist
                self.logger.debug("Configuring task to use the non-default "
                                  f"record store at location {record_store}.")
                Path(record_store).parent.mkdir(parents=True, exist_ok=True)
                config.project.record_store = type(config.project.record_store)(record_store)
                # Problem: We can change the attributes of the Sumatra project
                #   project in place, but when Sumatra saves the record, it
                #   updates the .smt/project file such that the value of
                #   `record_store` becomes the default for all future Task
                #   executions.
                # Solution: Create a throwaway project directory, and also
                #   change project.path to point there. This doesn't change
                #   how Sumatra behaves (project values are already loaded
                #   into runtime memory at this point), but any attempts by
                #   Sumatra to permanently change the project configuration
                #   are redirected to this throwaway directory, and discarded
                #   when we exit this function.
                tmpproject_dir = tempfile.mkdtemp()
                tmpproject_file = _get_project_file(tmpproject_dir)
                Path(tmpproject_file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_get_project_file(config.project.path), tmpproject_file)
                config.project.path = tmpproject_dir
                self.logger.debug("Created throwaway project directory "
                                  f"at location {tmpproject_dir}.")
            # Append a few chars from digest so simultaneous runs don't
            # have clashing labels
            # Probabilities of having a collision after 1000 times, if we run n tasks simultaneously each time
            # (Only tasks run simultaneously may clash, b/c of the timestamp)
            #   Digest length  |  P_coll (12 tasks)  |  (24 tasks)
            #     4            |  63.47              |  98.52%
            #     6            |   0.39%             |   1.63%
            #     8            |   0.0015%           |   0.0064%
            label = datetime.now().strftime(TIMESTAMP_FORMAT) + '_' + self.digest[:6]
            # Sumatra will still work without wrapping parameters with
            # ParameterSet, but that is the format it expects. Moreover,
            # doing this allows the smtweb interface to display parameters.
            # NOTE: Sumatra doesn't recognize the Pydantic-aware types, and
            #       serializes a ParameterSet more or less by taking its str.
            #       Thus to make the recorded parameters Sumatra-safe, we use
            #       our own JSON serializer, and parse the result back into
            #       a ParameterSet – this results in a ParameterSet containing
            #       only JSON-valid entries.
            self.logger.debug("Parsing parameters from Task description.")
            parameter_str = self.desc.json(indent=2)
            try:
                # parameters=config.ParameterSet(utils.full_param_desc(self))
                parameters = config.ParameterSet(json.loads(parameter_str))
            except Exception as e:
                # If creation of ParameterSet fails, save parameters as-is
                self.logger.debug("Creation of a ParameterSet failed; saving as "
                             "JSON string. The smtweb will not be able to "
                             "browse/filter parameter values.")
                parameters = parameter_str
            self.logger.debug(f"Creating a new Task record with label '{label}'...")
            smtrecord = config.project.new_record(
                parameters=parameters,
                input_data=input_data,
                script_args=type(self).__name__,
                executable=PythonExecutable(sys.executable),
                main_file=module.__file__,
                reason=self.reason,
                label=label
                )
            self.logger.debug("Task record created.")
            smtrecord.add_tag(STATUS_FORMAT % status)
            self.logger.debug(f"Adding record to Sumatra project '{config.project.name}'...")
            config.project.add_record(smtrecord)
            self.logger.debug("Record added to project.")
            self.logger.debug(f"Task execution start time: {datetime.now()}")
            start_time = time.time()
        elif not config.allow_uncommitted_changes:
            # Check that changes are committed. This is normally done in new_record().
            # See sumatra/projects.py:Project.new_record
            self.logger.debug("Task will not be recorded but config states to still check for uncommitted changes.")
            repository = deepcopy(config.project.default_repository)
            working_copy = repository.get_working_copy()
            config.project.update_code(working_copy)
            self.logger.debug("No uncommited change detected.")
        outputs = EmptyOutput(status=status)
        try:
            self.logger.debug("Executing the task’s code...")
            run_result = self._run(**dict(self.load_inputs()))
                # We don't use .dict() here, because that would dictifiy all nested
                # BaseModels, which would then be immediately recreated from their dicts
            self.logger.debug("Finished executing task’s code.")
            old_status = status
            status = "finished"
            self.logger.debug(f"Status: '{old_status}' → '{status}'.")
            self.logger.debug("Parsing task results...")
            outputs = self.Outputs.parse_result(run_result, _task=self)
        except (KeyboardInterrupt, SystemExit):
            self.logger.debug("Caught KeyboardInterrupt")
            old_status = status
            status = "killed"
            self.logger.debug(f"Status: '{old_status}' → '{status}'.")
            outputs = EmptyOutput(status=status)
            # When executing with multiprocessing, the mother process also
            # receives the interrupt and kills the spawned process.
            # The only statements that *are* executed are those within exception
            # handlers. So we need to reraise, to allow the unique_process_num
            # context manager in smttask.ui._run_task to clean up
            raise KeyboardInterrupt
        except Exception as e:
            old_status = status
            status = "crashed"
            self.logger.debug(f"Status: '{old_status}' → '{status}'.")
            if record:
                if smtrecord.outcome != "":
                    smtrecord.outcome += "\n" + repr(e)
                else:
                    smtrecord.outcome = repr(e)
            outputs = EmptyOutput(status=status)
            if config.on_error == 'raise':
                # NB: The following may be the most logical:
                #         raise TaskExecutionError(self) from e
                #     but while it shows the traceback which actually caused the error,
                #     it is no longer accessible to the debugger.
                #     This is why we use `with_traceback` below. See https://stackoverflow.com/questions/1603940/how-can-i-modify-a-python-traceback-object-when-raising-an-exception
                ei = sys.exc_info()
                raise TaskExecutionError(self).with_traceback(ei[2])
            else:
                traceback.print_exc()
        finally:
            # We place this in a finally clause, instead of just at the end, to
            # ensure this is executed even after a SIGINT during multiprocessing.
            if record:
                smtrecord.add_tag(STATUS_FORMAT % status)
                smtrecord.duration = time.time() - start_time
                if getattr(outputs, 'outcome', ""):
                    if smtrecord.outcome != "":
                        smtrecord.outcome += "\n"
                    if isinstance(outputs.outcome, str):
                        smtrecord.outcome += outputs.outcome
                    elif isinstance(outputs.outcome, (tuple, list)):
                        smtrecord.outcome += "\n".join(
                            (str(o) for o in outputs.outcome))
                    else:
                        self.logger.warning("Task `outcome` should be either a string "
                                    "or tuple of strings. Coercing to string.")
                        smtrecord.outcome += str(outputs.outcome)
            if len(outputs) == 0:
                self.logger.warning("No output was produced.")
            elif record and status == "finished":
                self.logger.debug("Saving output...")
                smtrecord.add_tag(STATUS_FORMAT % status)
                realoutputpaths = outputs.write()
                if len(realoutputpaths) != len(outputs):
                    self.logger.warning("Something went wrong when writing task outputs. "
                         f"\nNo. of outputs: {len(outputs)} "
                         f"\nNo. of output paths: {len(realoutputpaths)}")
                    if smtrecord.outcome != "":
                        smtrecord.outcome += "\n"
                    smtrecord.outcome += ("Error while writing to disk: possibly "
                                          "missing or unrecorded data.")
                else:
                    old_status = status
                    status = "finished"
                    self.logger.debug(f"Status: '{old_status}' → '{status}'.")
                # NB: `path` is absolute. `path` & `data_store.root` may include a symlink, so we need to resolve them to get the right anchor for a relative path
                outroot = Path(config.project.data_store.root).resolve()
                # Convert to relative output paths in a way which ensures we don't error out just before writing if there is an error
                relativeoutputpaths = []  # Fill list one at a time, so that we use the fallback path only for those which fail the conversion to relative (in theory, should be all or none, but also in theory, there should be no errors here)
                for path in realoutputpaths:
                    try:
                        relpath = Path(path).resolve().relative_to(outroot)
                    except Exception:   # (Normally this should be ValueError, but since we want to make sure we write out the results, we catch any exception. The only thing we want to let pass through are interrupt signals)
                        # For some unexpected reason, computing a relative path failed. Fall back to using the path iself; it might not be fully correct, but should provide enough info to allow the user to find the file
                        relpath = path
                    relativeoutputpaths.append(relpath)
                smtrecord.output_data = [
                    DataFile(str(relpath), config.project.data_store).generate_key()
                    for relpath in relativeoutputpaths]
                self.logger.debug(f"Task {status}")
                smtrecord.add_tag(STATUS_FORMAT % status)
            if record:
                config.project.save_record(smtrecord)
                config.project.save()
                self.logger.debug("Saved record")
                if record_store:
                    # Remove the directory with throwaway project file
                    self.logger.debug("Removing throwaway project directory "
                                      f"at location '{tmpproject_dir}'.")
                    shutil.rmtree(tmpproject_dir, ignore_errors=True)

        return outputs

class RecordedIterativeTask(RecordedTask):
    """
    A specialized `RecordedTask`, meant for tasks which are repeatedly applied
    on their own output. Examples would be integrating an ODE by repeatedly
    applying its discretized update equations, and optimization an objective
    function by an iterative procedure (e.g. via gradient descent, annealing,
    or genetic algorithm).

    The class variable `_iteration_parameter` is the name of the parameter which
    increments by 1 every time the task is applied. It must match a parameter
    in the Task inputs which has type `int`.

    The advantage over a normal `RecordedTask` is that when searching for
    previous runs, it will find output files that were produced with the same
    parameters but fewer iterations. This allows one to e.g. restart a fit or
    simulation from a previous stop point.

    .. Warning:: This task will only attempt to load the most advanced
       recorded result. So for example, if there are both results for 10 and
       20 iterations on disk, and we ask for >=20 iterations, the task will
       only attempt to load the latter. Normally this is the desired behaviour,
       however if the records fo 20 iterations are corrupted or partial, the
       task will not then attempt to load the results for 10 iterations.
       Rather, it would start again from 0.
    """

    _iteration_parameter: str
    _iteration_map: Dict[str, str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._iteration_parameter not in self.Inputs._unhashed_params:
            raise RuntimeError(
                f"The iteration parameter '{self._iteration_parameter}' for "
                f"for task '{self.name}' was not added to the list of unhashed "
                f"params in {self.name}.Inputs. This is required to match "
                "previous runs with different numbers of iterations.")
        elif self._iteration_parameter != self.Inputs._unhashed_params[0]:
            raise RuntimeError(
                f"The iteration parameter '{self._iteration_parameter}' must "
                f"be the first element in {self.name}.Inputs._unhashed_params.")

    def find_saved_outputs(self) -> Tuple[Dict[str,Path], bool, Union[Callable,None]]:
        """
        Return the list of paths where one would find the output from a
        previous run. Files are not guaranteed to exist at those locations,
        so opening the return paths should be guarded by a try...except clause.

        Returns
        -------
        FoundFiles:
            outputpaths: dict of {output name: path}
                Location of the found output files.
            is_partial: bool
                True if the returned output corresponds to a run with fewer iterations.
            param_update: Callable[[TaskOutput], dic] | None
                If `is_partial` is True, calling this function on the outputs
                returns a dict mapping input parameter names to their updated values.
                If `is_partial` is False, the function is undefined and will
                typically be set to None

        Raises
        ------
        FileNotFoundError:
            If not saved outputs are found
        """
        inroot = Path(config.project.input_datastore.root)
        searchdir = inroot/type(self).__name__

        ## Create a regex that will identify outputs produced by the same run,
        #  and extract their iteration step and variable name
        hashed_digest = self.hashed_digest
        iterp_name = self._iteration_parameter
        re_outfile = f"{re.escape(hashed_digest)}__{re.escape(iterp_name)}_(\d*)_(.*).json$"
            #          ^------- base.make_digest ---------------------^ ^-TaskOutputs.output_paths-^
        ## Loop over on-disk file names, find matching files and extract iteration number and variable name
        outfiles = {}
        for fname in os.listdir(searchdir):
            m = re.match(re_outfile, fname)
            if m is not None:
                assert fname == m[0]
                itervalue, varname = m.groups()
                # if not itervalue.isdigit():
                #     warn("The iteration step parsed from the output is not an"
                #          "integer. It will nevertheless be coerced to int.\n"
                #          f"Iteration: {itervalue}\nFile name: {fname}")
                itervalue = int(itervalue)
                if itervalue not in outfiles:
                    outfiles[itervalue] = {}
                outfiles[itervalue][varname] = searchdir/fname
        ## Check if there is an output matching the desired iterations
        iterp_val = getattr(self.taskinputs, iterp_name)
        if (iterp_val in outfiles
            and all(attr in outfiles[iterp_val] for attr in self.Outputs._outputnames_gen(self))):
            self.logger.debug(f"Found output from a previous run matching these parameters.")
            return FoundFiles(outputpaths=outfiles[iterp_val],
                              is_partial=False,
                              param_update=None)
        ## There is no exact match.
        #  Iterate in reverse order, return first complete set of files
        for n in reversed(sorted(outfiles)):
            # Skip any paths which may have more iterations
            if n > iterp_val:
                continue
            # Check that all required outputs are there
            if all(attr in outfiles[n] for attr in self.Outputs._outputnames_gen(self)):
                self.logger.debug(f"Found output from a previous run matching these "
                             f"parameters but with only {n} iterations.")
                def param_update(outputs):
                    return {in_param: getattr(outputs, out_param)
                            for out_param, in_param in self._iteration_map.items()}
                return FoundFiles(outputpaths=outfiles[n],
                                  is_partial=True,
                                  param_update=param_update)
        ## If we reached this point, there is no existing result on disk.
        raise FileNotFoundError

class MemoizedTask(Task):
    """
    Behaves like a Task, in particular with regards to computing descriptions
    and digests of composited tasks.
    However the output is not saved to disk and a sumatra record is not created.
    The intention is for tasks which are cheap to compute, and thus for which it
    does not make sense to store the output. A prime example would be the output
    of a random number generator, for which it is much more efficient to store a
    function, a random seed and some parameters.
    """
    def __init__(self, arg0=None, *, reason=None, **taskinputs):
        """
        Parameters
        ----------
        arg0: ParameterSet-like
            ParameterSet, or something which can be cast to a ParameterSet
            (like a dict or filename). The result will be parsed for task
            arguments defined in `self.inputs`.
        **taskinputs:
            Task parameters can also be specified as keyword arguments,
            and will override those in :param:arg0.
        reason: None | str
            Ignored because we aren't recording into a Sumatra db.
            Included for compability with Task.
        """
        super().__init__(arg0, reason=reason, **taskinputs)

    def run(self, cache=None, recompute=False, record=None, reason=None, record_store=None):
        if cache is None:
            cache = self.cache if self.cache is not None else True
        if record:
            logger.warning(f"Task {self.name} is a `MemoizedTask`, and therefore "
                           "never recorded; passing `record=True` has no effect.")
        if reason is not None:
            self.reason = reason
        if self._run_result is NotComputed or recompute:
            input_data = [input.generate_key() for input in self.input_files]
            module = sys.modules[type(self).__module__]
            if not config.allow_uncommitted_changes:
                # Check that changes are committed. This is normally done in new_record().
                # See sumatra/projects.py:Project.new_record
                repository = deepcopy(config.project.default_repository)
                working_copy = repository.get_working_copy()
                config.project.update_code(working_copy)

            self.logger.info(f"Running task in memory.")
            # We don't use .dict() here, because that would dictifiy all nested
            # BaseModels, which would then be immediately recreated from their dict
            try:
                run_result = self._run(**dict(self.load_inputs()))
            except Exception as e:
                # See comment above in _run_and_record
                ei = sys.exc_info()
                raise TaskExecutionError(self).with_traceback(ei[2].tb_next)
            output = self.Outputs.parse_result(run_result,  _task=self)
            if cache:
                self._run_result = output
                self.logger.debug(f"Memoized task result.")
        else:
            output = self._run_result
            self.logger.info("Loading memoized result.")
        return output.result

class UnpureMemoizedTask(MemoizedTask):
    """
    A Task whose output does *not* only depend on the inputs (and thus is not
    a pure function). An UnpureTask cannot be recorded, because its digest is
    computed from its output. For the same reason, it is always memoized and
    should never be cleared. (Since there may be use cases for clearing during a
    debugging session, it is not explicitely forbidden, but doing so will log a
    message at the 'error' criticality level.)

    To motivate the use of such a Task, consider the following set of operations:

    TaskA (s: string) -> Return the list of entries in a database containing `s`.
    TaskB (l: list|TaskA) -> Return a set of statistics for those entries.

    TaskA is unpure: it depends on `s`, but also on the contents of the database.
    TaskB is pure, and one can write a reproducible workflow by explicitely
    specifying all the entries listed in `l`. But that would be extremely
    verbose, and it would hide the origin of those entries. The definition
    above, in terms of the output of TaskA, is clearer and more concise.

    It is even more desirable to encode such a task sequence if the database
    changes rarely, for example only when new experiments are performed.
    However, if a normal Task is used to encode TaskA, then updating the
    database would not change the Task's digest, and thus the statistics
    would not be recomputed.
    What we want therefore is to define and display TaskA in terms of its inputs
    (as with a normal Task), but compute its digest from its outputs.

    Because an UnpureMemoizedTask is not recorded, it is also not meaningful to
    specify a `reason` argument.

    .. Important:: `UnpureMemoizedTask` still performs in-memory caching
       (memoization). This means that non-input dependencies (in the example
       above, the contents of the database) must not change during workflow
       execution.
       Similarly, `UnpureMemoizedTask` should still not have side-effects.
       Otherwise the result of tasks may depend on their execution order, which
       is undefined.
    """
    __slots__ = ('_memoized_run_result',)
    cache = True

    def __init__(self, arg0=None, **taskinputs):
        """
        Parameters
        ----------
        arg0: ParameterSet-like
            ParameterSet, or something which can be cast to a ParameterSet
            (like a dict or filename). The result will be parsed for task
            arguments defined in `self.inputs`.
        **taskinputs:
            Task parameters can also be specified as keyword arguments,
            and will override those in :param:arg0.
        """
        if 'reason' in taskinputs:
            raise TypeError("Specifying a `reason` for UnpureTask is not "
                            "meaningful, and therefore disallowed.")
        object.__setattr__(self, '_memoized_run_result', None)
        super().__init__(arg0, **taskinputs)

    def _get_run_result(self, recompute=False):
        if recompute or self._memoized_run_result is None:
            self.logger.info(f"Running task.")
            try:
                run_result = self._run(**dict(self.load_inputs()))
            except Exception as e:
                raise TaskExecutionError(self) from e
            result = self.Outputs.parse_result(run_result, _task=self)
            if result.digest != self._memoized_run_result:
                if self._memoized_run_result is not None:
                    logger.warning("Digest has changed for task {self.name}.")
                object.__setattr__(self, '_memoized_run_result', result)
        else:
            self.logger.info("Loading memoized result.")
        return self._memoized_run_result

    def run(self, cache=None, recompute=False, record=None,reason=None, record_store=None):
        if cache == False or self.cache != True:
            raise ValueError("The implementation of UnpureMemoizedTask "
                             "requires caching, and so it cannot be run with "
                             "``cache=False``.")
        if record:
            logger.warning(f"Task {self.name} is an `UnpureMemoizedTask`, and therefore "
                           "never recorded; passing `record=True` has no effect.")
        if reason is not None:
            self.reason = reason
        return self._get_run_result(recompute).result

    def clear(self):
        super().clear()
        self.logger.error(f"Task has cleared its memoization cache. "
                     "Since an UnpureMemoizedTask does not guarantee that its "
                     "result is reproducible, it is strongly advised not to "
                     "do this.")

    @property
    def hashed_digest(self):
        return self._get_run_result().hashed_digest
    @property
    def unhashed_digests(self):
        return self._get_run_result().unhashed_digests
    @property
    def digest(self):
        return self._get_run_result().digest

    def __hash__(self):
        return hash(self._get_run_result())
