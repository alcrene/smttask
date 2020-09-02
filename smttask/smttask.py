"""
This module defines the different types of Tasks, of which there are currently
three:

    - RecordedTask
    - RecordedIterativeTask
    - MemoizedTask

The purpose of each type, and their interface, are documented here. However, to
construct them, it is highly recommended to use the identically named
decorators in `smttask.decorators`.
"""

import sys
import os
import re
from warnings import warn
import logging
import time
from copy import deepcopy
from collections import deque, namedtuple
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Union, Callable, Dict, Tuple

from sumatra.core import TIMESTAMP_FORMAT
from sumatra.datastore.filesystem import DataFile
from sumatra.programs import PythonExecutable
# import mackelab_toolbox as mtb
# import mackelab_toolbox.iotools

import pydantic.parse

from .base import config, ParameterSet, Task, NotComputed
from .typing import PlainArg
from . import utils

# project = config.project

# TODO: Include run label in project.datastore.root

logger = logging.getLogger(__name__)

__all__ = ['Task', 'MemoizedTask']

FoundFiles = namedtuple('FoundFiles', ['outputpaths', 'is_partial', 'param_update'])

class RecordedTask(Task):

    def __init__(self, arg0=None, *, reason=None, **taskinputs):
        super().__init__(arg0, reason=reason, **taskinputs)
        self.outext = ""  # If not empty, should start with period
        if reason is None:
            warn(f"Task {self.name} was not given a 'reason'.")
        self.reason = reason

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
        searchdir = inroot/type(self).__name__

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

    def run(self, cache=None, recompute=False, record=None):
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
        """
        if cache is None:
            cache = self.cache if self.cache is not None else config.cache_runs
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
                    # Next line copied from pydantic.main.parse_file
                    _outputs[varnm] = pydantic.parse.load_file(
                        inroot/path,
                        proto=None, content_type='json', encoding='utf-8',
                        allow_pickle=False,
                        json_loads=self.Outputs.__config__.json_loads)
            except FileNotFoundError:
                pass
            else:
                logger.info(
                    type(self).__qualname__ + ": loading result of previous "
                    "run from disk.")
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
                    for k,v in new_inputs.items():
                        setattr(self.taskinputs, k, v)
                    assert self.digest == orig_digest
                    # Now that the info from the previous run has been used to
                    # update `taskinputs`, we delete `outputs` to indicate that
                    # they still need to be computed
                    outputs = None
                    continue_previous_run = True

        elif not recompute:
            logger.info(
                type(self).__qualname__ + ": loading memoized result")
            outputs = self._run_result

        if outputs is None:
            # We did not find a previously computed result, so run the task
            if recompute:
                logger.info(f"Recomputing task {self.name}.")
            elif continue_previous_run:
                logger.info(
                    self.name + ": continuing from a previous partial result.")
            else:
                logger.info(
                    self.name + ": no previously saved result was found; "
                    "running task.")
            outputs = self._run_and_record(record)

        if cache and self._run_result is NotComputed:
            self._run_result = outputs

        return outputs.result

    def _run_and_record(self, record: bool=None):
        if record is None:
            record = config.record
        input_data = [input.generate_key()
                      for input in self.input_files]
        # Module where task is defined
        # Decorators set the _module_name attribute explicitely, because with the
        # dynamically created class, the `type(self)` method gets the module wrong
        module_name = getattr(self, '_module_name', type(self).__module__)
        module = sys.modules[module_name]
        if record:
            # Append a few chars from digest so simultaneous runs don't
            # have clashing labels
            label = datetime.now().strftime(TIMESTAMP_FORMAT) + '_' + self.digest[:4]
            smtrecord = config.project.new_record(
                parameters=self.desc,
                input_data=input_data,
                script_args=type(self).__name__,
                executable=PythonExecutable(sys.executable),
                main_file=module.__file__,
                reason=self.reason,
                label=label
                )
            start_time = time.time()
        elif not config.allow_uncommitted_changes:
            # Check that changes are committed. This is normally done in new_record().
            # See sumatra/projects.py:Project.new_record
            repository = deepcopy(config.project.default_repository)
            working_copy = repository.get_working_copy()
            config.project.update_code(working_copy)
        outputs = self.Outputs.parse_result(
            # We don't use .dict() here, because that would dictifiy all nested
            # BaseModels, which would then be immediately recreated from their dict
            self._run(**dict(self.load_inputs())), _task=self)
        if record:
            smtrecord.duration = time.time() - start_time
        if len(outputs) == 0:
            warn("No output was produced.")
        elif record:
            realoutputpaths = outputs.write()
            if len(realoutputpaths) != len(outputs):
                warn("Something went wrong when writing task outputs. "
                     f"\nNo. of outputs: {len(outputs)} "
                     f"\nNo. of output paths: {len(realoutputpaths)}")
                smtrecord.outcome += ("Error while writing to disk: possibly "
                                      "missing or unrecorded data.")
            smtrecord.output_data = [
                DataFile(path, config.project.data_store).generate_key()
                for path in realoutputpaths]
        if record:
            config.project.add_record(smtrecord)

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
                if not itervalue.isdigit():
                    warn("The iteration step parsed from the output is not an"
                         "integer. It will nevertheless be coerced to int.\n"
                         f"Iteration: {itervalue}\nFile name: {fname}")
                itervalue = int(itervalue)
                if itervalue not in outfiles:
                    outfiles[itervalue] = {}
                outfiles[itervalue][varname] = searchdir/fname
        ## Check if there is an output matching the desired iterations
        iterp_val = getattr(self.taskinputs, iterp_name)
        if (iterp_val in outfiles
            and all(attr in outfiles[iterp_val] for attr in self.Outputs._outputnames_gen(self))):
            logger.debug(f"Found output from a previous run of task '{self.name}' matching these parameters.")
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
                logger.debug(f"Found output from a previous run of task '{self.name}' matching these "
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
    The intention is for tasks which are cheap to compute, and thus it does
    not make sense to store the output. A prime example would be a random
    number generator, for which it is much more efficient to store a function,
    a random seed and some parameters.
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

    def run(self, cache=None, recompute=False):
        if cache is None:
            cache = self.cache if self.cache is not None else config.cache_runs
        if self._run_result is NotComputed or recompute:
            input_data = [input.generate_key() for input in self.input_files]
            module = sys.modules[type(self).__module__]
            if not config.allow_uncommitted_changes:
                # Check that changes are committed. This is normally done in new_record().
                # See sumatra/projects.py:Project.new_record
                repository = deepcopy(config.project.default_repository)
                working_copy = repository.get_working_copy()
                config.project.update_code(working_copy)

            logger.info(f"Running task {self.name} in memory.")
            # We don't use .dict() here, because that would dictifiy all nested
            # BaseModels, which would then be immediately recreated from their dict
            output = self.Outputs.parse_result(
                self._run(**dict(self.load_inputs())),  _task=self)
            if cache:
                self._run_result = output
                logger.debug(f"Memoized result of task {self.name}.")
        else:
            output = self._run_result
            logger.debug(f"Retrieved memoized result of task {self.name}.")
        return output.result

class UnpureMemoizedTask(MemoizedTask):
    """
    A Task whose output does *not* only depend on the inputs (and thus is not
    a pure function). An UnpureTask cannot be recorded, because its digest is
    computed from its output. To motivate the use of such a Task, consider the
    following set of operations:

    TaskA (s: string) -> Return the list of entries in a database containing `s`.
    TaskB (l: list|TaskA) -> Return a set of statistics for those entries.

    TaskA is unpure: it depends on `s`, but also on the contents of the database.
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

    Because an UnpureTask is not recorded, it is also not meaningful to
    specify a `reason` argument.

    .. Important:: `UnpureTask` still performs in-memory caching (memoization).
       This means that non-input dependencies (in the example above, the
       contents of the database) must not change during workflow execution.
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
            result = self.Outputs.parse_result(
                self._run(**dict(self.load_inputs())),
                _task=self
            )
            if result.digest != self._memoized_run_result:
                if self._memoized_run_result is not None:
                    warn("Digest has changed for task {self.name}.")
                object.__setattr__(self, '_memoized_run_result', result)
        return self._memoized_run_result

    def run(self, cache=None, recompute=False):
        if cache == False or self.cache != True:
            raise ValueError("The implementation of UnpureMemoizedTask "
                             "requires caching, and so it cannot be run with "
                             "``cache=False``.")
        return self._get_run_result(recompute).result

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
