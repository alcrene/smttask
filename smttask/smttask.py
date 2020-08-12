import sys
import os
from warnings import warn
import logging
import time
from copy import deepcopy
from collections import deque, Iterable
from datetime import datetime
from pathlib import Path
from sumatra.core import TIMESTAMP_FORMAT
from sumatra.datastore.filesystem import DataFile
from sumatra.programs import PythonExecutable
import mackelab_toolbox.iotools as io

import pydantic.parse

from .base import config, ParameterSet, Task, NotComputed, RecordedTaskBase
from .typing import PlainArg
from . import utils

# project = config.project

# TODO: Include run label in project.datastore.root

logger = logging.getLogger()

__ALL__ = ['Task', 'InMemoryTask']

class RecordedTask(RecordedTaskBase):

    def __init__(self, params=None, *, reason=None, **taskinputs):
        super().__init__(params, reason=reason, **taskinputs)
        self.outext = ""  # If not empty, should start with period
        self.reason = reason

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
        if record is None:
            record = config.record
        inroot = Path(config.project.input_datastore.root)
        outputs = None

        # First try to load pre-computed result
        if self._run_result is NotComputed and not recompute:
            # First check if output has already been produced
            _outputs = {}
            try:
                for nm, path in zip(self.Outputs.__fields__, self._outputpaths_gen):

                    # Next line copied from pydantic.main.parse_file
                    _outputs[nm] = pydantic.parse.load_file(
                        inroot/path,
                        proto=None, content_type='json', encoding='utf-8',
                        allow_pickle=False,
                        json_loads=self.Outputs.__config__.json_loads)
            except FileNotFoundError:
                pass
            else:
                logger.debug(
                    type(self).__qualname__ + ": loading result of previous "
                    "run from disk.")
                # Only assign to `outputs` once all outputs are loaded successfully
                # outputs = tuple(_outputs)
                outputs = self.Outputs(**_outputs, _task=self)
        elif not recompute:
            logger.debug(
                type(self).__qualname__ + ": loading from in-memory cache")
            outputs = self._run_result

        if outputs is None:
            # We did not find a previously computed result, so run the task
            logger.debug(
                type(self).__qualname__ + ": No cached result was found; "
                "running task.")
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
                self._run(**self.load_inputs().dict()), _task=self)
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

        if cache and self._run_result is NotComputed:
            self._run_result = outputs

        return outputs.result

    @property
    def _outputpaths_gen(self):
        """
        Returns
        -------
        Generator for the output paths
        """
        return (Path(type(self).__name__)
                # / (self.digest + '_' + nm + self.outext)
                / f"{self.digest}_{nm}.json" for nm in self.Outputs.__fields__)
    @property
    def outputpaths(self):
        """
        Returns
        -------
        Dictionary of output name: output paths pairs
        """
        return {nm: path
                for nm, path in zip(self.Outputs.__fields__,
                                    self._outputpaths_gen)}

class InMemoryTask(Task):
    """
    Behaves like a task, in particular with regards to computing descriptions
    and digests of composited tasks.
    However the output is not saved to disk and a sumatra record is not created.
    The intention is for tasks which are cheap to compute, for which it does
    not make sense to store the output. A prime example would be a random
    number generator, for which it is much more efficient to store a function,
    a random seed and some parameters.
    """
    def __init__(self, params=None, *, reason=None, **taskinputs):
        """
        Parameters
        ----------
        params: ParameterSet-like
            ParameterSet, or something which can be cast to a ParameterSet
            (like a dict or filename). The result will be parsed for task
            arguments defined in `self.inputs`.
        **taskinputs:
            Task parameters can also be specified as keyword arguments,
            and will override those in :param:params.
        reason: None | str
            Ignored because we aren't recording into a Sumatra db.
            Included for compability with Task.
        """
        super().__init__(params, reason=reason, **taskinputs)

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

            logger.debug(f"Running task {self.name} in memory.")
            output = self._run(**self.load_inputs().dict())
            if cache:
                logger.debug(f"Caching result of task {self.name}.")
                self._run_result = output
        else:
            logger.debug(f"Result of task {self.name} retrieved from cache")
            output = self._run_result
        return output
