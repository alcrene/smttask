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
logger = logging.getLogger()

from .base import config, ParameterSet, Task, NotComputed, RecordedTaskBase, describe
from .typing import File, PlainArg, cast
from . import utils

# project = config.project

# TODO: Include run label in project.datastore.root

__ALL__ = ['Task', 'InMemoryTask']

# # Provide special executable class for runfiles
# class PythonRunfileExecutable(PythonExecutable):
#     # name = "Python (runfile)"
#     requires_script = False

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
        # Dereference links: links may change, so in the db record we want to
        # save paths to actual files
        # Typically these are files in the output datastore, but we save
        # paths relative to the *input* datastore.root, because that's the root
        # we would use to reexecute the task.
        # input_files = [os.path.relpath(os.path.realpath(input),
        #                                start=config.project.input_datastore.root)
        #                for input in self.input_files]
        if cache is None:
            cache = self.cache if self.cache is not None else config.cache_runs
        if record is None:
            record = config.record
        inroot = Path(config.project.input_datastore.root)
        outputs = None

        # First try to load pre-computed result
        if self._run_result is NotComputed and not recompute:
            # First check if output has already been produced
            _outputs = deque()
            try:
                for nm, p in zip(self.outputs, self._outputpaths_gen):
                    if isinstance(self.outputs, dict):
                        format = self.outputs[nm]
                    else:
                        format = None
                    _outputs.append(io.load(inroot/p, format=format))
            except FileNotFoundError:
                pass
            else:
                logger.debug(
                    type(self).__qualname__ + ": loading result of previous "
                    "run from disk.")
                # Only assign to `outputs` once all outputs are loaded successfully
                outputs = tuple(_outputs)
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
            module = sys.modules[type(self).__module__]
              # Module where task is defined
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
            outputs = cast(self._run(**self.load_inputs()),
                           self.output_type)
            # if not isinstance(outputs, Iterable):
            #     warn("Task {} did not return a tuple. This will cause "
            #          "problems when composing with other tasks.")
            if record:
                smtrecord.duration = time.time() - start_time
            if len(outputs) == 0:
                warn("No output was produced.")
            elif record:
                realoutputpaths = self.write(outputs)
                if len(realoutputpaths) != len(self.outputs):
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

        return outputs

    @property
    def _outputpaths_gen(self):
        """
        Returns
        -------
        Generator for the output paths
        """
        return (Path(type(self).__name__)
                / (self.digest + '_' + nm + self.outext)
                for nm in self.outputs)
    @property
    def outputpaths(self):
        """
        Returns
        -------
        Dictionary of output name: output paths pairs
        """
        return {nm: path
                for nm, path in zip(self.outputs, self._outputpaths_gen)}

    def write(self, outputs):
        """
        Parameters
        ----------
        outputs: tuple | dict
            If a dict, keys must match `self.outputs`.
            If a tuple, length must match `self.outputs`.
        """
        # FIXME - Output type (See also: base.Task.__init__)
        # Should be: if not isinstance(self.output_type, MultipleOutputsType):
        if len(self.outputs) == 1:
            # Wrap with a tuple so we can iterate over outputs
            outputs = (outputs,)
        if isinstance(outputs, tuple):
            # Standardize to dict format
            if not len(outputs) == len(self.outputs):
                logger.warning(
                    "Unexpected number of outputs: task defines {}, but {} "
                    "were passed.".format(len(self.outputs), len(outputs)))
            outputs = {nm: val for nm, val in zip(self.outputs, outputs)}

        outroot = Path(config.project.data_store.root)
        inroot = Path(config.project.input_datastore.root)
        orig_outpaths = self.outputpaths
        outpaths = []  # outpaths may add suffix to avoid overwriting data
        for nm in self.outputs:
            if isinstance(self.outputs, dict):
                format = self.outputs[nm]
            else:
                format = None
            path = orig_outpaths[nm]
            value = outputs[nm]
            _outpaths = io.save(outroot/path, value, format=format)
                # May return multiple save locations with different suffixes
            outpaths.extend(_outpaths)
            # Add link in input store, potentially overwriting old link
            for outpath in _outpaths:
                inpath = inroot/path.with_suffix(outpath.suffix)
                if inpath.is_symlink():
                    # Deal with race condition ? Wait future Python builtin ?
                    # See https://stackoverflow.com/a/55741590,
                    #     https://github.com/python/cpython/pull/14464
                    os.remove(inpath)
                else:
                    os.makedirs(inpath.parent, exist_ok=True)
                # Create the link as a relative path, so that it's portable
                # outrelpath = outpath.relative_to(inroot)
                # inrelpath  = inpath.relative_to(inroot)
                # depth = len(inrelpath.parents) - 1
                #     # The number of '..' we need to prepend to the link
                #     # The last parent is the cwd ('.') and so doesn't count
                # uppath = Path('/'.join(['..']*depth))
                # # os.symlink(outpath, inpath)
                # os.symlink(uppath.joinpath(outrelpath), inpath)
                os.symlink(utils.relative_path(inpath, outpath, through=inroot),
                           inpath)
        return outpaths

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
            output = self._run(**self.load_inputs())
            if cache:
                logger.debug(f"Caching result of task {self.name}.")
                self._run_result = output
        else:
            logger.debug(f"Result of task {self.name} retrieved from cache")
            output = self._run_result
        return output
