import sys
import os
from warnings import warn
import logging
import time
from copy import deepcopy
from collections import deque
from pathlib import Path
from sumatra.datastore.filesystem import DataFile
from sumatra.programs import PythonExecutable
from mackelab_toolbox.parameters import digest
import mackelab_toolbox.iotools as io
logger = logging.getLogger()

from .base import project, File, PlainArg, ParameterSet, TaskBase, NotComputed, cache_runs, RecordedTaskBase, describe

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

    def run(self, cache=None, recompute=False):
        """
        Set `smttask.cache_default = True` to have all tasks cached.
        """
        # Dereference links: links may change, so in the db record we want to
        # save paths to actual files
        # Typically these are files in the output datastore, but we save
        # paths relative to the *input* datastore.root, because that's the root
        # we would use to reexecute the task.
        # input_files = [os.path.relpath(os.path.realpath(input),
        #                                start=project.input_datastore.root)
        #                for input in self.input_files]
        if cache is None:
            cache = self.cache if self.cache is not None else cache_runs
        inroot = Path(project.input_datastore.root)
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
            record = project.new_record(
                parameters=self.desc,
                input_data=input_data,
                script_args=type(self).__name__,
                executable=PythonExecutable(sys.executable),
                main_file=module.__file__,
                reason=self.reason,
                )
            start_time = time.time()
            outputs = self._run(**self.load_inputs())
            record.duration = time.time() - start_time
            if len(outputs) == 0:
            # if len(self.outputpaths) == 0:
                warn("No output was produced.")
            else:
                realoutputpaths = self.write(outputs)
                if len(realoutputpaths) != len(outputs):
                    warn("Something went wrong when writing task outputs. "
                         f"\nNo. of outputs: {len(outputs)} "
                         f"\nNo. of output paths: {len(realoutputpaths)}")
                    record.outcome("Error while writing to disk: possibly "
                                   "missing or unrecorded data.")
                record.output_data = [
                    DataFile(path, project.data_store).generate_key()
                    for path in realoutputpaths]
            project.add_record(record)

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
                / (digest(self.desc) + '_' + nm + self.outext)
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
        if not isinstance(outputs, (tuple, dict)):
            logger.warning("Functions returning a single result must still "
                           "wrap it in a tuple or dict. Automatic wrapping "
                           "will be done here but is not robust in general.")
            outputs = (outputs,)
        if isinstance(outputs, tuple):
            # Standardize to dict format
            if not len(outputs) == len(self.outputs):
                logger.warning(
                    "Unexpected number of outputs: task defines {}, but {} "
                    "were passed.".format(len(self.outputs), len(outputs)))
            outputs = {nm: val for nm, val in zip(self.outputs, outputs)}

        outroot = Path(project.data_store.root)
        inroot = Path(project.input_datastore.root)
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
                os.symlink(outpath, inpath)
        return outpaths

class InMemoryTask(TaskBase):
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

    def run(self):
        if self._run_result is NotComputed:
            input_data = [input.generate_key() for input in self.input_files]
            module = sys.modules[type(self).__module__]
            # Check that changes are committed. This is normally done in new_record().
            # See sumatra/projects.py:Project.new_record
            repository = deepcopy(project.default_repository)
            working_copy = repository.get_working_copy()
            project.update_code(working_copy)

            self._run_result = self._run(**self.load_inputs())
        return self._run_result
