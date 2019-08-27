import sys
import os
import logging
import time
from pathlib import Path
from numbers import Number
from attrdict import AttrDict
from sumatra.projects import load_project
from sumatra.parameters import build_parameters, NTParameterSet
from sumatra.datastore.filesystem import DataFile
from sumatra.programs import PythonExecutable
from mackelab_toolbox.parameters import digest
import mackelab_toolbox.iotools as io
logger = logging.getLogger()

# TODO: Include run label in project.datastore.root

###########
# If needed, there variables could be overwritten in a project script
project = load_project()
# Define plain arguments types
PlainArg = (Number, str)
###########

# Monkey patch AttrDict to allow access to attributes with unicode chars
def _valid_name(self, key):
    cls = type(self)
    return (
        isinstance(key, str) and
        key.isidentifier() and key[0] != '_' and  # This line changed
        not hasattr(cls, key)
    )
import attrdict.mixins
attrdict.mixins.Attr._valid_name = _valid_name

# # Provide special executable class for runfiles
# class PythonRunfileExecutable(PythonExecutable):
#     # name = "Python (runfile)"
#     requires_script = False

class File:
     """Use this to specify a dependency which is a filename."""
#     def __init__(self, filename):
#         self.filename = filename
     @staticmethod
     def desc(filename):
         return NTParameterSet({
             'type': 'File',
             'filename': filename
         })

class Task:
    def __new__(cls, taskinputs, reason=None, *args, **kwargs):
        if isinstance(taskinputs, cls):
            return taskinputs
        else:
            return super().__new__(cls, *args, **kwargs)
    def __init__(self, taskinputs, reason=None):
        """
        taskinputs:
            dict, or path-like pointing to a parameter file.
            If task has only one input, the dict wrapper is optional.
        """
        assert hasattr(self, 'inputs')
        if isinstance(taskinputs, type(self)):
            # Skip initializion of pre-existing instance (see __new__)
            assert hasattr(taskinputs, '_inputs')
            return
        if isinstance(taskinputs, str):
            taskinputs = build_parameters(taskinputs)
        elif not isinstance(taskinputs, dict):
            if len(self.inputs) == 1:
                # For tasks with only one input, don't require dict
                θname, θtype = next(iter(self.inputs.items()))
                if not isinstance(taskinputs, θtype):
                    # Cast to correct type
                    taskinputs = θtype(taskinputs)
                taskinputs = NTParameterSet({θname: taskinputs})
            else:
                raise ValueError("`taskinputs` should be either a dictionary "
                                 "or a path to a parameter file.")
        self.outext = ""  # If not empty, should start with period
        self.taskinputs = taskinputs
        self.reason = reason
        self._inputs = None  # Where inputs are stored once loaded
        self.inputs = self.get_inputs
        self.outputpaths = []

    @property
    def desc(self):
        descset = NTParameterSet({
            'type': 'Task',
            'inputs': NTParameterSet({})
        })
        for k, v in self._input_descs.items():
            #vtype = type(self).inputs[k]
            if isinstance(v, PlainArg):
                vdesc = str(v)
            elif isinstance(v, Task):
                vdesc = v.desc
            elif isinstance(v, DataFile):
                vdesc = File.desc(v.full_path)
            else:
                raise TypeError
            descset['inputs'][k] = vdesc
        return descset

    @property
    def input_files(self):
        # Also makes paths relative, in case they weren't already
        store = project.input_datastore
        return [DataFile(Path(input.path).relative_to(store.root), store)
                for input in self._input_descs.values()
                if isinstance(input, DataFile)]

    def run(self):
        # Dereference links: links may change, so in the db record we want to
        # save paths to actual files
        # Typically these are files in the output datastore, but we save
        # paths relative to the *input* datastore.root, because that's the root
        # we would use to reexecute the task.
        # input_files = [os.path.relpath(os.path.realpath(input),
        #                                start=project.input_datastore.root)
        #                for input in self.input_files]
        input_data = [input.generate_key() for input in self.input_files]
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
        self.code()
        record.duration = time.time() - start_time
        if len(self.outputpaths) == 0:
            logger.warning("No output was produced.")
        else:
            record.output_data = [
                DataFile(path, project.data_store).generate_key()
                for path in self.outputpaths]
        project.add_record(record)

    @property
    def get_inputs(self):
        # self._inputs implements a cache, so data are only loaded once
        if self._inputs is None:
            datastore = project.input_datastore
            inputs = AttrDict()
            for name, θtype in type(self).inputs.items():
                θ = self.taskinputs[name]
                if issubclass(θtype, PlainArg):
                    inputs[name] = θ
                elif isinstance(input, File):
                    inputs[name] = DataFile(θ, datastore)
                else:
                    # Assume input is a Task
                    outputs = θtype(θ).outputs
                    if isinstance(outputs, dict):
                        for outname, output in outputs.items():
                            outputpath = io.find_file(datastore.root/output)
                            if isinstance(outputpath, list):
                                logger.warning("Multiple input files found: "
                                               + str(outputpath))
                                outputpath = outputpath[0]
                            inputs[name][outname] = DataFile(outputpath,
                                                             datastore)
                    else:
                        # Assume output is a single filename
                        outputpath = io.find_file(datastore.root/outputs)
                        if isinstance(outputpath, list):
                            logger.warning("Multiple input files found: "
                                           + str(outputpath))
                            outputpath = outputpath[0]
                        inputs[name] = DataFile(outputpath, datastore)
            self._input_descs = inputs
            self._inputs = AttrDict({k: v if not isinstance(v, DataFile)
                                          else io.load(v.full_path)
                                     for k,v in inputs.items()})
        return self._inputs

    @property
    def outputs(self):
        """
        Can return either single filename (default) or a dictionary
        of filenames.
        Redefine in derived class if you need multiple outputs.
        """
        return Path(type(self).__name__) / (digest(self.desc) + self.outext)

    def write_output(self, *args, **kwargs):
        """
        Use `arg` for single output, `kwargs` for multiple outputs
        """
        outroot = Path(project.data_store.root)
        inroot = Path(project.input_datastore.root)
        outputs = self.outputs
        tosave = []
        if len(args) > 0:
            assert isinstance(outputs, Path)
            assert len(args) == 1 and len(kwargs) == 0
            tosave.append((Path(outputs), args[0]))
        else:
            assert isinstance(outnames, dict)
            for k, v in kwargs.items():
                tosave.append((Path(outputs[k]), v))
        for save in tosave:
            outpaths = io.save(outroot/save[0], save[1])
                # May return multiple save locations with different filenames
            self.outputpaths.extend(outpaths)
            # Add link in input store, potentiall overwriting old link
            for outpath in outpaths:
                inpath = inroot/save[0].with_suffix(outpath.suffix)
                if os.path.exists(inpath):
                    # Deal with race condition ? Wait future Python builtin ?
                    # See https://stackoverflow.com/a/55741590,
                    #     https://github.com/python/cpython/pull/14464
                    os.remove(inpath)
                else:
                    os.makedirs(inpath.parent, exist_ok=True)
                os.symlink(outpath, inpath)
