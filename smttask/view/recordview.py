import logging
import os.path
from json import JSONDecodeError
from typing import Union, Any, Sequence
import copy
from functools import lru_cache
from sumatra.records import Record
from mackelab_toolbox import iotools

from .config import config
from . import utils
from smttask.base import Task  # Required for get_task_param

logger = logging.getLogger(__name__)

# TODO: Does Sumatra have a read-only Record ? I can't find one, but it would
#       make sense to use it as a base type.

class RecordView:
    """
    A read-only interface to Sumatra records with extra convenience methods.
    In contrast to Sumatra.Record, RecordView is hashable and thus can be used in sets
    or as a dictionary key.
    """
    # Within `get_output`, these are tried in sequence, and the first
    # to load successfully is returned.
    # At present these must all be subtypes of pydantic.BaseModel
    data_models = []

    ## RecordView creation and hashing ##
    def __new__(cls, record, *args, **kwargs):
        # To make RecordView as transparent a substitute to Record as possible,
        # we allow it to be used to initialize RecordView, but we don't want to
        # create an unnecessary indirection layer
        if isinstance(record, RecordView):
            return record
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, record):
        if isinstance(record, RecordView):
            # Nothing to do: __new__ simply returned `record`
            return
        elif not isinstance(record, Record):
            raise ValueError("'record' must be an instance of sumatra.records.Record.")
        self._record = record
        # Setting read-only attributes with a loop as below /seems/ to work, but
        # causes trouble when filtering
        #for attr in […]
            # def getter(self):
            #     return getattr(self._record, attr)
            # setattr(self, attr, property(getter))

    def __hash__(self):
        return hash(self.label)
        # # Hash must return an int
        # # The following returns the label, converting non-numeric characters to their
        # # ASCII value, so '20180319-225426' becomes 2018031945225426.
        # return int(''.join(c if c.isnumeric() else str(ord(c)) for c in self.label))

    ## New functionality ##
    @property
    def outputpaths(self):
        return [ os.path.join(self.datastore.root, output_data.path)
                 for output_data in self.output_data ]
    @property
    def outputpath(self):
        logger.warning("DEPRECATION: Use `outputpaths` instead of `outputpath`.")
        return self.outputpaths

    def get_output(self, name="", data_models=()):
        """
        Load the output data associated the provided name.
        (The association to `name` is done by matching the output path.)
        `name` should be such that that exactly one output path matches;
        if the record produced only one output, `name` is not required.

        After having found the output file path, the method attempts to
        load it with the provided data models; the first to succeed is returned.
        A list of data models can be provided via the `data_models`, but
        in general it is more convenient to set a default list with the class variable
        `RecordView.data_models`. Models passed as arguments have precedence.

        The expected data model type is a Pydantic BaseModel. Other types can
        be used, as long as

        - They implement a *class* method `parse_file` accepting a path
          to serialized model data;
        - Their `parse_file` method raises `json.JSONDecodeError` if
          deserialization is unsuccessful.
        """
        data_models = list(data_models) + config.data_models
        if not data_models:
            raise TypeError("`get_output` requires at least one data model, "
                            "given either as argument or by setting the class "
                            "attribute `RecordView.data_models`.")
        # TODO: Allow to specify extension, but still match files with _1, _2… suffix added by iotools ?
        # TODO: If name does not contain extension, force match to be right before '.', allowing for _1, _2… suffix ?
        # if '.' not in name:
        #     name += '.'
        paths = []
        for path in self.outputpaths:
            if name in path:
                paths.append(path)
        if len(paths) == 0:
            raise FileNotFoundError(f"The record {self.label} does not have an "
                                    f"output file with name '{name}'")
        elif len(paths) > 1:
            paths_str = '\n'.join(paths)
            raise ValueError(f"The record {self.label} has multiple files with "
                             f"the name '{name}':\n{paths_str}")
        else:
            for F in data_models:
                try:
                    return F.parse_file(paths[0])
                except JSONDecodeError:
                    pass
            raise JSONDecodeError(f"The file at location {paths[0]} is unrecognized "
                                  f"by any of the following types: {data_models}.")

    def get_param(self, name: Union[str, Sequence], default: Any=utils.NO_VALUE):
        """
        A convenience function for retrieving values from the record's parameter
        set. Attributes of object types are accessed with slightly convoluted
        syntax, and this gets especially cumbersome with nested parameters. This
        function is applied recursively, at each level selecting the appropriate
        syntax depending on the value's type.
        
        This is a wrapper around smttask.view._utils.get_task_param.
        
        Parameters
        ----------
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
        return get_task_param(self, name, default)
        

    ## Set all the Record attributes as read-only properties ##
    @property
    def timestamp(self):
        return self._record.timestamp
    @property
    def label(self):
        return self._record.label
    @property
    def reason(self):
        return self._record.reason,
    @property
    def duration(self):
        return self._record.duration
    @property
    def executable(self):
        return self._record.executable
    @property
    def repository(self):
        return self._record.repository
    @property
    def main_file(self):
        return self._record.main_file
    @property
    def version(self):
        return self._record.version
    @property
    @lru_cache(maxsize=None)  # No need for the LRU cache: there can only ever be one memoized value
    def parameters(self):     # By memoizing the return value, we allow multiple calls to access the same mutable varable
        # NB: Don't return record.parameters: that variable is mutable, and
        #     therefore a user could modify it by accident
        return copy.deepcopy(self._record.parameters)
    @property
    def input_data(self):
        return self._record.input_data
    @property
    def script_arguments(self):
        return self._record.script_arguments
    @property
    def launch_mode(self):
        return self._record.launch_mode
    @property
    def datastore(self):
        return self._record.datastore
    @property
    @lru_cache(maxsize=None)  # See `parameters()`
    def dependencies(self):
        return copy.deepcopy(self._record.dependencies)
    @property
    def input_datastore(self):
        return self._record.input_datastore
    @property
    def outcome(self):
        return self._record.outcome
    @property
    @lru_cache(maxsize=None)  # See `parameters()`
    def output_data(self):
        # Shallow copy should suffice: list of DataKey
        return copy.copy(self._record.output_data)
    @property
    def tags(self):
        return self._record.tags
    @property
    def diff(self):
        return self._record.diff
    @property
    def user(self):
        return self._record.user
    @property
    def on_changed(self):
        return self._record.on_changed
    @property
    def stdout_stderr(self):
        return self._record.stdout_stderr
    @property
    def repeats(self):
        return self._record.repeats

    # Reproduce the Record interface; database writing functions are deactivated.
    def __nowrite(self):
        raise AttributeError("RecordView is read-only – Operations associated with "
                             "running or writingg to the database are disabled.")
    def register(self, *args, **kwargs):
        self.__nowrite()
    def run(self, *args, **kwargs):
        self.__nowrite()
    def __repr__(self):
        return repr(self._record)
    def describe(self, *arg, **kwargs):
        return self._record.describe(*args, **kwargs)
    def __ne__(self, other):
        return self._record != (other._record if isinstance(other, RecordView) else other)
    def __eq__(self, other):
        return self._record == (other._record if isinstance(other, RecordView) else other)
    def difference(self, *args, **kwargs):
        return self._record.difference(*args, **kwargs)
    def delete_data(self):
        self.__nowrite()
    @property
    def command_line(self):
        return self._record.command_line
    @property
    def script_content(self):
        return self._record.script_content
        
# Can't place this in utils because it depends on RecordView and would create
# an import cycle
def get_task_param(obj, name: Union[str, Sequence], default: Any=utils.NO_VALUE):
    """
    A convenience function for retrieving values from nested parameter sets
    or tasks. Attributes of object types are accessed with slightly convoluted
    syntax, and this gets especially cumbersome with nested parameters. This
    function is applied recursively, at each level selecting the appropriate
    syntax depending on the value's type.

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
            if default is not utils.NO_VALUE:
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
            if default is not utils.NO_VALUE:
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
