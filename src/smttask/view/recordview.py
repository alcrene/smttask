import logging
import os.path
import collections.abc
# from json import JSONDecodeError
import json
from typing import Optional, Union, Any, Sequence, Dict, Callable
import copy
import pydantic
from functools import lru_cache

from scityping import Serializable

from sumatra.records import Record

from . import utils
from .. import iotools
from .config import config
from ..base import Task  # Required for get_task_param

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
    data_types = []

    ## RecordView creation and hashing ##
    def __new__(cls, record, rsview=None, *args, **kwargs):
        # To make RecordView as transparent a substitute to Record as possible,
        # we allow it to be used to initialize RecordView, but we don't want to
        # create an unnecessary indirection layer
        if isinstance(record, RecordView):
            return record
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, record: Record,
                 rsview: Optional["RecordStoreView"]=None):
        """
        :param:record: The record for which we provide a read-only wrapper
        :param:recordstore: A reference to the record store containing this Record.
           If not provided, operations which modify the record store cannot be
           completed, since we need a handle with which to call `record_store.save()`
        """
        if isinstance(record, RecordView):
            # Nothing to do: __new__ simply returned `record`
            return
        elif not isinstance(record, Record):
            raise ValueError("'record' must be an instance of sumatra.records.Record.")
        self._record = record
        self.rsview = rsview
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

    def get_output(self, name="", data_types=(Serializable,)):
        """
        Load the output data associated the provided name.
        (The association to `name` is done by matching the output path.)
        `name` should be such that that exactly one output path matches;
        if the record produced only one output, `name` is not required.

        After having found the output file path, the method attempts to
        load it with the provided data models; the first to succeed is returned.
        A list of data types can be provided via the `data_types`, but
        in general it is more convenient to set a default list with the class variable
        `RecordView.data_types`. Types passed as arguments have precedence.

        Data types are expected to be types defined by the *Scityping* package.
        Other types can be used, as long as they either:

        - Define a *class* method `validate`, which parses json data.
          It will be called as ``mytype.validate(json_data)``.
        - Accept json data as an argument, i.e. ``mytype(json_data)``

        In both cases, they must raise `TypeError` if `json_data` is not
        compatible with the type.

        If none of the types are able to derialize the data, the JSON data
        is returned as-is.
        """
        data_types = list(data_types) + self.data_types
        if not data_types:
            raise TypeError("`get_output` requires at least one data model, "
                            "given either as argument or by setting the class "
                            "attribute `RecordView.data_types`.")
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
            with open(paths[0], 'r') as f:
                json_data = json.load(f)
            for T in data_types:
                validate = getattr(T, "validate", None)
                if validate:
                    try:
                        obj = validate(json_data)
                    except (TypeError, pydantic.ValidationError):
                        pass
                    else:
                        return obj
                else:
                    try:
                        obj = T(json_data)
                    except TypeError:
                        pass
                    else:
                        return obj
            logger.debug(f"None of the types {data_types} does not provide a `validate` method; returning data as-is.")
            return json_data

            # for F in data_models:
            #     try:
            #         return F.parse_file(paths[0])
            #     except JSONDecodeError:
            #         pass
            # raise JSONDecodeError(f"The file at location {paths[0]} is unrecognized "
            #                       f"by any of the following types: {data_models}.")

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
        

    def update_reason(self, reason: Union[str,Dict[str,str],Callable[[str],str]],
                      mode: str="prepend"):
        """
        Update the 'reason' field for this record.

        :param:reason: Either:
            - String to add to the records reasons (or to replace with)
            - Callback function, taking the record's 'reason' string and
              returning the updated one. If this function returns `None` or
              the unmodified reason string, the record is not modified.
        :param:mode: One of 'prepend', 'append', 'replace all', 'replace substr', 'callback'.
            Modes 'replace substr' and 'callback' can be left unspecified:
            they are inferred from the type of `reason`.

        If the mode is 'prepend' or 'append', and `reason` is already a substring
        of the record's 'reason' field **at any position**, then the record
        is not modified. This is to reduce the likelihood of accidentally
        growing the 'reason' field (e.g., with two functions each prepending
        different strings).

        .. Note:: Some standardizations are applied to all reason strings,
           even if they are are otherwise unmodified.

        **Modes**

        ``"prepend"``
           The new reason is `reason` + record.reason.

        ``"append"``
           The new reason is record.reason + `reason`.

        ``"replace all"``
           The new reason is `reason`.

        ``"replace substr"``
           For each {pattern: string} pair in `reason`, we call
           ``re.sub(pattern, string, reason)``. All occurences of 'pattern'
           are replaced by 'string'.

        ``"callback"``
           The new reason is ``callback(reason)``.

        ``"standardize"``
           Only apply the standardizations.

        **Standardizations**

        - Sequences (tuple, list, etc.) of length one are replaced by their
          first element. This is because while it is possible to store sequences
          in the 'reason' field, a string is really the expected format and
          better supported (both by the schema and by the UI).

        .. Note:: At the risk of stating the obvious, this function will modify
           the underlying record store.
        """
        modes = {"prepend", "append", "replace all", "replace substr", "callback"}
        if mode not in modes:
            raise ValueError(f"'mode' must be one of {modes}; received {mode}.")
        if isinstance(reason, dict):
            if mode not in {"prepend", "replace substr"}:  # "prepend" is default
                raise ValueError("A dictionary argument is only compatiable "
                                 "with the 'replace substr' mode; received "
                                 f"mode={mode}.")
            mode = "replace substr"
        elif mode == "replace substr":
            raise TypeError("The mode 'replace substr' was specified, but the "
                            "'reason' argument is not a dictionary.")
        if isinstance(reason, collections.abc.Callable):
            if mode not in {"prepend", "callback"}:  # "prepend" is default
                raise ValueError("A dictionary argument is only compatiable "
                                 "with the 'replace substr' mode; received "
                                 f"mode={mode}.")
            mode = "callback"
        elif mode == "callback":
            raise TypeError("The mode 'callback' was specified, but the "
                            "'reason' argument is not a function.")
        # NB: It's possible for 'reason' to contain a tuple of strings instead
        #     the expected single string. This breaks the UIs a little though,
        #     so whenever possible we set the new reason to a string, even if
        #     the original was a tuple.
        record = self._record
        # record = self.record_store.get(self.project.name, self.label)
        # Apply the update
        if mode == "standardize":
            new_reason = record.reason
        elif mode == "callback":
            new_reason = reason(record.reason)
            if new_reason is None:
                new_reason = record.reason  # Might still be subject to some standardization below
        elif mode == "replace all":
            new_reason = reason
        elif mode == "replace substr":
            nsubs = 0
            new_reason = (record_view.reason,) if isinstance(record_view.reason, str) else record_view.reason
            for i, s in enumerate(new_reason):  # Modify the first matching tuple element
                for pattern, new_str in reason.items():
                    s, c = re.subn(pattern, new_str, s)
                    nsubs += c
                if nsubs:  # Checking nsubs may be overeager, but only if we have a len > 1 tuple, which isn't supposed to happen
                    new_reason = (*new_reason[:i], s, *new_reason[i+1:])
            if not nsubs:
                patterns = ", ".join((f'"{pattern}"' for pattern in reason))
                logger.info(f"Reason of record {record_view.label} was not modified: "
                            f"no string matches {patterns}.")
        else:
            # reason in {'prepend', 'append'}
            if record.reason is None:
                new_reason = reason
            elif reason in str(record.reason):  # Works whether record.reason is a tuple or str
                return
            elif mode == "prepend":
                new_reason = reason + record.reason
            else:
                new_reason = record.reason + reason
        # Apply standardization
        if isinstance(new_reason, collections.abc.Sequence) and len(new_reason) == 1:
            # Return a string whenever possible
            new_reason = new_reason[0]
        # Update record store if reason has changed
        if new_reason != record.reason:
            record.reason = new_reason
            rsview = self.rsview
            if rsview is None:
                logger.error(f"Record {self} was created without a reference "
                             "to a RecordStoreView: the updated reason may not be "
                             "saved to disk. To ensure your update is saved, "
                             "run `record_store.save(project_name, record)`.")
            else:
                rsview.record_store.save(rsview.project.name, record)


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
        # NB: Don't return record.parameters: that variable is mutable, and
        #     therefore a user could modify it by accident
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
                             "running or writing to the database are disabled.")
    def register(self, *args, **kwargs):
        self.__nowrite()
    def run(self, *args, **kwargs):
        self.__nowrite()
    def __repr__(self):
        return repr(self._record)
    def describe(self, *args, **kwargs):
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
        try:
            val = getattr(obj, name)
        except AttributeError as e:
            if default is not utils.NO_VALUE:
                val = default
            else:
                raise KeyError from e
    # NB: We especially want to make sure to avoid recursing when getattr failed and returned the default value
    if subname is not None:
        if val != default:
            val = get_task_param(val, subname, default)
        elif isinstance(val, (dict, Task)):
            logger.warning("Using `dict` or `Task` values as 'default' is not "
                           "supported.")
    return val
