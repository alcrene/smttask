"""
Copied from mackelab_toolbox.iotools, which was created on Wed Sep 20 16:56:37 2017

TODO: Update Format/defined_formats mechanism with scityping where possible
     In particular consider dropping the .npr type

@author: alex
"""

import sys
import os
import os.path
from warnings import warn
import builtins
from pathlib import Path
import io
from collections import namedtuple, OrderedDict
import logging
import numpy as np
import dill
from inspect import isclass
from parameters import ParameterSet
logger = logging.getLogger(__name__)

# TODO: Make dill an optional dependency, use pickle otherwise (with '.dill', '.pkl' extensions).
#       Do this by pulling out all the format-specific export code from `save`
#       into separate functions – then we can just define the fallback type as
#       either 'dill' or 'pkl' depending on whether `dill` could be loaded.
#       (c.f. comment in `save` starting with "Define the save functions below at top level of module […]")
# TODO: (Partly supersedes the one below) Remove any serialization code that
#       duplicates Pydantic functionality.

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    PathTypes = (str, os.PathLike)
else:
    PathTypes = (str,)

Format = namedtuple("Format", ['ext', 'save', 'load', 'bytes'])
    # save: save function, `save(filename, data)`
    # load: load function, `load(filename)`
    # bytes: whether to use the 'b' flag when opening a file for loading/saving
    # TODO: Allow multiple extensions per format
    # TODO: Use namedtuple defaults statt None once Python3.7 is standard
defined_formats = OrderedDict([
    # List of formats known to the load/save functions
    # The order of these formats also defines a preference, for when two types might be used
    ('npr',  Format('npr', None, None, True)),
    ('repr', Format('repr', None, None, False)),
    ('brepr', Format('brepr', None, None, True)), # Binary version of 'repr'
    ('dill', Format('dill', None, None, True))
    ])

_load_types = {}
_format_types = {}

def register_datatype(type, typename=None, format=None, alias=None):
    """
    :param alias: alternative name, usually a shortened form of the full typename.
    """
    global _load_types, _format_types
    assert(isclass(type))
    if typename is None:
        typename = type.__module__ + '.' + type.__qualname__
    if alias is None:
        alias = type.__qualname__
        if alias in _load_types:  # Don't let default override existing alias
            alias = None
    assert(isinstance(typename, str))
    _load_types[typename] = type
    if alias is not None:
        _load_types[alias] = type
    if format is not None:
        assert isinstance(format, str)
        _format_types[typename] = format

def find_registered_typename(type):
    """
    If `type` is a registered datatype, return it's associated name.
    Otherwise, find the nearest parent of `type` which is registered and return its name.
    If no parents are registered, return `type.__name__`.
    """
    def get_name(type):
        # _load_types is always a superset of _format_types
        for registered_name, registered_type in _load_types.items():
            if registered_type is type:
                return registered_name
        return None

    typename = get_name(type)
    if typename is None:
        if not isinstance(type, builtins.type):
            type = builtins.type(type)
        for base in type.mro():
            typename = get_name(base)
            if typename is not None:
                break
    if typename is None:
        typename = type.__name__
            # No registered type; return something sensible (i.e. the type's name)
    return typename

def get_format_from_ext(ext):
    format = None
    ext = ext.strip('.')
    for name, info in defined_formats.items():
        exts = [info.ext] if isinstance(info.ext, str) else info.ext
        for e in exts:
            if e.strip('.') == ext:
                format = name
                break
        if format is not None:
            break
    if format is None:
        raise ValueError("No format matching extension {} was found."
                         .format(ext))
    return format


# Register basic data types
# Python structures can be arbitrarily nested, so we can't assume that they
# would compatible with any strict format
register_datatype(list, format='dill')
register_datatype(tuple, format='dill')
register_datatype(set, format='dill')
register_datatype(dict, format='dill')

def get_free_file(path, bytes=True, max_files=100, force_suffix=False, start_suffix=None):
    """
    Return a file handle to an unused filename. If 'path' is free, return a handle
    to that. Otherwise, append a number to it until a free filename is found or the
    number exceeds 'max_files'. In the latter case, raise 'IOError'.

    Returning a file handle, rather than just a file name, avoids the possibility of a
    race condition (a new file of the same name could be created between the time
    where one finds a free filename and then opens the file).

    Parameters
    ----------
    path: str
        Path name. Can be absolute or relative to the current directory.
    bytes: bool (Default: True)
        (Optional) Specify whether to open the file for byte (True) or plain
        text (False) output. Default is to open for byte output, which
        is suitable for passing to `numpy.save`.
    max_files: int
        (Optional) Maximum allowed number of files with the same name. If this
        number is exceeded, IOError is raised.
    force_suffix: bool (default False)
        (Optional) If True, a suffix '_#', where # is a number, is always added
        to the file name. Forcing suffixes also changes the default value
        of 'start_suffix' to 1.
    start_suffix: int (default 2)
        If creating a file with 'path' is unsuccessful (or 'force_suffix is
        set to True), this is the first number to try appending to the file name.

    Returns
    -------
    filehandle
        Write-only filehandle, as obtained from a call to
        `open(pathname, 'mode='xb')`.
    pathname: str
        Pathname (including the possibly appended number) of the opened file.
    """

    # Get a full path
    # TODO: is cwd always what we want here ?
    if isinstance(path, Path):
        pathname = str(path.absolute())
    elif path[0] == '/':
        #path is already a full path name
        pathname = path
    else:
        #Make a full path from path
        pathname = os.path.abspath(path)

    # Set the default value for start_suffix
    if start_suffix is None:
        start_suffix = 1 if force_suffix else 2

    # Set the mode
    if bytes:
        mode = 'xb'
    else:
        mode = 'x'

    # Make sure the directory exists
    os.makedirs(os.path.dirname(pathname), exist_ok=True)

    try:
        if force_suffix:
            raise IOError
        else:
            f = open(pathname, mode=mode)
            return f, pathname
    except IOError:
        name, ext = os.path.splitext(pathname)
        for i in range(start_suffix, max_files+start_suffix):
            appendedname = name + "_" + str(i) + ext
            try:
                f = open(appendedname, mode=mode)
                return f, appendedname
            except IOError:
                continue

        raise IOError("Number of files with the name '{}' has exceeded limit."
                      .format(path))

def save(file, data, format=None, overwrite=False):
    """Save `data`.
    First the function checks if :param:data defines a `save()` method; if so,
    the method is called as `save(output_path)`. If this is successful, the
    function terminates.
    If the call is not successful, or :param:data does not define a `save()`
    method, then the function attempts to save to the formats defined by
    `format`. By default, only the 'numpy_repr' representation is saved,
    if `data` defines a numpy representation.
    Not only is the numpy representation format more future-proof, it can be an
    order of magnitude more compact.
    If the numpy_repr save is unsuccessful (possibly because `data` does not provide a
    `numpy_repr` method), then `save()` falls back to saving a plain (dill) pickle of 'data'.

    Parameters
    ----------
    file: str
        Path name or file object. Note that the file extension is mostly
        ignored and will be replaced by the one associated with the format.
        This is to allow saving to multiple formats.
    data: Python object
        Data to save
    format: str
        The format in which to save the data. Possible values are:
          - 'npr' (default) Save with the numpy_repr format. This is obtained by calling the
            method 'nprepr' on the `data`. If this call fails, a warning is issued
            and the 'dill' format is used.
            Output file have the extension 'npr'.
            Objects using this format should implement the `from_nprepr` method.
          - 'repr' Call `repr` on the data and save the resulting string to file. The save will
            fail (and fall back to 'dill' format) if the `repr` is simply inherited from object,
            as simply saving the object address is not useful for reconstructing it. Still, there
            is no way of ensuring that the `repr` is sufficiently informative to reconstruct the
            object, so make sure it is before using this format.
            Output file have the extension 'repr'.
            Objects using this format should implement the `from_repr` method.
          - 'dill' A dill pickle.
            Output file has the extension 'dill'
        Formats can also be combined as e.g. 'npr+dill'.
    overwrite: bool
        If True, allow overwriting previously saved files. Default is false, in
        which case a number is appended to the filename to make it unique.

    Returns
    -------
    List of output paths.
        List because many formats may be specified, leading to multiple outputs.
    """
    if isinstance(format, str):
        selected_formats = format
    else:
        if format is None:
            typename = find_registered_typename(type(data))
        else:
            if not isinstance(format, type):
                logger.error("The `format` argument should be either a string "
                             "or type. Provided value: {}"
                             "Attempting to infer type from data".format(format))
                typename = find_registered_typename(type(data))
            typename = find_registered_typename(format)
        if typename in _format_types:
            format = _format_types[typename]
        else:
            logger.error("Type '{}' has no associated format".format(typename))
            format = 'npr'

    selected_formats = set(format.split('+'))

    # Check argument - format
    bad_formats = [f for f in selected_formats if f not in defined_formats]
    selected_formats = selected_formats.difference(bad_formats)
    if len(bad_formats) > 0:
        format_names = ["'" + f + "'" for f in defined_formats]
        bad_format_names = ["'" + f + "'" for f in bad_formats]
        formatstr = "format"
        if len(format_names) > 1:
            format_names = ", ".join(format_names[:-1]) + " and " + format_names[-1]
        if len(bad_format_names) > 1:
            formatstr = "formats"
            bad_format_names = ", ".join(bad_format_names[:-1]) + " and " + bad_format_names[-1]
        logger.warning("Unrecognized save {} {}.".format(formatstr, bad_format_names)
                       + "Recognized formats are " + format_names)
        if len(selected_formats) == 0:
            logger.warning("Setting the format to {}.".format_names)
            # We don't want to throw away the result of a long calculation because of a
            # flag error, so instead we will try to save into every format and let the user
            # sort out the files later.
            format = '+'.join(format_names)

    get_output = None
    def set_str_file(filename):
        nonlocal get_output
        def _get_output(filename, ext, bytes, overwrite):
            return output(filename, ext, bytes, overwrite)
        get_output = _get_output

    # Check argument - file
    if isinstance(file, io.IOBase):
        thisfilename = os.path.realpath(file.name)
        if 'luigi' in os.path.basename(thisfilename):
            # 'file' is actually a Luigi temporary file
            luigi = True
        else:
            luigi = False
        filename = thisfilename  # thisfilename used to avoid name clashes
        if not any(c in file.mode for c in ['w', 'x', 'a', '+']):
            logger.warning("File {} not open for writing; closing and reopening.")
            file.close()
            set_str_file(thisfilename)
        else:
            def _get_output(filename, ext, bytes, overwrite):
                # Check that the file object is compatible with the arguments,
                # and if succesful, just return the file object unmodified.
                # If it is not successful, revert to opening a file as though
                # a filename was passed to `save`.
                # TODO: Put checks in `dummy_file_context`
                fail = False
                if (os.path.splitext(os.path.realpath(filename))[0]
                    != os.path.splitext(os.path.realpath(thisfilename))[0]):
                    logger.warning("[iotools.save] Given filename and file object differ.")
                    fail = True
                thisext = os.path.splitext(thisfilename)[1].strip('.')
                if not luigi and thisext != ext.strip('.'):
                    # Luigi adds 'luigi' to extensions of temporary files; we
                    # don't want that to trigger closing the file
                    logger.warning("[iotools.save] File object has wrong extension.")
                    fail = True
                if (bytes and 'b' not in file.mode
                    or not bytes and 'b' in file.mode):
                    if luigi:
                        # Luigi's LocalTarget always saves to bytes, and it's
                        # the Format class that takes care of converting data
                        # (possibly text) to and back from bytes.
                        logger.warning("\n"
                            "WARNING [iotools]: Attempted to save a 'luigi' target with the wrong "
                            "mode (binary or text). Note that Luigi targets "
                            "always use the same mode internally; use the "
                            "`format` argument to convert to/from in your code. "
                            "In particular, LocalTarget writes in binary. "
                            "Consequently, the file will not be saved as {}, "
                            "but as {}; specify the correct value to `bytes` "
                            "to avoid this message.\n"
                            .format("bytes" if bytes else "text",
                                    "text" if bytes else "bytes"))
                    else:
                        logger.warning("[iotools.save] File object has incorrect byte mode.")
                        fail = True
                if (overwrite and 'a' in file.mode):
                    # Don't check for `not overwrite`: in that case the damage is already done
                    logger.warning("[iotools.save] File object unable to overwrite.")
                    fail = True
                if fail:
                    logger.warning("[iotools.save] Closing and reopening file object.")
                    file.close()
                    set_str_file(thisfilename)
                    return output(filename, ext, bytes, overwrite)
                else:
                    return dummy_file_context(file)
            get_output = _get_output
    else:
        assert isinstance(file, PathTypes)
        filename = file
        set_str_file(file)

    # Ensure target directory exists
    dirname = os.path.dirname(filename)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    output_paths = []

    # If data provides a "save" method, use that
    # This overrides the "format" argument – only exception is if save fails,
    # then we reset it to what it was and try the other formats
    if isinstance(data, ParameterSet):
        # Special case of data with `save` attribute
        _selected_formats_back = selected_formats
        selected_formats = []  # Don't save to another format if successful
        with get_output(filename, ext="", bytes=False, overwrite=overwrite) as (f, output_path):
            # Close the file since Parameters only accepts urls as filenames
            # FIXME: This introduces a race condition; should use `f` to save
            #        This would require fixing the parameters package to
            #        accept file objects in `save()`
            pass
        try:
            logger.info("Saving ParameterSet using its own `save` method...")
            data.save(output_path, expand_urls=True)
        except (AttributeError, PermissionError) as e:
            logger.warning("Calling the data's `save` method failed with '{}'."
                           .format(str(e)))
            selected_formats = _selected_formats_back
        else:
            output_paths.append(output_path)
    elif hasattr(data, 'save'):
        _selected_formats_back = selected_formats
        selected_formats = []  # Don't save to another format if successful
        # See if this type is in the registered formats, so we can get the
        # expected extension
        typename = find_registered_typename(data)
            # Always returns a type name: if none is found, returns that of data
        format = _format_types.get(typename, None)
        if format is None or format not in defined_formats:
            ext = ""
        else:
            ext = defined_formats[format].ext
        with get_output(filename, ext=ext, bytes=False, overwrite=overwrite) as (f, output_path):
            # TODO: Use `f` if possible, and only `output_path` if it fails.
            pass
        try:
            logger.info("Saving data using its own `save` method...")
            data.save(output_path)
        except (AttributeError, PermissionError) as e:
            logger.warning("Calling the data's `save` method failed with '{}'."
                           .format(str(e)))
            selected_formats = _selected_formats_back
        else:
            output_paths.append(output_path)

    # Save to all specified formats
    for name, formatinfo in defined_formats.items():
        if name in ('npr', 'repr', 'brepr', 'dill'):
            # TODO: Define the save functions below at top level of module
            # and treat these formats as any other
            #       Make sure 'dill' is still used as backup
            continue
        if name in selected_formats:
            if formatinfo.save is None:
                logger.error("Format '{}' does not define a save function"
                             .format(name))
                fail = True
            else:
                fail = False
                ext = formatinfo.ext
                try:
                    with get_output(filename, ext, formatinfo.bytes, overwrite) as (f, output_path):
                        formatinfo.save(f, data)
                except IOError:
                    fail = True
                except Exception as e:
                    logger.error("Silenced uncaught exception during saving process to attempt another format.")
                    logger.error("Silenced exception was: " + str(e))
                    fail = True
                else:
                    output_paths.append(output_path)
            if fail:
                try: os.remove(output_path)  # Ensure there are no leftover files
                except: pass
                logger.warning("Unable to save to {} format."
                               .format(name))
                if 'dill' not in selected_formats:
                    # Warn the user that we will use another format
                    logger.warning("Will try a plain (dill) pickle dump.")
                    selected_formats.add('dill')
    # Save data as numpy representation
    if 'npr' in selected_formats:
        fail = False
        ext = defined_formats['npr'].ext
        try:
            with get_output(filename, ext, True, overwrite) as (f, output_path):
                try:
                    logger.info("Saving data to 'npr' format...")
                    np.savez(f, **data.repr_np)
                except AttributeError:
                    fail = True
                else:
                    output_paths.append(output_path)
        except IOError:
            fail = True
        if fail:
            # TODO: Use custom error type
            try: os.remove(output_path)  # Ensure there are no leftover files
            except: pass
            logger.warning("Unable to save to numpy representation ('npr') format.")
            if 'dill' not in selected_formats:
                # Warn the user that we will use another format
                logger.warning("Will try a plain (dill) pickle dump.")
                selected_formats.add('dill')

    # Save data as representation string ('repr' or 'brepr')
    for format in [format
                   for format in selected_formats
                   if format in ('repr', 'brepr')]:
        bytes = (format == 'brepr')
        fail = False
        if data.__repr__ is object.__repr__:
            # Non-informative repr -- abort
            fail = True
        else:
            ext = defined_formats['repr'].ext
            try:
                with get_output(filename, ext=ext, bytes=bytes, overwrite=overwrite) as (f, output_path):
                    try:
                        logger.info("Saving data to plain-text 'repr' format'")
                        f.write(repr(data))
                    except:
                        fail = True
                    else:
                        output_paths.append(output_path)
            except IOError:
                fail = True
        if fail:
            try: os.remove(output_path)  # Ensure there are no leftover files
            except: pass
            logger.warning("Unable to save to numpy representation ('npr') format.")
            if 'dill' not in selected_formats:
                # Warn the user that we will use another format
                logger.warning("Will try a plain (dill) pickle dump.")
                selected_formats.add('dill')

    # Save data in dill format
    if 'dill' in selected_formats:
        ext = defined_formats['dill'].ext
        try:
            with get_output(filename, ext, True, overwrite) as (f, output_path):
                logger.info("Saving data as a dill pickle.")
                dill.dump(data, f)
                output_paths.append(output_path)
        except IOError:
            # There might be other things to save, so don't terminate
            # execution because this save failed
            try: os.remove(output_path)  # Ensure there are no leftover files
            except: pass
            logger.warning("Unable to save picke at location {}."
                           .format(output_path))

    # Return the list of output paths
    return [Path(path) for path in output_paths]

def find_file(file, format=None):
    """
    If `file` has no extension and `format` is None:
    Returns the list of paths matching the file. If the list has length one,
    a simple path is returned
    Otherwise, returns the file path with the specified format.
    In both cases, if no file is found, raises FileNotFoundError.

    Parameters
    ----------
    file: str | path-like
    format: str | type
        Should correspond to an entry in `defined_formats`.
    """
    if isinstance(file, PathTypes):
        basepath, ext = os.path.splitext(file)
        dirname, basename = os.path.split(basepath)
        if dirname == '':
            dirname = '.'

        # Normalize the format and see if it is recognized
        if isinstance(format, str):
            formatext = defined_formats.get(format, None).ext
        elif isinstance(format, type):
            def_format = defined_formats.get(find_registered_typename(format), None)
            if def_format is not None:
                formatext = def_format.ext
            else:
                warn("The type {} does not correspond to one of the types "
                     "registered with `smttask.iotools.register_datatype`"
                     .format(format))
                formatext = None
        else:
            formatext = None

        # Find the file
        if len(ext) > 0 and formatext is None:
            format = ext.strip('.')
        if len(ext) == 0 and formatext is None:
            # Try every file whose name without extension matches `file`
            match = lambda fname: os.path.splitext(fname)[0] == basename
            fnames = [name for name in os.listdir(dirname) if match(name)]
            # Order the file names so we try most likely formats first (i.e. npr, repr, dill)
            # We do not attempt to load other extensions, since we don't know the format
            ordered_fnames = []
            for formatinfo in defined_formats.values():
                name = basename + '.' + formatinfo.ext
                if name in fnames:
                    ordered_fnames.append(os.path.join(dirname, name))
            if len(ordered_fnames) == 0:
                # No file was found
                raise FileNotFoundError(
                    "No file with base name '{}' and a recognized extension was "
                    "found.\nDirectory searched: {}"
                    .format(basename, dirname))
            elif len(ordered_fnames) == 1:
                ordered_fnames = ordered_fnames[0]
            return ordered_fnames

        elif os.path.exists(file):
            return file
        else:
            return basepath + "." + formatext.strip('.')
    else:
        raise ValueError("File finding is only implemented for path-like "
                         "arguments.")

def load(file, types=None, load_function=None, format=None, input_format=None):
    """
    Load file at `file`. How the data is loaded is determined by the input format,
    which is inferred from the filename extension. It can also be given by `format`
    explicitly.
    If `load_function` is provided, it is be applied to the loaded data. Otherwise,
    we try to infer type from the loaded data.
      - For 'npr' data, we look for a 'type' entry.
      - For 'repr' data, we look at the first part of the returned string, before
        any whitespace or special characters.
    Type inference can be completely disabled by passing `load_function=False`.
    If a type is found, we then try to match it to one of the entries in `types`. Thus
    `types` should be a dictionary of str:Type pairs, where the keys match the type name
    stored in files, and the Type is a loaded Python class which is associated to that
    name; typically, the type name will be the class name, although that is not required.
    It is also possible to add to the list of default types by calling this module's
    `add_load_type` method.

    Parameters
    ----------
    file: str | file object (TODO)

    types: dict
        (Optional)
    load_function: function
        (Optional) Post-processing function, called on the result of the loaded
        data. I.e. does not override `type`, but provides a handle to process
        the result.
    format: str
        Specify the format of the data; overrides any deductions from the
        filename. Effectively this specifies the loading function, and thus
        should correspond to a key in `types`.
    input_format: str (DEPRECATED)
        Deprecated synonym for `format`.
    """
    global _load_types
    if types is not None:
        types = {**_load_types, **dict(types)}
    else:
        types = _load_types

    if format is None:
        format = input_format

    if isinstance(file, PathTypes):
        # _, ext = os.path.splitext(file)
        # if len(ext) > 0 and format is None:
        #     format = ext.strip('.')
        ordered_fnames = find_file(file, format)
        if isinstance(ordered_fnames, list):
            # Try to load every file name in sequence. Terminate after the first success.
            for fname in ordered_fnames:  # Empty list -> FileNotFound below
                try:
                    # Recursively call `load`
                    data = load(fname, types, load_function)
                except (FileNotFoundError):
                    # Basically only possible to reach here with a race condition, where
                    # file is deleted after having been listed
                    # TODO: Also catch loading errors ?
                    continue
                else:
                    return data
            # No file was found
            raise FileNotFoundError(
                "No file with base name '{}' and a recognized extension was "
                "found.\nDirectory searched: {}"
                .format(basename, dirname))

        elif os.path.exists(ordered_fnames):
            # Normal exit from `load` recursion
            openfilename = ordered_fnames
            basepath, ext = os.path.splitext(openfilename)
        else:
            # No file was found
            raise FileNotFoundError("No file '{} was found. (Specified "
                                    " format: '{}')".format(file, format))

    elif isinstance(file, io.IOBase):
        ext = os.path.splitext(file.name)[1]
    elif hasattr(file, 'fn'):
        # Treats Luigi LocalTarget
        ext = os.path.splitext(file.fn)[1]
    else:
        raise TypeError("[iotools.load] File '{}' is of unrecognized type '{}'."
                        .format(file, type(file)))

    if format is None:
        format = get_format_from_ext(ext[1:])
    if format not in defined_formats:
        warn("smttask.iotools.load: Unrecognized format `{}`. "
             "Ignoring `format` argument and using file extension instead."
             .format(format))
        format = get_format_from_ext(ext[1:])

    if format not in ('npr', 'repr', 'dill'):
        # TODO: Define following load functions separately, add them to
        # to defined_formats, and then treat these as any other format
        formatinfo = defined_formats[format]
        data = defined_formats[format].load(basepath+ext)
    elif format == 'npr':
        with wrapped_open(file, 'rb') as f:
            data = np.load(f)
                # np.load provides dict access to the file object;
                # but doesn't load it to memory. We must make sure it's in
                # memory before we exit `with` block.
            in_memory = False
            if load_function is False:
                pass
            elif load_function is not None:
                data = load_function(data)
                # Is it safe to assume that `load_function` always puts data in memory ?
                in_memory = True
            elif 'type' in data:
                # 'type' stored as a 0D array
                if (data['type'].ndim == 0
                    and data['type'].dtype.kind in {'S', 'U'}
                    and str(data['type']) in types) :
                    # make sure it's really 0D
                    cls = types[str(data['type'])]
                    if hasattr(cls, 'from_repr_np'):
                        data = cls.from_repr_np(data)
                        in_memory = True
                else:
                    # TODO: Error message
                    pass
            if not in_memory:
                # Since it's still not in memory, just load it as a dictionary.
                data = dict(data)

    elif format == 'repr':
        with wrapped_open(openfilename, 'r') as f:
            data = f.read()
        if load_function is False:
            pass
        elif load_function is not None:
            data = load_function(data)
        else:
            # Search for the type name in the initial data
            if data[0] == '<':
                i, j = data.index('>'), data.index('<')
                if i == -1 or (j > -1 and j < i):
                    # Misformed type specification
                    pass
                else:
                    test_clsname = data[1:i]
                    if test_clsname in types:
                        cls = types[test_clsname]
                        if hasattr(cls, 'from_repr'):
                            data = cls.from_repr(data)
    elif format == 'dill':
        with wrapped_open(openfilename, 'rb') as f:
            try:
                data = dill.load(f)
            except EOFError:
                logger.warning("File {} is corrupted or empty. You should "
                               "delete it to prevent confusion.".format(file))
                raise FileNotFoundError

    return data

class output():
    def __init__(self, path, ext, bytes, overwrite=False):

        # Add extension
        basepath, _ = os.path.splitext(path)
            # Remove possible extension from path
        if len(ext) > 0 and ext[0] != ".":
            ext = "." + ext
        self.output_path = basepath + ext
        self.orig_output_path = self.output_path
        self.overwrite = overwrite
        self.bytes = bytes

    def __enter__(self):
        # Open file
        try:
            if not self.overwrite:
                self.f, self.actual_path = get_free_file(self.output_path, bytes=self.bytes)
            else:
                mode = 'wb' if self.bytes else 'w'
                self.f = open(self.orig_output_path, mode)
                self.actual_path = self.orig_output_path
        except IOError:
            logger.error("Could not create a file at '{}'."
                         .format(self.orig_output_path))
            raise

        return self.f, self.actual_path

    def __exit__(self, type, value, traceback):
        self.f.close()

class dummy_file_context:
    def __init__(self, file):
        self.f = file

    def __enter__(self):
        return self.f, os.path.realpath(self.f.name)

    def __exit__(self, type, value, traceback):
        # Since the file was not created in this context, don't close it
        pass

class wrapped_open:
    # Acts like `open`, but if given a file object, just returns the object.
    def __init__(self, file, mode):
        self.file = file
        self.mode = mode
        self.exit = None

    def __enter__(self):
        if isinstance(self.file, io.IOBase):
            if self.file.mode != mode:
                logger.warning("Already open file object has mode '{}', but "
                               " '{}' is expected."
                               .format(self.file.mode, self.mode))
            self.f = self.file
            self.exit = False  # Don't close a file we didn't open
        elif hasattr(self.file, 'open'):
            # Treats Luigi LocalTarget
            self.f = self.file.open(self.mode)
            self.exit = True
        else:
            self.f = open(self.file, self.mode)
            self.exit = True
        return self .f

    def __exit__(self, type, value, traceback):
        if self.exit is None:
            logger.warning("ERROR [iotools.wrapped_open]: `self.exit()` was "
                           "never set.")
        if self.exit:
            self.f.__exit__()
