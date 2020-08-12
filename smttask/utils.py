import os.path
from pathlib import Path

def relative_path(src, dst, through=None, resolve=True):
    """
    Like pathlib.Path.relative_to, with the difference that `dst` does not
    need to be a subpath of `src`.

    In typical use, `src` would be point to directory, and `dst` to a file.

    Parameters
    ----------
    src: Path-like
        Returned path starts from here.
    dst: Path-like
        Returned path points to here. If ``relpath`` is the returned path, then
        `dst` points to the same location as concatenating `src` and ``relpath``.
    through: Path-like
        Generally does not need to be provided; by default it is obtained with
        `os.path.commonpath`. When provided, the returned path always goes
        through `through`, even when unnecessary.
    resolve: bool
        Whether to normalize both `src` and `dst` with `Path.resolve`.
        It is hard to construct an example where doing this has an undesirable
        effect, so leaving to ``True`` is recommended.

    Examples
    --------
    >>> from smttask.utils import relative_path
    >>> pout = Path("/home/User/data/output/file")
    >>> pin  = Path("/home/User/data/file")
    >>> relative_path(pin, pout, through="/home/User/data")
    PosixPath('../output/file')
    """
    src=Path(src); dst=Path(dst)
    if resolve:
        src = src.resolve()
        dst = dst.resolve()
    if through is None:
        through = os.path.commonpath([src, dst])
    if through != str(src):
        dstrelpath = dst.relative_to(through)
        srcrelpath  = src.relative_to(through)
        depth = len(srcrelpath.parents)
        uppath = Path('/'.join(['..']*depth))
        return uppath.joinpath(dstrelpath)
    else:
        return dst.relative_to(src)

def is_valid_desc(desc, required_keys, optional_keys=None,
                  expected_types=None, on_wrong_desc_type='false'):
    """
    Checks that a description provides a given set of keys.
    Optional keys are not required, but will not trigger a warning if present.

    Parameters
    ----------
    desc: dictionary
    required_keys: list of strings
    optional_keys: list of strings
    expected_types: dict
        For each provided key name, test the corresponding value with
        'isinstance'.
        keys: key names in `required_keys` or `optional_keys`.
        values: types, or tuples of types.
    on_wrong_desc_type: 'false' | 'raise'
        What to do if `desc` is an invalid type.
        'false': Return False (same as a non-Task desc)
        'raise': Raise an TypeError exception

    Returns
    -------
    bool
        True if desc is a valid Task description. False otherwise.

    Raises
    ------
    TypeError
        If `desc` ist not a dictionary and `on_wrong_desc_type=True`.
    """
    if optional_keys is None:
        optional_keys = []
    if expected_types is None:
        expected_types = {}
    if not isinstance(desc, dict):
        if not isinstance(on_wrong_desc_type, str):
            warn("`on_wrong_desc_type` should be a string.")
            on_wrong_desc_type = 'false'
        on_wrong_desc_type = on_wrong_desc_type.lower()
        if on_wrong_desc_type == 'raise':
            raise TypeError("`desc` must be a dictionary.")
        elif on_wrong_desc_type != 'false':
            warn("`on_wrong_desc_type` should be either 'false' or 'raise'.")
        else:
            return False
    # Check that required keys are provided
    desc_keys = set(desc)
    required_keys = set(required_keys)
    if not required_keys <= desc_keys:
        return False
    # Check for unexpected keys
    recognized_keys = required_keys | set(optional_keys)
    unrecognized_keys = desc_keys - recognized_keys
    if len(unrecognized_keys) > 0:
        warn("Description is valid but provides additional unrecognized "
             "keys : {}".format(','.join(unrecognized_keys)))
    # Check value types
    keys_with_wrong_types = {}
    for key, T in expected_types.items():
        if key not in recognized_keys:
            warn(f"Can't check type of unrecognized key '{key}'")
        elif key in desc:
            if not isinstance(desc[key], T):
                keys_with_wrong_types[key] = T
    if len(keys_with_wrong_types) > 0:
        warn("Description seems valid, but the values for the following keys "
             "have the wrong type:\n"
             "\n".join(f"{key}: {T} (should be {expected_types[key]}"
                       for key, T in keys_with_wrong_types.items()))
        return False
    # If we made it here, all checks succeeded (or only printed warnings)
    return True

def is_task_desc(desc, tasks=None,
                 on_task_not_found='warn', on_wrong_desc_type='false'):
    """
    Returns True if the provided `desc` describes a Task.
    Effectively this means checking for the `taskname`, `inputs` and
    `module` keys.

    Parameters
    ----------
    desc: dictionary
    tasks: dict of tasks
        If passed, the taskname will be check to see if it is recognized.
        Action taken depends on `on_task_not_found`
        Ignored if `on_task_not_found = True`.
    on_task_not_found: 'false' | 'true' | 'warn'
        What to do if the taskname is not found in `tasks`.
        This parameter is ignored if `tasks` is None.
        'false': Return False.
        'true':  Return True anyway; no checking is performed.
        'warn':  Return True, but issue a warning.
    on_wrong_desc_type: 'false' | 'raise'
        What to do if `desc` is an invalid type.
        'false': Return False (same as a non-Task desc)
        'raise': Raise an TypeError exception

    Returns
    -------
    bool
        True if desc is a valid Task description. False otherwise.

    Raises
    ------
    TypeError
        If `desc` ist not a dictionary and `on_wrong_desc_type=True`.
    """
    return is_valid_desc(desc,
                         ['taskname', 'inputs', 'module'],
                         expected_types={'taskname': str},
                         on_wrong_desc_type=on_wrong_desc_type)
    # if 'taskname' not in desc:
    #     return False
    # if 'inputs' not in desc:
    #     warn("Description defines a task name, but does not define inputs.")
    #     return False
    # if 'module' not in desc:
    #     warn("Description defines a task name and inputs, but does not "
    #          "specify the module in which it was defined.")
    #     return False
    # if len(desc) > 3:
    #     warn("Description seems like a task, but has additional entries "
    #          "beyond 'taskname' and 'inputs'.")
        # Let code continue and return True
    # If we made it here, we have both 'taskname' and 'inputs' keys
    taskname = desc['taskname']
    # if not isinstance(taskname, str):
    #     warn("Description seems like a task, but it's 'taskname' value "
    #          "is not a string.")
    #     return False
    if (tasks is None
        or taskname in tasks
        or on_task_not_found == 'true'):
        return True
    # Taskname is not in tasks
    if not isinstance(on_task_not_found, str):
        warn("`on_task_not_found` should be a string.")
        on_task_not_found = 'warn'
    on_task_not_found = on_task_not_found.lower()
    if on_task_not_found == 'false':
        return False
    if on_task_not_found != 'warn':
        warn("`on_task_not_found` should be one of 'true', 'false' or 'warn'.")
        # We default to 'warn' by continuing
    warn("Description is a valid task name, but the name '{}' was not found "
         "in the provided task list.")
    return True
