"""
Additional functionality which is smttask-specific.
"""
import numpy as np

from .config import config
from .utils import get_task_param, Singleton
from .view.recordfilter import record_filter

# Sentinel value
class DOES_NOT_HAVE_PARAM_CLASS(metaclass=Singleton):
    def __repr__(self):
        return "<record does not have parameter>"
DOES_NOT_HAVE_PARAM = DOES_NOT_HAVE_PARAM_CLASS()

@record_filter
def task(taskname: str):
    """
    Keep records which were run for task `taskname`. The entire task name must
    be specified, but the match is case insensitive.

    .. Note:: This searches the *script_arguments* entry, which *smttask*
       repurposes to store the task name.
    """
    taskname = taskname.lower()
    def filter_fn(record): return taskname == record.script_arguments.lower()
    return filter_fn

@record_filter
def params(eq=None, *, ge=None, le=None, gt=None, lt=None, isclose=None):
    """
    Keep records for which specific parameters satisfy the given conditions.
    Conditions should be specified as dictionaries; keys for nested parameters
    can be specified with dots between levels.

    Which operation is used to compare the parameter values is specified by
    keyword (if no keyword is given, equality is assumed). The recognized
    keywords and their corresponding operators are as follows:

    - `eq`: Keep records for which record.param = value
    - `ge`: Keep records for which record.param ≥ value
    - `le`: Keep records for which record.param ≤ value
    - `gt`: Keep records for which record.param > value
    - `lt`: Keep records for which record.param < value
    - `isclose`: Keep records for which record.param ≈ value
      Tested as ``np.isclose(record.param, value)``.

    When mulitple conditions are specified (either as dictionaries with
    multiple entries or by giving multiple keyword arguments), only records
    satisfying all conditions are kept.

    Parameters
    ----------
    eq, ge, le, gt, lt, isclose: dict
        See above.

    Example
    -------
    >>> rsview = RecordStoreView()
    >>> rsview.filter.params(eq={'a': 1}, isclose={'φ': 1.57079})
    """
    def filter_fn(record):
        paramset = config.ParameterSet(record.parameters)
        if eq:
            for test_k, test_v in eq.items():
                if np.any(get_task_param(paramset, test_k, DOES_NOT_HAVE_PARAM) != test_v):
                    return False
        if ge:
            for test_k, test_v in ge.items():
                if np.any(get_task_param(paramset, test_k, DOES_NOT_HAVE_PARAM) < test_v):
                    return False
        if le:
            for test_k, test_v in le.items():
                if np.any(get_task_param(paramset, test_k, DOES_NOT_HAVE_PARAM) > test_v):
                    return False
        if gt:
            for test_k, test_v in gt.items():
                if np.any(get_task_param(paramset, test_k, DOES_NOT_HAVE_PARAM) <= test_v):
                    return False
        if lt:
            for test_k, test_v in lt.items():
                if np.any(get_task_param(paramset, test_k, DOES_NOT_HAVE_PARAM) >= test_v):
                    return False
        if isclose:
            for test_k, test_v in isclose.items():
                if not np.all(np.isclose(get_task_param(paramset, test_k, DOES_NOT_HAVE_PARAM), test_v)):
                    return False
        return True
    return filter_fn

@record_filter
def match(ref, eq=None, *, ge=None, le=None, gt=None, lt=None,
          isclose=None, include_reference=True):
    """
    Similar to the `params` `RecordFilter`, with the difference that rather
    than specifying a value explicitely, a reference record or task is provided.
    Records which match the reference in the specified parameters are kept.

    See `params` for definition of the comparison keyword arguments.

    Parameters
    ----------
    ref: Record | Task | dict
        Comparisons are performed between ``get_param(ref, param)``
        and ``get_param(record, param)``.
    eq, ge, le, gt, lt, isclose: list
        See `params`. Specify as list instead of dict since values are
        taken from `ref`.
    include_reference: bool
        Whether to keep the record used for reference. This is only relevant
        if `ref` is `Record`; if it is a `Task` or dict, it is always kept
        (since no test can guarantee that the it is the same).
    """
    # Avoid calling get_task_param for every record by computing it now
    # and reusing `params`.
    if eq is not None:
        eq = {k: get_task_param(ref, k, DOES_NOT_HAVE_PARAM) for k in eq}
    if ge is not None:
        ge = {k: get_task_param(ref, k, DOES_NOT_HAVE_PARAM) for k in ge}
    if le is not None:
        le = {k: get_task_param(ref, k, DOES_NOT_HAVE_PARAM) for k in le}
    if gt is not None:
        gt = {k: get_task_param(ref, k, DOES_NOT_HAVE_PARAM) for k in gt}
    if lt is not None:
        lt = {k: get_task_param(ref, k, DOES_NOT_HAVE_PARAM) for k in lt}
    if isclose is not None:
        isclose = {k: get_task_param(ref, k, DOES_NOT_HAVE_PARAM) for k in isclose}

    filter_fn = params(eq=eq, ge=ge, le=le, gt=gt, lt=lt, isclose=isclose)
    if not include_reference:
        _filter_fn = filter_fn
        def filter_fn(record):
            if record is ref:
                return False
            return _filter_fn(record)

    return filter_fn
