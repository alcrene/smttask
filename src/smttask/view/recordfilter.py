import logging
from typing import Union, Set, Callable
from collections import namedtuple
import sumatra.recordstore

logger = logging.getLogger(__name__)

# Prepended filters can be faster, because they are applied by the underlying
# record store.
PrependedFilter = namedtuple('PrependedFilter', ['name', 'args', 'kwargs'])

# TODO: This could be filled by a subdecorator of the record_filter. E.g.:
# >>> @record_filter
#     def tags(tags):
#         ...
# >>> @tags.django_filter:
#     def tags_django(tags):
#         ...
prependable_filters = {
    sumatra.recordstore.ShelveRecordStore: []
}
if sumatra.recordstore.have_http:
    prependable_filters[sumatra.recordstore.HttpRecordStore] = []
if sumatra.recordstore.have_django:  # sumatra.recordstore has a guard: if django cannot be loaded, it doesn’t define DjangoRecordStore and falls back to ShelveRecordStore
    prependable_filters[sumatra.recordstore.DjangoRecordStore] = ['tags']

class RecordFilter:
    """
    One can overwrite RecordFilter.on_error_defaults to change default behaviour
    for all filters.
    Filters can have three defined behaviours when they catch an error. A common
    need for example is to have the filter catch AttributeError, either to
    reject all elements that don't have a particular attribute (False), or to
    avoid filtering elements that don't have a particular attribute (True). By
    default the `AttributeError` error in `on_error_defaults` is set to False.
      - False: the condition returns False
      - True: the condition returns True
      - 'raise': the error is reraised. The same happens if there is no entry
        corresponding to the error in `on_error_defaults`.
    """
    on_error_defaults = {
        AttributeError: False
    }
    registered_filters = {}

    def __init__(self, record_store_view, filter_fn=None):
        if filter_fn is None:
            filter_fn = generic_filter
        self.rsview = record_store_view
        self.filter_fn = filter_fn
        # self.parameters = ParameterSetFilter(record_list)
        # # Multi-condition filter wrappers
        # self.any = _AnyRecordFilter(self)
        # self.all = _AllRecordFilter(self)
        
    def __dir__(self):  # Defining __dir__ allows autocomplete and dir() to work
        """Return the list of registered filters."""
        return list(self.registered_filters)

    def __call__(self, *args, errors=None, **kwargs):
        on_error = self.on_error_defaults
        if errors is not None:
            on_error.update(errors)
        filter_fn = self.filter_fn(*args, **kwargs)
        def test(rec):
            "Applies `filter_fn` and catches errors defined in `on_error`."
            try:
                return filter_fn(rec)
            except tuple(on_error.keys()) as e:
                if on_error[type(e)] == 'raise':
                    raise
                else:
                    logger.debug("Filtering raised {}. Ignored. (RecordFilter."
                                 "on_error_defaults)".format(str(type(e))))
                    return on_error[type(e)]

        return self.rsview.copy(iterable=filter(test, self.rsview))

    def __getattr__(self, attr):
        # Check if we can use an prepended filter. Requirements:
        #   - Only prepended filters have been applied on top of record store
        #     (checked with `_iterable is None`)
        #   - the particular record store supports that particular filter
        rs_type = type(self.rsview.record_store)
        if (self.rsview._iterable is None
            and rs_type in prependable_filters
            and attr in prependable_filters[rs_type]):
            rsview = self.rsview.copy()
            return PrependedRecordFilter(rsview, attr)
        recfilter = RecordFilter(self.rsview, self.registered_filters[attr])
        docstring = getattr(self.registered_filters[attr], '__doc__', None)
        if docstring:
            recfilter.__doc__ = docstring
        return recfilter

class PrependedRecordFilter(RecordFilter):
    def __init__(self, record_store_view, filter_name: str):
        super().__init__(record_store_view, filter_fn=None)
        self.filter_name = filter_name
    def __call__(self, *args, **kwargs):
        rsview = self.rsview.copy()
        rsview._prepended_filters.append(
            PrependedFilter(self.filter_name, args, kwargs))
        return rsview

def record_filter(fn):
    """
    Decorator which adds a function to the set of registered filters.
    Record filter functions must have the format::

       def my_filter([args]):
           def filter_fn(record: Record) -> bool:
               ...

    .. note:: Because the set of registered filters is kept as a class variable
       of `RecordFilter`, there can be only one set. If more than one set of
       registered filters is really needed, one can subclass `RecordFilter` and
       define a new `record_filter` decorator which adds to the subclass'
       register.

    todo?: Allow specifying a different name as decorator argument ?

    Example
    -------
    >>> from smttask.smtview import RecordStoreView, record_filter
    >>> rsview = RecordStoreView(...)
    >>> @record_filter
        def successful(yes):
            'yes=True(False): Keep only successful (unsuccessful) records.'
            def filter_fn(record):
                return ('_finished_' in record.tags) == yes
            return filter_fn
    >>> rsview.filter.successful(yes=True)
    """
    name = fn.__name__
    RecordFilter.registered_filters[name] = fn
    return fn

@record_filter
def generic_filter(fn: Callable):
    """
    The default filter: keep records for which `fn` returns True.
    Equivalent to Python's `filter`.
    """
    def filter_fn(record):
        return fn(record)
    return filter_fn

#######################
## Builtin filters

from datetime import datetime, timedelta

# TODO: A parameters filter. See smttask.task_filters

@record_filter
def before(date, *args):
    """
    Keep only records which occured before the given date. Date is exclusive.
    Can provide date either as a single tuple, or multiple arguments as for
    `datetime.datetime()`; a `datetime` instance is also accepted.

    As a convenience, tuple values may be concatenated and replaced by a
    single integer of 4 to 8 digits; if it has less than 8 digits, it is
    extended to the earliest date (so 2018 -> 20180101).
    """
    if isinstance(date, datetime):
        if len(args) > 0:
            raise ValueError("Too many arguments for `filter.before()`")
    elif isinstance(date, tuple):
        date = datetime(*(date+args))
    elif isinstance(date, int) and len(str(date)) <= 8 and len(args) == 0:
        # Convenience interface to allow dropping the commas
        datestr = str(date)
        if len(datestr) < 4:
            raise ValueError("Date integer must give at least the year.")
        elif len(datestr) < 8:
            Δi = 8 - len(datestr)
            datestr = datestr + "0101"[-Δi:]
            date = int(datestr)
        year, month, day = date//10000, date%10000//100, date%100
        date = datetime(year, month, day)
    else:
        date = datetime(date, *args)
    if not isinstance(date, datetime):
        tnorm = lambda tstamp: tstamp.date()
    else:
        tnorm = lambda tstamp: tstamp

    def filter_fn(record):
        return tnorm(record.timestamp) < date
    return filter_fn

@record_filter
def after(date, *args):
    """
    Keep only records which occurred after the given date. Date is inclusive.
    Can provide date either as a single tuple, or multiple arguments as for
    `datetime.datetime()`; a `datetime` instance is also accepted.

    As a convenience, tuple values may be concatenated and replaced by a
    single integer of 4 to 8 digits; if it has less than 8 digits, it is
    extended to the earliest date (so 2018 -> 20180101).
    """
    if isinstance(date, datetime):
        if len(args) > 0:
            raise ValueError("Too many arguments for `filter.after()`")
    elif isinstance(date, tuple):
        date = datetime(*(date+args))
    elif isinstance(date, int) and len(str(date)) <= 8 and len(args) == 0:
        # Convenience interface to allow dropping the commas
        # Date can be an integer of length 4, 5, 6, 7 or 8; if less than 8
        # digits, will be extended with the earliest date (so 2018 -> 20180101)
        datestr = str(date)
        if len(datestr) < 4:
            raise ValueError("Date integer must give at least the year.")
        elif len(datestr) < 8:
            Δi = 8 - len(datestr)
            datestr = datestr + "0101"[-Δi:]
            date = int(datestr)
        year, month, day = date//10000, date%10000//100, date%100
        date = datetime(year, month, day)
    else:
        date = datetime(date, *args)
    if not isinstance(date, datetime):
        tnorm = lambda tstamp: tstamp.date()
    else:
        tnorm = lambda tstamp: tstamp

    def filter_fn(record):
        return tnorm(record.timestamp) >= date
    return filter_fn

@record_filter
def on(date, *args):
    """
    Keep only records which occurred on the given date.
    Can provide date either as a single tuple, or multiple arguments as for
    `datetime.datetime()`; a `datetime` instance is also accepted.

    As a convenience, tuple values may be concatenated and replaced by a
    single integer of 4 to 8 digits; if it has less than 8 digits, it is
    extended to the earliest date (so 2018 -> 20180101).

    .. TODO:: ``.filter.on(202011)`` is equivalent to ``.filter.on(20201101)``, when arguably
       it would be more intuitively equivalent to ``.filter.after(202011).filter.before(202012)``
    """
    if isinstance(date, datetime):
        if len(args) > 0:
            raise ValueError("Too many arguments for `filter.on()`")
    elif isinstance(date, tuple):
        date = datetime(*(date+args))
    elif isinstance(date, int) and len(str(date)) <= 8 and len(args) == 0:
        # Convenience interface to allow dropping the commas
        datestr = str(date)
        if len(datestr) < 4:
            raise ValueError("Date integer must give at least the year.")
        elif len(datestr) < 8:
            Δi = 8 - len(datestr)
            datestr = datestr + "0101"[-Δi:]
            date = int(datestr)
        year, month, day = date//10000, date%10000//100, date%100
        date = datetime(year, month, day)
    else:
        date = datetime(date, *args)
    after = date
    before = date + timedelta(days=1)

    if not isinstance(date, datetime):
        tnorm = lambda tstamp: tstamp.date()
    else:
        tnorm = lambda tstamp: tstamp

    def filter_fn(record):
        return tnorm(record.timestamp) >= date and tnorm(record.timestamp) < before
    return filter_fn

@record_filter
def label(substr: str):
    """Keep records for which the label contains `substr`."""
    def filter_fn(record): return substr in record.label
    return filter_fn

@record_filter
def output(minimum=1, maximum=None):
    """
    Keep only records whose number of output files is between `minimum`
    and `maximum`.
    """
    def filter_fn(record):
        return ((minimum is None or len(record.output_data) >= minimum)
                 and (maximum is None or len(record.output_data) <= maximum))
    return filter_fn

@record_filter
def outcome(substr: str):
    """Keep records for which the “outcome” value contains `substr`."""
    def filter_fn(record): return substr in record.outcome
    return filter_fn

@record_filter
def outcome_not(substr: str):
    """Keep records for which the “outcome” value does not contain `substr`."""
    def filter_fn(record): return substr not in record.outcome
    return filter_fn

@record_filter
def outputpath(substr: str):
    """Keep records for which at least one output file path contains `substr`."""
    def filter_fn(record):
        return any(substr in path for path in record.outputpath)
    return filter_fn

@record_filter
def reason(substr: str):
    """Keep records for which the “reason” value contains `substr`."""
    def filter_fn(record):
        return any(substr in line for line in record.reason)
    return filter_fn

@record_filter
def reason_not(substr: str):
    """Keep records for which the “reason” value does not contains `substr`."""
    def filter_fn(record):
        return not any(substr in line for line in record.reason)
    return filter_fn

@record_filter
def script(substr: str):
    """Keep records for which the “main_file” value contains `substr`."""
    def filter_fn(record): return substr in record.main_file
    return filter_fn

@record_filter
def script_arguments(substr: str):
    """Keep records for which the “script_arguments” value contains `substr`."""
    def filter_fn(record): return substr in record.script_arguments
    return filter_fn

@record_filter
def stdout_stderr(substr: str):
    """Keep records for which the “stdout_stderr value contains `substr`."""
    def filter_fn(record): return substr in record.stdout_stderr
    return filter_fn

@record_filter
def tags(tags: Union[Set[str], str]):
    """Keep records containing all the specified tags."""
    tags = {tags} if isinstance(tags, str) else set(tags)
    def filter_fn(record):
        return tags <= record.tags
    return filter_fn
tags_and = tags

@record_filter
def tags_or(tags: Union[Set[str], str]):
    """Keep records containing at least one of the specified tags."""
    tags = {tags} if isinstance(tags, str) else set(tags)
    def filter_fn(record):
        return tags & record.tags
    return filter_fn

@record_filter
def tags_not(tags: Union[Set[str], str]):
    """Keep records containing none of the specified tags."""
    tags = {tags} if isinstance(tags, str) else set(tags)
    def filter_fn(record):
        return not (tags & record.tags)
    return filter_fn

@record_filter
def user(substr: str):
    """Keep records for which the “user” value contains `substr`."""
    def filter_fn(record): return substr in record.user
    return filter_fn

@record_filter
def version(prefixstr: str):
    """Keep records for which the “version” begins with `prefixstr`."""
    def filter_fn(record): return record.version.startswith(prefixstr)
    return filter_fn
