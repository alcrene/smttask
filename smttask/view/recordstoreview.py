from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Iterable
from collections import namedtuple
import re
import itertools
from functools import partial
from typing import Union, Callable, Sequence, List, Tuple, Dict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from parameters import ParameterSet
import sumatra.projects
from sumatra.records import Record
from sumatra.recordstore import RecordStore
from smttask.base import Task
from smttask import _utils

try:
    # These are used only for visualization; we want to degrade gracefully
    # if they are not installed.
    import holoviews as hv
    def hv_is_ready() -> bool:
        if not hv.util.settings.list_backends():
            logger.error("Before calling a visualization function using "
                         "HoloViews, make sure at least one backend is loaded.\n"
                         "You can do this by executing `hv.extension('bokeh')`.")
            return False
        else:
            return True
except ImportError:
    def hv_is_ready() -> bool:
        logger.error("Holoviews is required to use this this visualization "
                     "function.")

import mackelab_toolbox as mtb
import mackelab_toolbox.utils
import mackelab_toolbox.parameters

from .recordview import RecordView
from .recordfilter import RecordFilter
from .config import config

logging.basicConfig()   # In case user has not initialized logging
logger = logging.getLogger(__name__)

###############
# Utility functions

def _rename_to_free_file(path):
    new_f, new_path = iotools.get_free_file(path, max_files=100)
    new_f.close()
    os.rename(path, new_path)
    return new_path

##################################
# Custom errors
class RecordNotFound(Exception):
    pass

##################################
# RecordStoreView

# RecordStore Iterators
# Ideally each sumatra.RecordStore would define __iter__, and then the
# implementation of .list() would just be `list(self)`.
# Unfortunately that isn't the case, so as a workaround, we copy the
# iteration definition from each RecordStore below

from sumatra.recordstore import DjangoRecordStore, HttpRecordStore, ShelveRecordStore
from textwrap import dedent

def _shelve_rs_iter(rsview):
    try:
        data = self.record_store.self[self.project.name]
    except KeyError:
        return iter([])
    else:
        for prefilter in rsview._prepended_filters:
            raise RuntimeError(
                f"ShelveRecordStore does not accept {prefilter.name} as a "
                "prepended filter. This is likely a bug in the filter definition.")
        return iter(data.values())
# def _http_rs_iter(rsview):
#     raise NotImplementedError  # Implementation will wait until I have a need for this
def _django_rs_iter(rsview):
    db_records = rsview.record_store._manager.filter(
        project__id=rsview.project.name).select_related()
    for prefilter in rsview._prepended_filters:
        if prefilter.name != 'tags':
            raise RuntimeError(
                f"ShelveRecordStore does not accept {prefilter.name} as a "
                "prepended filter. This is likely a bug in the filter definition.")
        else:
            tags = prefilter.args + tuple(prefilter.kwargs.values())
            if len(tags) != 1:
                raise TypeError("'tags' filter expects exactly one argument. "
                                f"(received {len(tags)})")
            tags = tags[0]
            if not hasattr(tags, "__len__"):
                tags = [tags]
            for tag in tags:
                db_records = db_records.filter(tags__contains=tag)
    for db_record in db_records:
        try:
            yield db_record.to_sumatra()
        except Exception as err:
            errmsg = dedent("""\
                Sumatra could not retrieve the record from the record store.
                Your parameters may have been recorded in a format which
                Sumatra is unable to parse. Another possibility is that your
                record store was created with an older version of Sumatra.
                Please see http://packages.python.org/Sumatra/upgrading.html for
                information on upgrading. The original error message was: '%s:
                %s'""" % (err.__class__.__name__, err))
            raise err(errmsg)

_rs_iter_methods = {
    DjangoRecordStore: _django_rs_iter,
    # HttpRecordStore: _http_rs_iter,
    ShelveRecordStore: _shelve_rs_iter
}

def _make_path_relative(path, rsview: RecordStoreView=None):
    """Attempt to create a path relative to `root`, otherwise return `path`."""
    if rsview is None:
        rsview = RecordStoreView
    root = rsview.default_project_dir
    if root:
        try:
            path = Path(path).relative_to(root)
        except ValueError:
            pass
    return path

class RecordStoreView:
    """
    This class provides a read-only view to a subset of a `~sumatra.recordstore.RecordStore`.
    It is composed of two parts: the underlying record store, and an iterable
    which iterates over all or some of the records of that record store.
    A `RecordStoreView` provides an iterator which ensures that all returned
    elements RecordViews and therefore also read-only.

    The given iterable is not automatically converted to a list – a version
    underlined by a list (which thus supports len(), indexing, etc.) can be
    obtained with the .list property.
    `RecordStoreView` objects provide a filter interface. If `rsview` is such
    an object, then
        - `rsview.filter(cond)` is the same as `filter(rsview, cond)`
        - `rsview.filter.output_data()` keeps only records which produced output.
    The list of defined filters is stored in the dictionary
    ``rsview.filter.registered_filters``. Custom filters can also be defined
    with the decorator `smttask.view.recordfilter.record_filter`.

    Exception to the 'read-only' property: the methods `add_tag` and `remove_tag`
    must for obvious reasons modify the underlying record store.

    .. caution:: If constructed from a generator, iterating over the RecordStoreView
       will consume the generator. Use the `.list` property to ensure the
       used iterable is non-consumable.
    """
    default_project_dir = None
    # When displaying a summary of all records in a RecordStore, each
    # summary field produces one figure. The names must match record field
    # names exactly.
    # Currently the only implemented visualization is a histogram, so the
    # fields must have numerical values.
    summary_fields: ClassVar[List[str]] = ['timestamp', 'duration']
    field_initializers: ClassVar[Dict[str,Callable]] = \
        {'duration' : lambda dur: pd.Timedelta(dur, 's'),
         'reason'   : lambda r: "\n".join(r),
         'main_file': _make_path_relative,
         'version'  : lambda s: s[:8]}

    def __init__(self, iterable=None, project=None):
        """
        The typical use is to construct `RecordStoreView` from within the
        project, without parameters: `project` and `iterable` are then
        determined automatically.

        Main differences with sumatra RecordStore:

        - Read-only:
          Record store attributes are exposed as properties, and can't be
          accidentally modified.
          (However: `self.recordstore` returns the record store itself, and so
          is not read-only.)
        - Cached list: the first time `list` is called, the result is cached.
          With large record stores, this can save many seconds each time we
          iterate over records.
        - Many more filtering possibilities.
          `sumatra.recordstore.RecordStore` only allows filtering by tag;
          in contrast, `RecordStoreView` allows using arbitrary functions as
          filters, and provides a collection of builtin ones.
        - The `rebuild_input_datastore` method. This is used when a new version
          of the project code changes the expected location of input files;
          `rebuild_input_datastore` can then be used to create symlinks in
          all the expected input locations, without touching the original data.

        Parameters
        ----------
        iterable: Iterable | RecordStore
            In particular, `iterable` may be a generator obtained by filtering
            a `RecordStoreView`.
        project: sumatra.projects.Project
        .. note:: If there is a need, we could allow specifying the project by
           name, along with a record store.
        """
        self.filter = RecordFilter(self)
        self._prepended_filters = []  # List of filters which can be applied
                                      # through the DB interface
        # Basic logic: if _iterable is None, then the view should be of the
        # entire record store (this is where the _rs_iter_methods are used).
        self._iterable = iterable
        if isinstance(iterable, RecordStoreView):
            if project is not None and iterable.project is not project:
                raise ValueError("If `project` is provided, it must match the "
                                 "project associated with the given RecordStoreView.")
            self._project = iterable.project
        else:
            self._project = project or sumatra.projects.load_project(self.default_project_dir)
            if isinstance(iterable, RecordStore):
                if self._project.record_store is not iterable:
                    raise ValueError("If `project` is provided, it must match the "
                                     "project associated with the given RecordStore.")
                self._iterable = None

        self._labels = None
        
        self.field_initializers = self.field_initializers.copy()
        self.field_initializers['main_file'] = partial(_make_path_relative, rsview=self)

    def copy(self, iterable=None):
        """
        Return a new `RecordStoreView` for the same record store.
        The view will include the same records as `self`, unless `iterable`
        is specified.
        """
        if iterable is None:
            iterable = self._iterable
        rsview = type(self)(iterable, self.project)
        rsview._prepended_filters = self._prepended_filters.copy()
        return rsview

    def splitby(self,
                split_fields: Sequence[str],
                split_names: Sequence[str]=None,
                drop_unused_split_fields: bool=True,
                get_field_value: Optional[Callable[[Any, str, Any], Any]]=None
                ) -> Dict[Tuple[str], RecordStoreView]:
        """
        Split the RecordStoreView into multiple, disjoint views, based on their
        values in the fields specified in `split_fields`.
        This is analogous to a 'groupby' operation.
        Grouping is done in the order in which fields appear in `split_fields`.

        Parameters
        ----------
        split_fields: List[str]
            The record attributes used to split record store views. See above
            for the treatment of hierarchical parameters.
        split_names: List[str]
            List of field names to use in the keys of the returned dictionaries.
            If not provided, inferred from `split_fields`.
        drop_unused_split_fields: bool
            Whether to omit from the key type fields which lead to no splitting.
            If True, fields are removed if they satisfy one of two conditions:
            1) All records have the same value for that field.
               (E.g. if 'α' is given in the split fields, but all records have
               the same value for 'α'.)
               This is independent of the ordering in `split_fields`.
            2) If higher priority fields would produce the same split.
               (E.g. if setting `split_fields` to either ['α', 'β'] or ['α']
               would produce the same splits; i.e. the values of 'β' are
               constant when conditioned on 'α'.)
               This depends on the ordering, with rightmost fields removed first.
            Default is True.
        get_field_value: Callable
            The function to use to recover field values from records.
            This is used e.g. by *smttask* to simplify the specification of
            parameters from nested task specifications.
            The function should have the same signature as `getattr`:
            (obj, attr:str, default) -> obj.attr or default
            If unspecified, it defaults to `view.config.get_field_value`.

        Returns
        -------
        dict: {multikey: RecordStoreView}
            *multikey* is a `namedtuple`, of same length as `split_fields`.
            It stores the value for each of these fields in that group.
            Its field names are determined by `split_names`.
        """
        # TODO: Allow namedtuple as `split_names` parameter
        # Normalize arguments
        if split_names is None:
            # Construct split names from `split_fields`. Because we use
            # `namedtuple`, names cannot contain periods.
            split_names = [s.rsplit('.', 1)[-1] for s in split_fields]
            # Ensure names are unique
            if len(split_names) != len(set(split_names)):
                raise ValueError("The split names inferred from `split_columnns` "
                                 f"are not unique: {split_names}. Please "
                                 "provide a `split_names` argument.")
        SplitKey = namedtuple('SplitKey', split_names)
        if get_field_value is None:
            get_field_value = config.get_field_value

        # Construct a dataframe containing the fields to split on
        df_data = []
        for record in self:
            df_data.append([str(get_field_value(record, colname, "<No value>"))
                            for colname in split_fields])
        df = pd.DataFrame(df_data, index=self.labels(), columns=split_fields)
        # Extract a list of columns who's values are used (in order) in groupby. `.apply(str)` ensures values are hashable
        split_values = [df[split_level] for split_level in split_fields]
        # Create a dict of RSViews. Keys are tuples, each element corresponding to one split level
        split_views = {SplitKey(*(k if isinstance(k, tuple) else (k,)))  # groupby keys are only wrapped in tuple if longer than 1
                        : type(self)([self.get(lbl) for lbl in lbls])
                       for k, lbls in df.groupby(split_values).groups.items()}
        # Check if we can remove unused keys  (There's probably a more elegant way than this)
        if drop_unused_split_fields:
            # Two reasons to drop a field:
            # 1) All records have the same value for that field
            #    (drops fields in any position)
            split_keys_vals = {field: set() for field in SplitKey._fields}
            for key in split_views.keys():
                for field, val in zip(SplitKey._fields, key):
                    split_keys_vals[field].add(val)
            # Use list to keep order
            fields_to_drop = [k for k,v in split_keys_vals.items() if len(v) <= 1]
            # 2) The field is not required to distinguish records
            #    (drops fields right-to-left until a minimal distinguishing set is reached)
            split_keys = split_views.keys()
            reduced_keys = [()]*len(split_keys)
            reduced_key_components = []
            for i, key_comp in enumerate(SplitKey._fields):
                reduced_keys = [red_key + (full_key[i],)
                                for full_key, red_key in zip(split_keys, reduced_keys)]
                reduced_key_components.append(key_comp)
                if len(set(reduced_keys)) == len(split_keys):
                    # The (possibly reduced) set of keys is already sufficient to distinguish all record stores
                    for field in SplitKey._fields[i+1:]:
                        if field not in fields_to_drop:
                            fields_to_drop.append(field)
                    break
            if len(fields_to_drop):
                NewSplitKey = namedtuple('SplitKey',
                                        [field for field in SplitKey._fields
                                         if field not in fields_to_drop])
                new_keys = []
                for key in split_views.keys():
                    new_keys.append(NewSplitKey(
                        *(val for field,val in zip(SplitKey._fields,key)
                          if field not in fields_to_drop)))
                assert len(new_keys) == len(split_views)
                split_views = {new_key: view for new_key, view in
                                zip(new_keys, split_views.values())}
        return split_views

    # FIXME: This function will be confusing with a consumable iterator
    # (specifically, calling iter() twice returns two coupled iterators, each
    # advancing the other)
    # Solution:
    #   - Define _partial_list (initialized to [])
    #   - On first call, __iter__ returns an iterator α which also adds values
    #     to _list when they are returned
    #   - On subsequent calls, return mixed iterator: yield from _partial_list
    #     until it is exhaused, then yield from α.
    #   - Once _iterable is exhausted, assign _partial_list to _iterable
    def __iter__(self):
        """
        Return an iterator over the records.
        If a list has already been constructed (usu. by calling `.list`), it
        is used to avoid querying the record store again.
        Sumatra `Record`s are converted to `RecordView`s before being returned.
        """
        if isinstance(self._iterable, list):
            it = iter(self._iterable)
        elif self._iterable is None:
            try:
                it = _rs_iter_methods[type(self.record_store)](self)
            except KeyError:
                # This always works, but defeats the purpose of using an iterator
                it = iter(self.record_store.list(self.project.name))
        else:
            import collections.abc
            if not isinstance(self._iterable, (filter, collections.abc.Sequence)):
                raise NotImplementedError(
                    "There are issues with consumable iterators for records "
                    "which should be resolved before using them.")
            it = iter(self._iterable)
        for record in it:
            if isinstance(record, RecordView):
                # Skip the unecessary casting step
                yield record
            elif isinstance(record, Record):
                yield RecordView(record)
            else:
                raise ValueError(f"A RecordStoreView may only be composed of sumatra "
                                 "records, but this one contains element(s) of "
                                 f"type '{type(record)}'")

    def __len__(self):
        try:
            return len(self._iterable)
        except TypeError as e:
            return NotImplemented  # Because `list` calls `__len__`, raising
                                   # an error here would prevent it from working

    def __getitem__(self, key):
        if isinstance(key, int) and isinstance(self._iterable, list):
            return self._iterable[key]
        elif isinstance(key, str):
            return self.get(key)
        elif self._iterable is None:
            raise RuntimeError("RecordStoreView was not constructed from an "
                               "iterable. Call `.list` before using indexing.")
        else:
            res = self._iterable[key]
        if isinstance(res, Iterable):
            res = type(self)(res)
        elif isinstance(res, Record):
            res = RecordView(res)
        return res

    def rebuild_input_datastore(
        self,
        link_creation_function: Callable[[Record], List[Tuple[Path, Path]]]):
        """
        Iterate through the record store, recompute the output file links for
        each record and recreate all the links in the input data store (i.e.
        on the file system) to match the recompute names.

        Parameters
        ----------
        link_creation_function: Callable
            (record) -> [(link location 1, link target 1), (link location 2, link target 2), ... ]
            Both *link location* and *link target* should be relative to the
            roots of the input and output data stores respectively.
        """
        inroot = Path(config.project.input_datastore.root)
        outroot = Path(config.project.data_store.root)
        # TODO: Find a way to loop backwards without consuming the RS view first

        symlinks = {}
        # `symlinks` is indexed by 'link location', and the record list iterated
        # from oldest to newest, so that newer records overwrite older ones
        logger.info("Iterating through records...")
        for record in tqdm(self.filter.output(minimum=1).list[::-1]):
            # The output() filter ensures we don't waste time with records that
            # produced no output.
            try:
                rel_symlinks = link_creation_function(record)
            except Exception:
                # We are permissive here: especially in early stages of a project,
                # misformed records are not unlikely and it's fine to skip
                # over them
                logger.debug(f"Skipped record {record.label}: input symlinks could not be computed.")
                continue
            else:
                for link_location, link_target in rel_symlinks:
                    abs_target_path = (outroot/link_target).resolve()
                        # Raises error if the path does not exist
                    abs_location_path = inroot/link_location
                    symlinks[abs_location_path] = _utils.relative_path(
                        abs_location_path.parent, abs_target_path)

        # Create all the symlinks
        # Iterate through `symlinks` and create them the links defined therein.
        # If a file already exists where we want to place a link, we do the
        # following:
        #   - If it's a link that already points to the right location, do nothing
        #   - If it's a link that points to another location, replace it
        #   - If it's an actual file, append a number to its filename before
        #     creating the link.
        logger.info("Creating symlinks...")
        num_created_links = 0
        for link_loc, rel_target in tqdm(symlinks.items()):
            src = link_loc.parent/rel_target
            if link_loc.is_symlink():
                if link_loc.resolve() == src.resolve():
                    # Present link is the same we want to create; don't do anything
                    continue
                else:
                    # Remove the deprecated link
                    link_loc.unlink()
                    logger.debug(f"Removed deprecated link '{link_loc} -> {link_loc.absolute()}'")

            if link_loc.exists():
                assert not link_loc.is_symlink()
                # Rename the path so as to not lose data
                renamed_path = _rename_to_free_file(move['new path'])
                logger.debug(f"Previous file '{link_loc}' was renamed to '{renamed_path}'.")
            else:
                # Make sure the directory hierarchy exists
                link_loc.parent.mkdir(exist_ok=True)

            link_loc.symlink_to(rel_target)
            logger.debug(f"Added link '{link_loc}' -> {rel_target}")
            num_created_links += 1

        logger.info(f"Created {num_created_links} new links in {inroot}.")

    # Shorthand
    rebuild_links = rebuild_input_datastore


    @property
    def latest(self):
        """
        Return the record with the latest timestamp.
        """
        # TODO: Django pre-filter; see sumatra.recordstore.django_store.__init__.py:most_recent
        latest = None
        for rec in self:
            if latest is None:
                latest = rec
            elif latest.timestamp < rec.timestamp:
                latest = rec
        return latest
    @property
    def earliest(self):
        """
        Return the record with the earliest timestamp.
        """
        earliest = None
        for rec in self:
            if earliest is None:
                earliest = rec
            elif earliest.timestamp >= rec.timestamp:
                # > would also work, but >= makes this a better opposite
                # operation to `earliest`
                earliest = rec
        return earliest

    ## Recordstore-modifying interface ##
    # TODO: Add this functionality to RecordView ?
    # TODO: Calling .save() on each record is expensive (see sumatra.recordstore.django_store.__init__)
    #       Couldn't we skip most of the that work, since the the only thing which changes is the tags ?

    def add_tag(self, tag):
        """
        Add a tag to all records in a record store view.

        .. Note:: At the risk of stating the obvious, this function will modify
           the underlying record store.
        """
        for record_view in self:
            record = self.record_store.get(self.project.name, record_view.label)
            record.add_tag(tag)
            self.record_store.save(self.project.name, record)

    def remove_tag(self, tag):
        """
        Remove the tag from all records in the record store view.

        .. Note:: At the risk of stating the obvious, this function will modify
           the underlying record store.
        """
        for record_view in self:
            record = self.record_store.get(self.project.name, record_view.label)
            record.tags = set(t for t in record.tags if t != tag)
            self.record_store.save(self.project.name, record)


    ## Read-only interface to RecordStore ##

    @property
    def project(self):
        return self._project
    @property
    def record_store(self):
        return self._project.record_store

    # def list_projects(self):
    #     return self.recordstore.list_projects()
    def get(self, label):
        if isinstance(label, (list, tuple)):
            return [self.get(label_) for label_ in label]
        else:
            return RecordView(self.record_store.get(self.project.name, label))
    @property
    def list(self):
        """Ensure the contents of the iterable are cached as a list.

        .. Note: the result is cached. Since each filter creates a new view,
           this means that each filter may potentially keep its own cache.

        :returns: self
        """
        if not isinstance(self._iterable, list):
            self._iterable = list(self)
        return self
    def aslist(self):
        """Return records as a list. Triggers caching of the result."""
        self.list;
        assert isinstance(self._iterable, list)
        return self._iterable
    def labels(self):
        """Return the list of labels.

        Note: like `list`, the result is cached.
        """
        if self._labels is None:
            if self._iterable is None:
                self._labels = self.record_store.labels(self.project.name)
            else:
                self._labels = [record.label for record in self]
        return self._labels
    def most_recent(self):
        "Return the label of the most recent record."
        return self.latest.label
    def export_records(self, records, indent=2):
        return self.record_store.export_records(records, indent=indent)
    def export(self, indent=2):
        records = self.aslist()
        return self.export_records(records, indent=indent)

    ## Representation / Visualization functions ##

    @property
    def summary(self) -> RecordStoreSummary:
        """
        Return a RecordStoreSummary.
        NOTE: This becoming obsolete by a combination of `dframe` and `summaries`.
        """
        return RecordStoreSummary(self.list)
    def dframe(self,
               include: Sequence[str]=(
                'timestamp', 'duration', 'reason', 'outcome', 'main_file',
                'script_arguments', 'parameters', 'tags', 'command_line',
                'version', 'executable'),
               exclude: Sequence[str]=(),
               field_types: Optional[Dict[str,Callable]]=None) -> pd.DataFrame:
        """
        Convert to a Pandas DataFrame. Record attributes are mapped to columns.
        
        Parameters
        ----------
        include: Determines both which record fields to include, and in which
           order.
        exclude: Has precedence over `include`.
        field_types: Collection of initializers for the column fields.
            Typically a plain type (like `str` or `int`), but can also be a
            function. This is applied to the column value before constructing
            the DataFrame.
        """
        if self._iterable is None:
            raise RuntimeError(
                "To ensure the iterator is not consumable, call `list` "
                "before constructing a DataFrame.")

        # Placeholder function for nicer display of field names;
        # applied after iterating through records
        def format_field_name(field):
            return field

        fields = tuple(field for field in include if field not in exclude)
        field_initializers = self.field_initializers
        if field_types:
            field_initializers = {**field_initializers, **field_types}

        data = []
        labels = []
        for record in self:
            labels.append(record.label)
            entry = []
            for field in fields:
                value = getattr(record, field)
                if field in field_initializers:
                    value = field_initializers[field](value)
                entry.append(value)
            data.append(entry)
        data = np.array(data[::-1], dtype=object)  # Make sort_index easier with [::-1]

        index = pd.Index(labels, name='label')
        fieldnames = tuple(format_field_name(field) for field in fields)

        if len(data) == 0:
            data = data.reshape((0, len(fieldnames)))
        df = pd.DataFrame(data, index=index,
                          columns=fieldnames
                          ).sort_index(ascending=False)
        return df

    # NOTE: The RSView subclass in IndEEG.viewing.record_store_viewer
    #       adds support for automatically separating visualization histograms
    #       by split (assuming `splitby` was previously called).
    #       If this is functionality we want to add, we should see if
    #       it makes sense to port those functions here.
    def _repr_mimebundle_(self, include=None, exclude=None):
        if not isinstance(self._iterable, Sequence):
            # Can't compute stats without risking to consume the iterable
            # -> Fall back to repr
            return None
        elif not hv_is_ready():
            # HoloViews is either not installed or no backend configured.
            # -> Fall back to repr
            return None
        else:
            return hv.Layout([self.counts_table()]
                             + [self.summary_hist(feature)
                                for feature in self.summary_fields]) \
                   .cols(1) \
                   ._repr_mimebundle_(include, exclude)

    @property
    def summaries(self):
        if getattr(self, '_summaries', None) is None:
            self._summaries = self.compute_summaries()
        return self._summaries

    def compute_summaries(self):
        df = self.dframe(include=self.summary_fields)
        hists = {}
        for field in self.summary_fields:
            values = df[field]
            if np.issubdtype(getattr(values, 'dtype', None), 'timedelta64'):
                # Time deltas need to be converted to floats before passing to `histogram`
                # Conversion requires choosing a time unit – we do so based on the maximum value
                max_val = values.max()
                if max_val < pd.Timedelta(5, 'm'):
                    unit = 's'
                    unitstr = 's'
                elif max_val < pd.Timedelta(3, 'h'):
                    unit = 'm'
                    unitstr = 'min'
                elif max_val < pd.Timedelta(3, 'D'):
                    unit = 'h'
                    unitstr = 'hours'
                else:
                    unit = 'D'
                    unitstr = 'days'
                values = values.astype(f"timedelta64[{unit}]")
                dim = hv.Dimension(field, unit=unitstr)
            else:
                dim = hv.Dimension(field)
            hist = hv.operation.histogram(hv.Table(values, kdims=[dim]),
                                          bins='auto')
            hist = hist.relabel(group=field, label='all records')
            hists[field] = hist
        return hv.HoloMap(
            hists, kdims=[hv.Dimension('rec_stat', label='record statistic')]) \
            .opts(framewise=True)

    def counts_table(self, max_rows=10) -> hv.Table:
        if not self._iterable:
            raise RuntimeError(
                "To ensure the iterator is not consumable, call `list` "
                "before constructing a DataFrame.")
        table = hv.Table({'RSView': ['all records'],
                          'No. records': [len(self)]},
                         kdims=['RSView'], vdims=['No. records'])
        # row height is 25; one header + one data row
        return table.opts(height=2*25)

    def summary_hist(self, stat_field: str) -> hv.Overlay:
        """
        `stat_field`: One of the fields listed in `self.summary_fields`.
        """
        # Ensure that `stat_field` matches one of the values
        if stat_field not in self.summary_fields:
            raise ValueError(f"`stat_field` must be one of {self.summary_fields}. "
                             f"Received {repr(stat_field)}.")
        hists = self.summaries.select(rec_stat=stat_field)
            # .select returns a HoloMap, if len(hists) == 0 or len(hists) ≥ 2
            # If a HoloMap, it has the same kdims
            # If a Histogram (len(hists)==1), the selected kdim is dropped
        if isinstance(hists, hv.Histogram):
            # Inflate into HoloMap, for consistency
            hists = hv.HoloMap({stat_field: hists}, kdims=[self.summaries.get_dimension('rec_stat')])
            # hists = [hists]
        elif len(hists.kdims) > 1:
            # (.drop_dimension raises TypeError if it drops the last dimension)
            hists.drop_dimension('rec_stat')
        return hists.overlay().opts(
            title=stat_field, height=250, responsive=True, legend_position='right')

# TODO: For unmerged summaries, don't display # of records, and use 'duration'
#       instead of 'avg duration' as a column heading.
class RecordStoreSummary(dict):
    """
    **Class attributes**

    - `re_merge_labels`: Records are merged if the first match group this regex
      results in the same string.
      Default matches the standard timestamp label format with alphanumerical
      suffix: YYYYMMDD-HHMMSS_[aa]

      Timestamps must match but the suffix may differ, as long as it is composed
      only of alphanumerical characters. So records with the following labels
      would be merged:

      - 20200604-123060
      - 20200604-123060_1
      - 20200604-123060_10
      - 20200604-123060_a4ef

      But the following wouldn't

      - 20200604-123060_a4ef_
      - 20200604-123060_a4-ef
      - 20200604-123160
      - 20200604-123260

      Default value: ``r'^(\d{8,8}-\d{6,6})_([a-zA-Z0-9]+)?$'``

    """
    re_merge_labels = r'^(\d{8,8}-\d{6,6})_([a-zA-Z0-9]+)?$'
    def __init__(self, recordlist, base=None, merge=False):
        """
        Parameters
        ----------
        recordlist: Iterable[Record] | None
            Records to summarizes. Records whose labels differ by only a
            suffix are combined.
            Typically a RecordStoreView, but any iterable of records will do.
            Set to `None` creates an empty summary, unless `base` is not None.
        base: None | dict
            Initialize the summary with this dictionary. If `None`, summary
            is initialized as an empty dictionary to which the entries of
            `recordlist` are then added.
        merge: bool
            Whether to merge similar labels according to `re_merge_labels`.
            Default is ``False``.
        """
        if base is None: base = ()  # Empty initialization
        elif not isinstance(base, dict):
            raise ValueError("`base` argument to `RecordStoreSummary` most be "
                             "an dict or a derived class, like "
                             "RecordStoreSummary.")
        super().__init__(base)

        if recordlist is None:
            return  # Skip iteration over `recordlist`
        if not merge:
            for r in recordlist:
                assert r.label not in self
                self[r.label] = [r]
            return
        assert merge
        for r in recordlist:
            # For labels following the standard format, merge records whose
            # labels differ only by a suffix
            # Scripts for these records were started within the same second,
            # thus almost assuredly at the same time with a dispatch script
            m = re.match(self.re_merge_labels, r.label)
            if m is None or not m.groups():
                # Not a standard label format -- no merging
                assert(r.label not in self)
                self[r.label] = [r]
            else:
                # Standard label format
                shared_lbl = m[1]
                if shared_lbl in self:
                    self[shared_lbl].append(r)
                else:
                    self[shared_lbl] = [r]

    @property
    def merged(self):
        """Return a copy of the summary where similar labels are merged."""
        return RecordStoreSummary(mtb.utils.flatten(self.values()), merge=True)
    @property
    def unmerged(self):
        """Return a copy of the summary where similar labels are unmerged."""
        return RecordStoreSummary(mtb.utils.flatten(self.values()), merge=False)

    def __call__(self, *args, **kwargs):
        """
        Call `.dataframe()` with the given arguments
        """
        return self.dataframe(*args, **kwargs)

    def __str__(self):
        return str(self.dataframe())
    def __repr__(self):
        """Used to display the variable in text-based interpreters."""
        return repr(self.dataframe())
    def _repr_html_(self):
        """Used by Jupyter Notebook to display a nicely formatted table."""
        df = self.dataframe()
        # Instead of letting `_repr_html` truncate long lines, add hard line
        # breaks to the data (`self.dataframe()` returns a copy)
        colwidth = pd.get_option('display.max_colwidth')
        df.transform(self._add_newlines, colwidth=colwidth)
        pd.set_option('display.max_colwidth', None)
            # Deactivate line truncation for call to `_repr_html_`
        df_html = df._repr_html_()
        # Print newlines in text fields correctly
        df_html = df_html.replace("\\n", "<br>")
        pd.set_option('display.max_colwidth', colwidth)
            # Return `max_colwidth` to previous value
        return df_html

    def head(self, nrows):
        headkeys = itertools.islice(self.keys(), nrows)
        headrecs = {key: self[key] for key in headkeys}
        return RecordStoreSummary(None, headrecs)
    def tail(self, nrows):
        nkeys = len(self.keys())
        tailkeys = itertools.islice(self.keys(), nkeys-nrows, None)
        tailrecs = {key: self[key] for key in tailkeys}
        return RecordStoreSummary(None, tailrecs)

    @staticmethod
    def _truncate_value(attr, value, max_chars, max_lines):
        if attr == 'tags':
            # There's ordering of more relevant information with tags, and they
            # are usually short anyway.
            return value
        if isinstance(value, tuple):
            return tuple(RecordStoreSummary._truncate_value(attr, v, max_chars, max_lines)
                         for v in value)
        value = '\n'.join(value.splitlines()[:max_lines])
        if len(value) > max_chars:
            if attr == 'main_file':
                value = '…'+value[-max_chars+1:]
            else:
                value = value[:max_chars-1]+'…'
        return value

    def dataframe(self, fields=('reason', 'outcome', 'tags', 'main_file', 'duration'),
                  parameters=(), hash_parameter_sets=True,
                  max_chars=50, max_lines=1):
        """
        Parameters
        ----------
        ...
        max_chars: int
            Truncate record value after this many characters.
            Special case:
                'main_file' is truncated from the end instead of the beginning.
                'tags' are never truncated
        max_lines: int
            Keep only this number of lines, even if more lines would fit
            within the character limit.
        """
        def combine(recs, attr):
            # Combine the values from different records in a single entry
            def get(rec, attr):
                # Retrieve possibly nested attributes
                if '.' in attr:
                    attr, nested = attr.split('.', 1)
                    return get(getattr(rec, attr), nested)
                else:
                    value = getattr(rec, attr)
                    if hash_parameter_sets and isinstance(value, ParameterSet):
                        # Get a hash fingerprint (first 7 hash chars) of the file
                        h = mtb.parameters.get_filename(value)
                        value = '#' + h[:7]
                    return self._truncate_value(
                        attr, value, max_chars, max_lines)
            if attr == 'duration':
                durations = (r.duration if r.duration is not None else 0
                             for r in recs)
                s = sum(durations) / len(recs)
                h, s = s // 3600, s % 3600
                m, s = s // 60, s % 60
                return "{:01}h {:02}m {:02}s".format(int(h),int(m),int(s))
            else:
                vals = []
                for r in recs:
                    try:
                        vals.append(get(r, attr))
                    except (AttributeError, KeyError):
                        # Add string indicating this rec does not have attr
                        vals.append("undefined")
                vals = set(mtb.utils.flatten(vals))
                # Choose the most appropriate join character
                if any('\\n' in v for v in vals):
                    join_str = '\\n'
                elif sum(len(v) for v in vals) > pd.get_option('display.max_colwidth'):
                    join_str = '\\n'
                elif not any(',' in v for v in vals):
                    join_str = ', '
                elif not any(';' in v for v in vals):
                    join_str = '; '
                else:
                    join_str = ' | '
                # Join all parameters from the merged records into a single string
                return join_str.join(str(a) for a in vals)
        def format_field(field):
            # Take a field key and output the formatted string to display
            # in the dataframe header
            if field == 'duration':
                field = 'avg. duration'
            field = field.replace('.', '\n.')
            return field

        data = []
        # Append parameters to the list of fields
        # Each needs to be prepended with the record attribute 'parameters'
        if isinstance(parameters, str):
            parameters = (parameters,)
        fields += tuple('parameters.' + p for p in parameters)
        for lbl, sr in self.items():
            entry = tuple(combine(sr, field) for field in fields)
            entry = (len(sr),) + entry
            data.append(entry)
        data = np.array(data)

        fieldnames = tuple(format_field(field) for field in fields)
            # Add line breaks to make parameters easier to read, and take less horizontal space
        if len(data) == 0:
            data = data.reshape((0, len(fieldnames)+1))
        return pd.DataFrame(data, index=self.keys(),
                            columns=('# records',) + fieldnames).sort_index(ascending=False)

    def array(self, fields=('reason', 'tags', 'main_file', 'duration'),
                  parameters=()):
        """
        Return the summary as a NumPy array.
        NOTE: Not implemented yet
        """
        raise NotImplementedError

    @staticmethod
    def _add_newlines(val, colwidth):
        # TODO: Preferentially break on whitespace, '_'
        if isinstance(val, pd.Series):
            return val.transform(RecordStoreSummary._add_newlines,
                                 colwidth=colwidth)
        s = str(val)
        l = colwidth
        nlines = int(np.ceil(len(s) / l))
        return '\n'.join([s[i*l:(i+1)*l] for i in range(nlines)])
