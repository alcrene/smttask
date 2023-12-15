# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (smttask-docs)
#     language: python
#     name: smttask-docs
# ---

# %% [markdown]
# # Finding records
#
# Projects quickly accumulate thousands of recorded tasks, and finding particular results can be like finding a needle in a haystack. In theory all of the execution information is recorded in a Sumatra *data store* (in particular the execution time and duration, task and parameters, and the location of output files); the examples below show different methods to query that database and reload results from within a Jupyter notebook.
#
# > By design, only *RecordedTasks* are saved to the Sumatra database. *MemoizedTasks* are not.
#
# **TODO**: Create a demo project within the *smttask* repo
#
# **NOTE**: The visualization tools have improved since this document was written. Until we have proper auto-built API documentation, for the most up to date version, please peruse the source code of [smttask.view.recordstoreviewer](../smttask/view/recordstoreviewer.py).

# %%
import smttask
import smttask.utils
from tqdm.auto import tqdm
from mackelab_toolbox.meta import print_api
from smttask.param_utils import dfdiff, ParameterComparison

# %% [markdown]
# *smttask* provides the `RecordStoreView` class for interfacing with the record store. It can be called without arguments if the current directory is within the tracked project.

# %%
rsview = smttask.RecordStoreView().filter.tags("finished")

# %% [markdown]
# > Smttask follows the behaviour of Sumatra and automatically tags records during execution, to track their status. These status tags are:
# >   + **\_\_initialized\_\_**  — Task terminated before starting to run
# >   + **\_\_running\_\_**  – Task is still running
# >   + **\_\_crashed\_\_**   – Task terminated prematurely with an error
# >   + **\_\_killed\_\_**   – Task was killed
# >   + **\_\_finished\_\_** – Task completed successfully.
# >
# > An initial filter for **\_\_finished\_\_** tags is computationally very cheap, and avoids iterating over incomplete runs.
# > The filtering mechanism is explained [below](#Filtering).

# %% [markdown]
# `RecordStoreView` wraps an iterable over the records, which may or may not be consumable. Without any additional filtering, this iterable is over the entire record store.
# Iterating over records can be slow; calling `.list` on the record store view will make it cache the records internally as a list, much accelerating further iterations.

# %%
rsview.list;

# %% [markdown]
# ## Record list summary
#
# `RecordStoreView` provides the property `summary` (of type `RecordStoreViewSummary`), which in a notebook displays as a Pandas *Dataframe* summarising the records. This is a good way to get an initial overview of a data store, or to view the result after the list has been filtered (see [Filtering](#Filtering)).
#
# The output can be adjust with the following methods:
#
# - `merged` (property): Combine records with similar labels (by default, the same timestamp; type `RecordStoreViewSummary?` for instructions for how to change the merge pattern). The number of merged records is displayed in each row.
# - `unmerged` (property): Inverts the `merged` operation.
# - `head(nrows)`: Restrict the summary to the first `nrows`.
# - `tail(nrows)`: Restrict the summary to the last `nrows`.
# - `dataframe(...)`: Return the *Dataframe* used for display. Arguments are provided to adjust the content:
#   + *fields*: Which fields to include as columns in the dataframe. Default: *reason*, *outcome*, *tags*, *main_file* *duration*.
#   + *parameters*: Which parameters to include; specified as tuples of string; snested parameters can be specified with dots.
#   + *max_chars*: Truncate columns to this number of characters.
#   + *max_lines*: Keep only this number of lines from a field, even if more lines would fit within the character limit.

# %%
rsview.summary.merged.tail(15)

# %% [markdown]
# ## Basic record selection
#
# - `.get()`: Return the record(s) matching a specific label(s).
# - `.earliest`: Return the earliest record.
# - `.latest`: Return the latest record. A simple way to obtain an individual record.
# - `.list`: Make the RecordStoreView non-consuming (convert its iterable to a list). This is done in-place. Avoids querying the record store for subsequent iterations.
# - Standard “smart” indexing (i.e `rsview[key]`): Uses some heuristics to determine what to index:
#   + By label, if *key* is a str. Equivalent to `.get(key)`.
#   + The cached `.list`, if it is available and *key* is an int. Equivalent to `.list[key]`.
#   + The underlying iterable, otherwise. (No public equivalent, but can be achieved with `._iterable[key]`.) \
#   This is provided as a convenience during exploration.

# %%
rsview.latest

# %%
rsview.get('20201118-212254_c2c6ab')

# %%
rsview.get(['20201118-212254_c2c6ab', '20201119-095543_8b5fc1'])

# %%
rsview['20201118-212254_c2c6ab']

# %%
rsview[0]

# %% [markdown]
# ## Sumatra.RecordStore interface
#
# `RecordStoreView` also reproduces the part of the interface provided by Sumatra's `RecordStore` which makes senses for a read-only view.
#
# | *smttask.RecordStoreView* | *sumatra.recordstore.RecordStore* |  Description |
# |:---|:---|:---|
# | `.aslist()`    | `.list(...)`  | Return the records as a list. |
# | `.labels()`    | `.labels(...)` | Return the list of record labels (RecordStoreView caches the value). |
# | `.most_recent()` | `.most_recent(...)` | Return the *label* of the most recent record. Equivalent to `.latest.label`. |
# | `.export(indent=2)` | `.export(...)` |  Return a string with a JSON representation of the project record store. |
# | `.export_records(records, indent=2)` | `.export_records(...)` | Return a string with a JSON representation of the given records |
#

# %% [markdown]
# ### Exception to read-only interface
#
# `RecordStoreView` also adds `add_tag` and `remove_tag` methods, which modifying the underlying recorstore by respectively adding and removing tags to every record in the view. Combined with [filtering](#Filtering), this is an efficient way to mark particular records for later access, especially because tag filters are by far the [fastest](#Pre-vs-post-filters). For example, one can tag all records required for a particular figure:
# ```python
# rsview.filter.[date/version/parameter conditions].add_tag('figure1')
# ```
# Retrieving those records can then be done in milliseconds, even with a store containing thousands of records:
# ```python
# records = rsview.filter.tag('figure1').filter.[panel 1 condition]
# ```

# %% [markdown]
# ## Filtering
#
# The primary mechanism for pairing down the number of records is the *filter*. Filters can be chained, so for example to select all records between September 12th (inclusive) and 16th (exclusive) 2020:

# %%
records = (rsview.filter.after(20200912)
                 .filter.before(20200916)
          ).list
records.summary

# %% [markdown]
# > Filters return generators (to make chaining cheap), which is why we use the `.list` on the result to avoid the RecordStoreView being consumed the first time we use it.

# %% [markdown]
# ### Builtin filters
#
# The builtin filters are listed below; they are all accessed as attributes, as `filter.<filter name>`.

# %%
for fltr in rsview.filter.registered_filters.values():
    print_api(fltr)

# %% [markdown]
# #### Selecting based on parameters
#
# The filters `params` and `match` are specialized by *smttask* to recognize *task descriptions*: when specifying hierarchical parameters which include task descriptions, the *input* level may be omitted. For example, the following matches the key `'optimizer.model.params.μtilde'`, even though for some records the full key would be `'inputs.optimizer.inputs.model.inputs.params.μtilde'`. This is not only shorter, but at least in some cases will match records parameterized both with values and upstream tasks.
#
# (Internally, the `params` and `match` filter use the `get_task_param` function [detailed below](#Retrieving-parameter-values).)

# %%
rsview.filter.params(
    eq={'optimizer.model.params.μtilde': [-3.4729471730413772, -0.13678844546243388]}
).summary

# %%
# Full key of the first record includes three task descs:
rsview['20201119-090726_a8a2b0'].parameters[
    'inputs.optimizer.inputs.model.inputs.params.μtilde']

# %% [markdown]
# ### Pre vs post filters
#
# Most builtin filters, and all custom filters, are *post*-filters: an iterable over the record store is first created, then iterated over and the filter applied to each record. This is flexible but each query to the record store carries substantial overhead.
#
# Fore record stores built on top of database interfaces, specifically the Django record store, many filter can in theory be integrated into the SQL query which constructs the iterator. Since in this case the filter is applied before the iterator, we call it a *pre* filter, and it can be much faster. However, support for pre-filters must be provided on a per-record store and per-filter basis; at present, only a pre-filter for `tags` is provided, since that is the one also provided by Sumatra.
#
# Available pre-filters are applied automatically, *as long as they appear before any post-filter*. As an example, compare the execution time of the two following queries, which differ only in the order of their filters.

# %%
rsview2 = smttask.RecordStoreView()

# %%
# %time rsview2.filter.tags('killed').filter.label('202011').list;

# %%
# %time rsview2.filter.label('202011').filter.tags('killed').list;

# %% [markdown]
# > **tl;dr**: Some filters have *pre* versions. Apply those first.

# %% [markdown]
# ### Custom filters
#
# Rather than using one of the builtin filters listed above, one may instead pass an arbitrary function to the `.filter` attribute. This function should take one argument (the record) and return a bool; records for which it returns `True` are kept.
#
# For example, to kept only records whose duration was greater than 10 hours, one could do (duration is recorded in seconds):
#
# > `rsview.filter(...)` and `rsview.filter.generic_filter(...)` are semantically equivalent.

# %%
records = rsview.filter.output().filter(lambda rec: rec.duration > 10*60*60).list
records.summary

# %% [markdown]
# ## Comparing records
#
# ### Binary comparisons
#
# Let's say you have two records, *record1* and *record2*, which were produced using almost identical parameterisations but gave different results.

# %%
record1 = rsview.get('20201119-075320_7b84e7')
record2 = rsview.get('20201119-042150_e177c4')

# %% [markdown]
# If you don't remember exactly how each was run (or misremember), how do you determine what might explain the difference between the results ? Simply comparing the two parameter sets by eye is virtually impossible if there are more than a handful of parameters.

# %%
params1 = record1.parameters
params2 = record2.parameters

# %%
# Commented out for brevity
#print(params1.pretty())

# %% [markdown]
# *smttask* provides the function `dfdiff` for comparing two parameter sets. It works with hierarchical parameter sets, and keeps only those entries which differ. The result is returned as a Pandas Dataframe, so it displays nicely and can be further indexed.

# %%
dfdiff(params1, params2)

# %%
dfdiff(params1, params2).sort_index().loc[('inputs','optimizer','model')]

# %% [markdown]
# ### Comparing multiple records
#
# *smttask* also provides the `ParameterComparison` object, which is not limited to binary comparisons and works directly on either records or parameter sets. Below we compare all the records executed on the 10th of November 2020:
#
# > `ParameterComparison` is essentially doing an outer product of all key/value pairs in all parameter sets, and storing any difference. In the worst case, the memory requirements can therefore be exponential in the number of records compared.

# %%
cmp = ParameterComparison(rsview.filter.on(20201110))

# %% [markdown]
# To display the results of the comparison, use the `.dataframe()` method. By default, differences in hierarchical parameters are folded (indicated by `<+>`) to keep things legible even with large hierarchichal parameter sets.

# %%
cmp.dataframe()

# %% [markdown]
# We use the `depth` keyword argument to drill down into the the comparison.
#
# Often when doing this we want to hide certain columns; for example, since the *reason* column is free-form text, it may not useful in determining what caused two computations to differ. Since this is a standard *Dataframe*, we can hide columns with `.drop(columns=...)`.

# %%
cmp.dataframe(depth=2).drop(columns=['reason'])

# %% [markdown]
# > **Hint**: Don't underestimate the value of recording useful information in a task's “reason” attribute. Below are the recorded “reasons” for the first 4 entries in the same record list:

# %%
for rec, _ in zip(records, range(4)):
    print(f"----- {rec.label} -------")
    print("\n".join(rec.reason))
    print("")

# %% [markdown]
# ### Limitation
#
# Parameters for a task *T* can be specified either as values or as other tasks. In the latter case, the serialization of the task creates a *task description* with the keys *taskname*, *module* and *inputs*. The *inputs* entry can itself contain the serialization of other tasks – thus is the entire specification for *T* saved and recoverable from the record store.
#
# However, this does make it more difficult to compare records if in some cases parameters are specified as values, and in others as task descriptions – since those can never be equal.
#
# The function `fold_task_inputs()` (found in *smttask.utils*) can help working with these nested parameter sets created by chained tasks: it replaces task descriptions by the contents of their *inputs*. (In many cases the *taskname* is the same for all records.) This doesn't solve the problem of values differing from task descriptions, but at least the latter don't create such deeply nested hierarchies.

# %%
records = rsview.filter.on(20201110).list
cmp = ParameterComparison(params=[smttask.utils.fold_task_inputs(rec.parameters)
                                  for rec in records],
                          labels=[rec.label for rec in records])

# %%
cmp.dataframe(depth=1)

# %% [markdown]
# ## Retrieving parameter values
#
# Serialization establishes an equivalence between `Tasks`, *task descriptions* and `ParameterSet`s, but each has their own syntax to retrieve particular parameters. This is especially cumbersome with nested structures, where these types can arbitrarily mix. The function `get_task_param` (again from *smttask.utils*) provides a unique syntax that works with every object, and supports nested selection.

# %%
smttask.utils.get_task_param(records.latest, 'optimizer.model.params.μtilde')    # Record

# %%
smttask.utils.get_task_param(records.latest.parameters,                          # ParameterSet
                             'optimizer.model.params.μtilde')

# %%
smttask.utils.get_task_param(smttask.Task.from_desc(records.latest.parameters),  # Task
                             'optimizer.model.params.μtilde').get_value()
