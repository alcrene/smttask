# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python (smttask)
#     language: python
#     name: smttask
# ---

# %% [markdown]
# # Basic test for the visualization functions of RecordStoreView
#
# Because these produce figures, they are currently implemented in a Jupyter notebook. They must be run manually and the output visually inspected.

# %%
from smttask.view import RecordStoreView
from tqdm import tqdm

# %%
import holoviews as hv
hv.extension('bokeh')

# %% [markdown]
# Create a new test project and `cd` into that directory.

# %%
import os
os.chdir('test_project')

# %% [markdown]
#     import conftest
#     conftest.pytest_sessionstart(None)

# %% [markdown]
# From this point on our working directory (including for imports) is *smttask/tests/test_project*.

# %%
import tasks

# %% [markdown]
# Run a number of tasks. Sumatra overhead is currently at 9 s / task, so we don't run too many.

# %%
task_list = [tasks.Orbit(start_n=0, n=n, x=1, y=1., reason='viz test')
             for n in [1000000,2000000,3000000,4000000,5000000]
            ]

# %%
for task in tqdm(task_list):
    task.run()

# %%
rsview = RecordStoreView()

# %% [markdown]
# Without calling `.list`, the iterable is consumable, so only a basic representation of the record store is possible

# %%
rsview

# %% [markdown]
# After converting the iterable to a list, we get a rich representation summarizing record times and durations

# %%
rsview.list
