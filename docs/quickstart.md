% This file as just a copy of the Readme, without the `Motivation` section.
% It should be kept in sync with README.md

Quick start
===========

Installation
------------

*SumatraTask* requires Python >= 3.7.

At present it also requires some development packages (*mackelab_toolbox*, *parameters* and *sumatra*). These dependencies are taken care of by first installing the package requirements:

    pip install -r requirements.txt
    pip install .

Configuration
-------------
After installing `smttask` in your virtual environment, change to your project directory and run

    smtttask project init

Follow the prompts to configure a new Sumatra project with output and input directories. The current implementation of `smttask` requires that the output directory by a subdirectory of the input directory.[^init-wraps-smt]

**Hint** It's a good idea to keep the VC repository tracked by Sumatra as lean as possible. Things like project reports and documentation are best kept in a different repository.

[^init-wraps-smt]: The `smttask project init` command is a wrapper around Sumatra's `smt init`, with helpful defaults and type checking. However, you may also initialize a project with Sumatra directly:

        cd /path/to/project/deathstar
        smt init --datapath data/run_dump --input data deathstar

*SumatraTask* workflows
-----------------------

Workflows constructed with *SumatraTask* have a number of benefits over the common “all-in-one” scripts:[^1]

- Lazy execution of expensive computations;
- Automatic on-disk caching of expensive computations;
- Optional, in-memory caching of intermediate computations;
- Fully reproducible workflows: every required parameter, and every package version, is recorded;
- Composability: *Tasks* can be used as inputs to other *Tasks*;
- Portability: Any *Task* can be serialized to a JSON file, and then executed from that file. This is great for running batches of jobs, either on a local or a remote machine.

All this with minimal markup. How minimal ? Suppose you have a analysis function called `analyze`, taking a NumPy array and some parameters `dt` and `nbins`, and returning three values:

```python
def analyze(arr, dt, nbins):
  ...
  return (μ, σ, p)
```

To turn this into a *Task*, you would do the following:

```python
@RecordedTask
def analyze(arr: Array, dt: float, nbins: int) -> Tuple[float,float,float]:
  ...
  return (μ, σ, p)
```

and add the following imports to the top of your file:

```python
from typing import Tuple
from smttask import RecordedTask
from scityping.numpy import Array
```

That's it ! This is still 100% valid Python, so you can run it directly within your notebook or editor. All it requires is two things:

- That each task be a [*pure function*](https://en.wikipedia.org/wiki/Pure_function).
- That all the inputs be *serializable* to JSON.

Note that there is no way *SumatraTask* can check that a function is pure, so it relies on you to do so. Be especially careful with functions that depend on objects which conserve state via private attributes, for example random number generators.

The requirement for *serializability* means that we need to provide for each data type a pair of functions to serialize and deserialize values to and from JSON. Under the hood, *SumatraTask* uses [*Pydantic*](https://pydantic-docs.helpmanual.io) for serialization, so most built-in types are already supported. Additional types geared for scientific computing (such as NumPy arrays and dtypes) are also defined in [*scityping*](scityping.readthedocs.io/).

Ensuring all our input data are serializable is not always trivial, but it is the only thing required to unlock all the benefits mentioned [above](#sumatratask-workflows).


[^1]: There are a few mature workflow management libraries for Python that provide notable improvement over the all-in-one scripts, for example [Snakemake](https://snakemake.readthedocs.io/en/stable/), [Luigi](https://luigi.readthedocs.io/en/stable/), [Nextflow](https://www.nextflow.io/) and [DoIt](https://pydoit.org/). Philosophically, *SumatraTask* is probably closest to *Luigi*, in that it is 100% Python and does not aspire to be a task scheduler; *SumatraTask* workflows can easily be wrapped as *Snakemake* rules to make use of its scheduler, for example. Moreover, *SumatraTask* has its own compelling features: the minimal markup, the integration with the [Sumatra](https://pythonhosted.org/Sumatra/) electronic lab book, both upstream and downstream composability (*Snakemake* and *Luigi* only support downstream composition well), and in-memory tasks.

Running tasks
-------------

- As part of a script.

  One could define, for example, the following file named *run.py*:

  ```python
  import numpy as np
  from project.tasks import analyze

  tasks = []
  for dt in [0.1, 0.3, 0.5]:
    tasks.append(analyze(arr=np.array([1, 2, 3]), dt=0.5, nbins=2))

  for task in tasks:
    task.run()
  ```

  Typically such a *run.py* file would be excluded from version control.
  Especially convenient is using a Jupyter notebook for such a run file, to allow easy in-line documentation.

- From a task description file.

  In the example example, we could change
  
  ```python
    task.run()
  ```
  
  to
  
  ```python
    task.save("taskdir")
  ```
  
  Now, instead of executing the task, the script generates a complete, self-contained *task description file* (basically a JSON file) and places it within the directory *taskdir* with a unique, automatically generated file name.[^2] Task description files can be executed from the command line:
  
  ```bash
  smttask run taskdir/task_name
  ```
  
  This approach is especially convenient for generating task file locally, and running them on a more powerful computation cluster. Although *SumatraTask* [is not a scheduler](#features), the `smttask run` command does provide basic multiprocessing and queueing capabilities. For example, the following would run all task files under *taskdir*, four at a time:
  
  ```bash
  smttask run -n4 taskdir/*
  ```

[^2]: It is also possible to specify a filename to `task.save()`.

Usage recommendations
---------------------

   - Keep extra project files (such as notes, pdfs or analysis notebooks – anything that does not serve to reproduce a run) in a different repository. Every time you run a task, Sumatra requires you to commit any uncommitted changes, which will quickly become a burden if your repository includes non-code files. Jupyter notebooks are *especially* problematic, because every time they are opened, the file metadata is changed. (Strongly recommended in this case is to pair the notebook to a Python script with [Jupytext](https://jupytext.readthedocs.io), and only add the script to version control.)
   
   This comment about separating the code repository is even more important if you use the 'store-diff' option. Otherwise you will end up with very big diffs, and each recorded task may occupy many megabytes.
   - It will happen that you run a task only to realize that you forgot to make a small change in your last commit. It's tempting then to use `git commit --amend`, so as to not create a new unnecessary commit – *do not do this*. This will change the commit hash, and any Sumatra records pointing to the old one will be invalidated. And no matter how careful you are to "only do this when there are no records pointing to the old commit", it *will* happen, and you *will* hate yourself.

(differences-with-sumatra)=
Recording changes compared to Sumatra
-------------------------------------

  - *SumatraTask* sets the “main file” to the module where the Task is defined. This may not be the file passed on the command line.
  - The file passed on the command line is logged as “script arguments”.

### Limitations

  - `stdout` and `stderr` are currently not tracked.


Features
--------

_SumatraTask_ **will**
  - Manage saving and loading paths, so you can concentrate on what your code
    should do rather than where it should save its results.
    All tasks are saved to a unique location, no previous location is ever
    overwritten, and results are load paths are resolved transparently as
    needed.
  - Automatically load previous computation results from disk when available.
  - Record code version and parameters in a Sumatra project database.
  - Allow you to insert breakpoints anywhere in your code.

_SumatraTask_ **will not**
  - Schedule tasks: tasks are executed sequentially, using plain Python
    recursion to resolve the dependency tree. To automatically set up sets of
    tasks to run in parallel with resource management, use a proper scheduling
    package such as [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html), [Luigi](https://luigi.readthedocs.io/en/stable/), [NextFlow](https://www.nextflow.io/) or [DoIt](https://pydoit.org/).
    _smttask_ provides a helper function to generate _snakemake_ workflows;
    a similar bridge to other managers should also be possible.


### Compared to Luigi/Snakemake

The result of tasks can be kept in memory instead of, or in addition to, writing to disk.

  ~ This allows for further separation of workflows into many small tasks. A good example where this is useful is a task creating an iterator which returns data samples. This is a typical way of feeding data to deep learning libraries, but since an iterator cannot be reliably reloaded from a disk file, such a task does not fit well within a Luigi/Snakemake workflow.

Entire workflows can be executed within the main Python session.

  ~ This is especially useful during development: the alternative, which is to spawn new processes for each task (perhaps not even Python processes), can make it easy to lose information from the stack trace, or prevent the usage of `breakpoint()`.
  
Allows for different parent task

  ~ Luigi/Snakemake make it easy to use the same task as parent for multiple child tasks, but using different parents for the same child is cumbersome and leads to repeated code. (I think ?)

Manages output/input file paths.

  ~ Luigi/Snakemake require you to write task-specific code to determine the output and input file paths; Luigi's file path resolution in particular is somewhat cumbersome. With *smttask*, file paths are automatically determined from the task name and parameters, and you never need to see them.


### Compared to Sumatra

Both input and output filenames can be derived from parameters
  ~ (Sumatra requires inputs to be specified on the command line)
