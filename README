Motivation
==========

This packages extends the computation tracking capabilities of [Sumatra](https://pythonhosted.org/Sumatra/) with “Task” constructs (borrowed from [Luigi](https://luigi.readthedocs.io/)). This allows better composability of tasks, and enables a workflow based on runfiles. In particular, a runfile may be a full-blown Jupyter or RStudio notebook, allowing one to produce highly reproducible, easily documentable analyses.

Sumatra implements an electronic lab book, logging execution parameters and code versions every time you run a computational script. If you are unfamiliar with it, you can get started [here](https://pythonhosted.org/Sumatra/getting_started.html).

Note that in order to implement the runfile workflow, this packages changes somewhat the code files Sumatra tracks as explained [below](runfile-pattern). If you are already used to Sumatara, make sure you understand these changes before running this on anything important.

Installation
============

Install with

    pip install -r requirements.txt

This ensures that the `mackelab-toolbox` dependency is correctly retrieved from the github repository.
As this package is still in development, the `requirements.txt` is configured to produce an editable install.

Runfile pattern
===============

Consider the following computational workflow:

In file *run.py*

    from tasks import Task
    Task.run('params')

In file *tasks.py*

    import smttask
    class Task(smttask.Task):
        [define task]

If tasks are self-contained, it should not be required to track *run.py* in version control – we really only care about *tasks.py*. For this reason, when executing an `SmtTask`, _it is the module where the task is defined_ that is logged as “main file”, not the file passed on the command line. So in the example above, running

    python run.py

would result in a Sumatra record with `tasks.py` as its main file. Since the “script arguments” entry is no longer really meaningful, we use it to record the task name.

This approach allows us to launch tasks from a run file, which is a lot more convenient than launching them from the command line. The runfile may even be a Jupyter or RStudio notebook, enabling for rich documentation capabilities of your workflows.

**Caution**: If you use the run file approach, make sure tasks are truly composable, and that your run file does not contain anything that can affect the outcome of a task. Things like

run.py

    from tasks import Task
    Task.foo = 100000
    Task.run('params')

would be irreproducible, since Sumatra did not log the new value of `foo`.

Usage recommendations
=====================

   - Keep extra project files (such as notes, pdfs or analysis notebooks – anything that does not serve to reproduce a run) in a different repository. Every time you run a task, Sumatra requires you to commit any uncommitted changes, which will quickly become a burden if your repository includes non-code files. Jupyter notebooks are *especially* problematic, because every time they are opened, the file metadata is changed.
   This comment about separating the code repository is even more important if you use the 'store-diff' option. Otherwise you will end up with very big diffs, and each recorded task may occupy many megabytes.
   - It will happen that you run a task only to realize that you forgot to make a small change in your last commit. It's tempting then to use `git commit --amend`, so as to not create a new unnecessary commit – *do not do this*. This will change the commit hash, and any Sumatra records pointing to the old one will be invalidated. And no matter how careful you are to "only do this when there are no records pointing to the old commit", it *will* happen, and you *will* hate yourself.

Changes compared to Sumatra
===========================

  - As noted above, SumatraTask sets the “main file” to the module where the Task is defined. This may not be the file passed on the command line.
  - The file passed on the command line is logged as “script arguments”.

Limitations
-----------

  - `stdout` and `stderr` are currently not tracked.


Features
========

_Smttask_ will
  - Manage saving and loading paths, so you can concentrate on what your code
    should do rather than where it should save its results.
    All tasks are saved to a unique location, no previous location is ever
    overwritten, and results are load paths are resolved transparently as
    needed.
  - Automatically load previous computation results from disk when available.
  - Record code version and parameters in a Sumatra project database.

_Smttask_ **will not**
  - Schedule tasks: tasks are executed sequentially, using plain Python
    recursion to resolve the dependency tree. To automatically set up sets of
    tasks to run in parallel with resource management, use a proper scheduling
    package such as [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html)
    or [Luigi](https://luigi.readthedocs.io/en/stable/).
    _smttask_ provides a helper function to generate _snakemake_ workflows;
    a similar bridge to _luigi_ should also be possible.


Compared to Luigi/Snakemake
-----------------
  - The result of tasks can be kept in memory instead, or in addition, to writing to disk.
    This allows for further separation of workflows into many small tasks.
    In particular, a data loading task which standardizes input data is a common workflow step but would make little sense in Luigi/Snakemake as a task on its own.
  - Allow for different parent task
    Luigi/Snakemake make it easy to use the same task as parent for multiple child tasks, but using different parents for the same child is cumbersome and leads to repeated code. (I think ?)
  - Manages output/input file paths. Luigi/Snakemake require you to write task-
    specific code to determine the output and input file paths; Luigi's file
    path resolution in particular is somewhat cumbersome. With *smttask*, file
    paths are automatically determined from the task name and parameters, and
    you never need to see them.


Compared to Sumatra
-------------------
  - Both input and output filenames can be derived from parameters
    (Sumatra requires inputs to be specified on the command line)
