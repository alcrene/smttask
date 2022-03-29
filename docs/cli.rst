**********************
Command line interface
**********************

When installed, *Smttask* adds the following commands to the shell.

- ``smttask project init``  

  Run the initialization wizard for SumatraTask.
  
- ``smttask run``

  Run a task description file. This is the most common way of running a task, especially for batched runs.
  
  The command supports basic multiprocessing and queueing of multiple tasks.
  Also supports using a separate temporary record store; this is useful when running many parallel jobs on a compute cluster where the only available DB backend is SQLite, to avoid file lock conflicts.
  
- ``smttask store find_output``

  Find output files for a previously run task, given its task description file.
  
- ``smttask store rebuild``

  Rebuild the input datastore.
  
  This is useful if e.g. an update has caused all of the Task digests to
  change, in order for previous computations to be found by future Tasks.
  
- ``smttask store create_surrogates``

  Create surrogate records for outputs without records
  
  This allows routines which query the record store for outputs to work as
  expected, but of course statistics like run time for surrogate records are
  undefined.
  
- ``smttask store merge``

  Merge entries from the SOURCES record store(s) into the target record store.

  Intended usage is for combining run data that was recorded in separate
  record stores with the --record-store option of `smttask run`.
