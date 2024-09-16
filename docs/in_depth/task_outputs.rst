******************************
Managing recorded task outputs
******************************

Comparison with a makefile solution (i.e. why we use hashes)
============================================================

One of the distinguishing features of *smttask* is that it takes care of keeping track of your computation results; this contrasts with a Makefile based system like Snakemake, where the filename is what determines whether a computation occurred. There are pros and cons to both approaches:

With a makefile based system, output results would be stored in files with names like::

    TrainClassifier_subject=3_smoothing=butterworth-0.2-1.0_model=MLP-100-100-100.dat
    TrainClassifier_subject=3_smoothing=butterworth-0.2-1.0_model=MLP-200-200-200.dat
    TrainClassifier_subject=3_smoothing=butterworth-1.0-1.0_model=MLP-100-100-100.dat
    ...

When a new result named say ``TrainClassifier_subject=3_smoothing=butterworth-1.0-1.0_model=MLP-200-200-200.dat`` is requested, ``make`` does one of two things. If the file already exists, it stops immediately. If it doesn’t exist, ``make`` will match the name to a parameterized rule, then execute the rule using those parameters. This is fairly easy to understand as a user, and it makes it possible to find files by hand simply by browsing the folder.

Although simple for a user, this approach does have its limitations. For one, all parameters need to be specified on the command – passing large arrays as parameters, or a Python object, is nigh impossible. Also, even when all the parameters take the form of numbers and strings, the developer of the makefile rules still needs to anticipate which parameters will be needed by users – if there are dozens of different parameters, listing them all in the file name is at best impracticable, and may be impossible due to filename length limits. Real numbers are also inconvenient: specifying 3π/8 as a double precision float would require 16 digits. Finally, the syntax for writing parameterized rules can easily become arcane, especially for people who don’t regularly write makefiles. 

*smttask* addresses these issues by naming files based on a hash of all of their parameters. Since just about any parameter can be hashed (if not, one can define a serializer – see :ref:`serialization`), each set of parameters maps to a unique, well-defined filename of the form::

    TrainClassifier_abf612689ffed594beaaafbdd9923549.json

On the flip side, with filenames computed from hashes, it is no longer possible to find result files simply by browsing the output folder. In theory one could open each file and inspect the contents to find the output from a particular set of parameters, but *smttask* provides two much more convenient ways to retrieve results:

- :ref:`access_by_reconstruction`
- :ref:`querying_run_database`

These methods exploits *smttask*’s completely systematic way of assigning filenames to propose an interface which is arguably even easier to use than the filename-based system described above.

.. _access_by_reconstruction:

Accessing results by reconstructing tasks
=========================================

When *smttask* runs a recorded task, it always starts by checking the output directory to see if a result exists. If so, execution is skipped; the result is loaded from disk and returned, exactly as the the task had been executed. So often the easier approach is to ignore the task recording completely; for example, consider the following 

.. code-block:: python

   # Code which generates a list of parameters and stores them
   # in `paramlist`

   tasks = [MyTask(**params) for params in paramlist]

   for task in tasks:
     result = task.run()
     plt.plot(result.x, result.y)

On the first run, this block will need to run each task, which may take a lot of time. However subsequent executions will be fast, since each task is automatically retrieved from disk. This approach has the advantage of fully documenting how the results were obtained, and of making the figures completely reproducible, even if the intermediate results are lost.

Note also that it makes no difference in this approach whether the parameters


.. _querying_run_database:

Querying the run database
=========================

Sometimes a query based approach is what we want. There are three main situations where this happens in our own practice:

- We are interested in inspecting the runs themselves: to determine run times, compare parameters, to see what has / has not yet been executed, etc.
- We want to operate on the database of run records, e.g. delete outdated run records / result files.
- We want to plot results based on “whatever has already been executed”. For example, a sensitivity analysis might require hundreds of random permutations of a model, and we want to check after a few dozen whether it is worth continuing.

For this *smttask* provides the *RecordStoreView* object, accessible as ``smttask.view.RecordStoreView``.

TODO: Explain how to use ``RecordStoreView``.

.. _file_layout:

Internal file layout structure
==============================

The two methods discussed above cover all anticipated usage of *smttask*,  such that we do not expect that a user should need to browser the internal file structure manually. Nevertheless, there are always unexpected situations, and for those cases it may be useful to know how *smttask* organizes its output files.

TODO


For developers
==============

Which functions compute task hashing
--------------------------------------

To uniquely and reproducibly compute hashes of Task parameters, we use ``hashlib.sha1``. A SHA1 hash is also what git uses to ensure that all commits have unique names. SHA hashes are quite long however; since we anticipate that a data store would contain at most a few tens of thousands of runs of the same task, we shorten hashes to their first 10 characters. (Determined by the private variable ``TaskInput._digest_length``) We call these shortened strings 'digests', paralleling the use in `hashlib`.

Digests are computed on a task’s inputs, as soon as the task is created. This ensures that the same task always has the same digest, even when some of the inputs are mutable. (The executation of a task may modify its inputs in-place.) There is likely some way to exploit this to break out of *smttask* “one task: one result” guarantee, but *smttask* was written to save you from yourself, not from an adversarial user.

.. Note:: Basing a digest on the outputs would not make sense: we need to compute it before running the task in order to check if the task has already been run with these parameters.

.. Note:: We don’t use the ``hash()`` function, even though it is faster when available. That function is meant for a different purpose; in particular, ``hash()`` should only be defined for immutable types. If we add ``__hash__`` to mutable types to make them hashable, we are breaking some pretty fundamental conventions, since Python will now assume that they are mutable. Moreover, for security reasons ``hash()`` is intentionally not reproducible across Python sessions. One can set the PYTHONSEED so that results are reproducible, but a) users may not want to weaken there security, and b) it doesn’t change that stable hashes are not, and never will be, the goal of ``hash()``.

   That said, we do define a ``__hash__`` method on tasks, defined as ``hash(self.taskinputs.digest)``, since they should only be mutable in the sense of a lazy operation: a task has only two states (executed or not), and its identity does not depend on this state.

- `Task.compute_hashed_digest`: As a general rule, all task input are hased with this function.
- `Task.compute_unhashed_digest`: In some exceptions, certain parameters may be left unhashed and included as key:value pairs. This is used to supporte IterativeTask, which needs to know the value of the parameter being iterated over.
- `make_digest`: Fixes the convention for how we concatenate the results of hashed and unhashed digests

The UnpureMemoizedTask works different and uses a hash of its _outputs_ to form the digest. This depends on the methods
- `stablehexdigest`
- `TaskOutput.hashed_digest`
- `TaskOutput.unhashed_digest`
- `TaskOutput.digest`


Which functions interact with output files
------------------------------------------

Functions which depend on the directory layout of input/output data stores (found in ``smttask/base.py`` unless indicated otherwise):

- `TaskInput.load()`  : Reads result files from the input data store for *upstream tasks*
- `Task.get_output()` : User-facing function: returns a result associated with a particular name
- `Task._parse_result_file()`: Used in `.run()` when skipping past executing and retrieving past results
- `TaskOutput.write()`  : Writes result files to the output data store
- `TaskOutput.outputdir()`: The subdirectory, relative to datastore root, where we store outputs for this task. Usually the task name.
- `TaskOutput.outputpaths()`: Expected task result file locations, relative to data store root. These are computed purely based on task metadata, so the files may not exist. For the same reason, they also don’t include annex files.
- `task_types::RecordedTask.run()`: This is the function which checks for result files from previous runs, and if found, loads them instead of running the task.

Functions which depend on the file naming convention
- `TaskOutput._output_types()`

Other functions either take already determined paths as arguments, or defer to one of the functions above to determine an input or output path. E.g. `TaskDesck.load()` takes `obj` as an argument, which may be a path, then loaded the task data into `data`

In addition, the function

- `TaskDesc.save()`

will determine a save path for task descriptions. These are typically outside the datastore.
