Configuration
-------------
After installing `smttask` in your virtual environment, change to your project directory and run

.. code:: bash

   smtttask init

Follow the prompts to configure a new Sumatra project with output and input directories. The current implementation of `smttask` requires that the output directory by a subdirectory of the input directory. The init script is a wrapper around Sumatra's init script, with helpful defaults and type checking. However, you may also initialize a project with Sumatra directly:

.. code:: bash

   cd /path/to/project/deathstar
   smt init --datapath data/run_dump --input data deathstar

Note that in the same way as Sumatra, the project directory must be within a version control (VC) repository, since Sumatra relies on VC to record the code version.

**Hint**: It's a good idea to keep the VC repository tracked by Sumatra as lean as possible. Things like project reports and documentation are best kept in a difference repository.

Specifying a task
-----------------
Tasks are most easily created by decorating a function:

.. code:: python

   from smttask import RecordedTask

   @RecordedTask
   def Add(a: float, b: float, n: int=10) -> float:
     for i in range(n):
       a += b
     return a,

A few remarks:

- Task functions must be **stateless**. That means that they should not be class methods (unless they are static) and should not have any side-effects, such as changing class or module variables. This is essential because a fundamental assumption of *smttask* is that the output of a task is entirely determined by its inputs. There is no way for *smttask* to check for statelessness, so you are responsible for ensuring this assumption is valid.

- All function arguments have type annotations. This is required by *smttask* to construct the associated Task. If an argument can take different types, use `~typing.Union` to specify that.

- The output type must also be indicated via function annotation. There is also a more verbose notation (detailed below) allowing to specify more outputs. The use of `~typing.Union` here is untested and not recommended.

- We capitalized the function name :func:`Add` here. This is because the decorator converts the function into a class (a subclass of `~smttask.Task`). This choice is of course purely stylistic.

There are currently four available Task decorators:

``@RecordedTask``
   Standard task which will be recorded by Sumatra.
``@RecordedIterativeTask``
   A recorded task with a special *iteration parameter.* This parameter can be used to reuse previous partial computations of the same task with fewer steps.
   Typical use cases are iterative fitting procedures or simulations.
``@MemoizedTask``
   Stantard task which is *not* recorded by Sumatra.
   Because the result is not written to disk, it does not need to be serializable and can be any Python object.
   Used as component of a larger pipeline.
``@UnpureMemoizedTask``
   A special task intended to simplify workflow definitions, by encapsulating tasks which depend on computer state.
   The typical case is a database query: we want to define the workflow with “list entries from DB” but the digest should be computed from the *result* of that query.
   This is especially useful if the state changes seldomly, since any change of state would cause all dependent tasks to have new digests.

Tasks as inputs
^^^^^^^^^^^^^^^
You can specify a Task type as an input to another:

.. code:: python

   class Mul(RecordedTask):
   def Mul(a: Add, b: float) -> float:
     return a*b

Note that it is not necessary for a task to explicitly state that its input(s) should be another task, and in fact *not* doing so greatly simplifies composability of tasks. By specifying only the required type (possibly as a `~typing.Tuple`, if the task returns multiple values), any task returning a result of appropriate type is accepted.

Multiple output values
^^^^^^^^^^^^^^^^^^^^^^
There are two ways to specify that a task should return multiple outputs. One is simply to specify it as a `~typing.Tuple`:

.. code:: python

   @RecordedTask
   def Add(a: float, b: float, n: int=10) -> Tuple[float, int]:
     ...

Such a task is treated as having a single output (a tuple). The output is saved to a single file, and you use indexing to retrieve a particular result.

Alternatively, one can explicitely construct the `~smttask.TaskOutput` type:

.. code:: python

   from smttask import TaskOutput

   class AddOutputs(TaskOutput):
     x: float
     n: int

   @RecordedTask
   def Add(a: float, b: float, n: int=10) -> AddOutputs:
     ...

With this approach, it is possible to assign names to the output values. Moreover, the values of ``x`` and ``n`` will be saved to separate files (differentiated by their names).

No matter the notation used, when used as an input to another Task, the receiving Task sees a tuple. It is currently not possible to index outputs by name.

Limitations
^^^^^^^^^^^
Output types must be supported by Pydantic, although with Pydantic's hooks for defining custom encoders and validators, this is almost always a solvable problem. You can check whether a type ``MyType`` is supported by executing the following snippet:

.. code:: python

   from pydantic import BaseModel
   class Foo(BaseModel):
     a: MyType

If this raises an error stating that no validator was found, you will need to define a custom data type, as detailed in the `Pydantic documentation <https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types>`_.

The one type I have found which is explicitely not supported is `Generator`. In that case a solution is to define a class with `__iter__()` and validation methods, and use that instead of the built-in `Generator` type.
