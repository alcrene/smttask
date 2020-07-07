Configuration
-------------
After installing `smttask` to your virtual environment, run

.. code:: bash

   smt init --datapath [path/to/output/dir] --input [path/to/input/dir] [projectname]

The current implementation of `smttask` requires that the output directory
by a subdirectory of the input directory. The typical configuration is

.. code:: bash

   cd /path/to/project/deathstar
   smt init --datapath data/run_dump --input data deathstar

Specifying a task
-----------------
Subclass `Task`.
Specify input types at the class level with `inputs` dictionary.
Specify output types at the class level with `outputs` dictionary.

.. code:: python

   from smttask import RecordedTask

   class Add(RecordedTask):
       inputs = {'a': float, 'b': float, 'n': int}
       outputs = {'o': float}

       def _run(a, b, n=10):
           for i in range(n):
               a += b
           return a,

**Important** Tasks must always return a *tuple*.
There are facilities to avoid having to dereference the tuple all the time
in downstream tasks.
(See `Automatic unpacking of return tuple <#automatic-unpacking-of-return-tuple>`_)

Multiple input types
^^^^^^^^^^^^^^^^^^^^
If an argument can take multiple types, specify it as a tuple

.. code:: python

   class Add(RecordedTask):
       inputs = {'a': float, 'b': (int, float), 'n': int}
       ...

Nested inputs
^^^^^^^^^^^^^
The special `InputTuple` type is provided to specify input tuples.
For example, say we want our task to compute the integer power `n` of some
number `x`, and that `(x,n)` should be provided as a tuple. We can specify
this as

.. code:: python

   class Pow(RecordedTask):
       inputs = {'nx': InputTuple(float, (int, float))}
       ...

Inputs will then be properly casted to a `(float, int)` or a `(float, float)`
tuple, and only size-2 inputs will be accepted for the parameter `nx`.
Note that we can also specify type alternatives within the InputTuple.

Tasks as inputs
^^^^^^^^^^^^^^^
You can specify a Task type as an input to another:

.. code:: python

   class Mul(RecordedTask):
       inputs = {'a': Add, 'b': float}
       outputs = …
       def _run(a, b):
           return a*b,

(Trailing comma to return a tuple.)
Note it's not necessary for a task to explicitly state that its input(s) should
be another task, and in fact not doing so greatly simplifies composability of
tasks. By specifying only the required type (possibly as an InputTuple, if
the task returns multiple values), any task returning a result of appropriate
type is accepted.

**Warning**: It is not recommended to specify both Tasks and plain types as
input types. Multiple Tasks are OK.

Automatic unpacking of return tuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If input is specified as plain type, and a Task is used to compute it, the
(tuple) result of that task *is automatically indexed*. This allows one to
interchange Task and variable inputs transparently. So this works:

.. code:: python

   class Sub(RecordedTask):
       inputs = {'a': float, 'b': float}
       outputs = {'c': float}
       def _run(a, b):
           return a - b
   task1 = Sub(5, 1)
   task1.run()     # returns (4,)

and this also works (recall that `a - b` would be undefined if `a` were a tuple)

.. code:: python

   task2 = Sub(Add(5, 2, 3), 3)
   task2.run()        # returns (8,)

In this latter case the task `Sub` recognized that its `_run` routine is
expecting a packaged argument, and that it could unpack the result of `Add`
unambiguously. Unpacking will NOT happen if
  - The input task returns multiple outputs.
  - The input is specified as an `InputTuple`, since this is taken to mean
    that we are expecting packaged values.
  - The input is specified as a Task, since this is taken to mean that we are
    expecting task output.
This last reason is also why it is not recommended to specify both plain and
Task types for the same input.
