Tasks
=====

Specifying a task
-----------------
Tasks are most easily created by decorating a function:

.. code:: python

   from smttask import RecordedTask

   @RecordedTask
   def Add(a: float, b: float, n: int=10) -> float:
     for i in range(n):
       a += b
     return a

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

For more advanced usage, callable classes can also be used to define tasks.
This can be useful to define utility methods which depend on the task inputs.

.. code:: python

   from smttask import RecordedTask

   @RecordedTask
   class CombAdd(a: float, b: float, n: int=10, m: int=10) -> List[float]:

     def gen_combs(self):  # Yields n*m values
       for n in range(self.taskinputs.n):
         for m in range(self.taskinputs.m):
           yield (n, m)

     def __call__(self, a: float, b: float, n: int=10) -> float:
       vals = [n*a + m*b
               for n, m in self.gen_combs()]
       return vals

     def unpack_result(self, result):
       return {nm: r for nm, r in zip(self.gen_combs(),
                                      result)
              }

   task = CumAdd(a=2.1, b=1.1)
   # Get the (n,m) combinations used by the task
   task.gen_combs()
   # Run the task
   res = task.run()
   # Replace the list with a dictionary explicitely relating an (n,m) pair to a result
   resdict = task.unpack_result(res)

Note how in this example

- We define the task within the `__call__` method.
  The task method must have this name.
- We can use `self` within `__call__` without it being added to the task arguments.
  Any other name for the first argument *will not* work.
  (Or rather, it will be included in the task arguments.)
  It is not necessary it have a `self` argument, although if one is not needed,
  then probably decorating a function suffices.
- We use `self.taskinputs` to access the task inputs.
- The use of `gen_combs` to generate the `(n,m)` combinations avoids the need
  for external to know implementation details, like whether we loop over `n`
  or `m` first.
- We provide an `unpack_result` method; this can be a convenient pattern for
  saving outputs in a compressed format.
  The name `unpack_result` is not special and the function is not used
  internally by the task: it is only to simplify user code. [#unpack]_


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

Automatic expansion of inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following hypothetical task dependencies, which would be one way of loading a dataset distributed over multiple files:

.. code:: python

   @MemoizedTask
   def LoadDatafile(path: Path) -> Array:
     ...

   @MemoizedTask
   def LoadDataset(datafiles: list[LoadDatafile]) -> dict[str,Array]:
     ...

As a variation, one might want to keep a name associated to each file, and instead write the second task as

.. code:: python

   @MemoizedTask
   def LoadDataset(datafiles: dict[str,LoadDatafile]) -> dict[str,Array]:
     ...

In both of these cases, what the developer expects is clearly for each task entry in the list ``datafiles`` to be executed, the results of each ``LoadDatafile`` task combined into either a list or dictionary, before finally executing the ``LoadDataset`` task. This indeed what happens, both with built-in python types like ``list`` and ``dict``, and with custom types like *addict*’s ``Dict`` or *parameters*’ ``ParameterSet``. Therefore *in most cases this Just Works* as expected.

In certain cases however it may be necessary to adjust this behaviour. Under the hood, what *SumatraTask* does is inspect each argument, and if it is a `Collection <https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection>`_ (i.e. an iterable with a length), then the argument is expanded to inspect its elements. *Collections* include tuples, lists and sets, all of which are usually cheap to iterate through. Some iterable types don’t make sense to expand, like ``str`` and ``bytes``, and these are listed in the configuration option ``smttask.config.terminating_types``.

Therefore, to prevent expansion of the custom type ``MyType``, it only needs to be added to ``smttask.config.terminating_types`` (this is a set, which is why we use ``add``):

.. code:: python

   import smttask
   smttask.config.terminating_types.add(MyType)

Note that this is only necessary if ``isinstance(MyType, collections.abc.Collection)`` returns ``True`` AND that iterating through ``MyType`` is expensive. (E.g. if iteration involves costly I/O operations to load each element.)

Remember that inputs are generally *not* explicitely typed as tasks, and that our recommendations would be to type ``LoadDataset`` as

.. code:: python

   @MemoizedTask
   def LoadDataset(datafiles: list[Array]) -> dict[str,Array]:
     ...

Therefore it is not possible for *SumatryTask* to know before hand whether a collection passed as input may contain a Task to execute. Because of this, *all* inputs which are sized iterables are expanded. (Unless they match an entry in ``smttask.config.terminating_types``.) 


Limitations
-----------

New output types need their own serializers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Output types must be supported by Scitying or Pydantic, although with those packages' hooks for defining custom encoders and validators, this is almost always a solvable problem. [#almost_always]_ You can check whether a type ``MyType`` is supported by executing the following snippet:

.. code:: python

   from scityping.pydantic import BaseModel
   class Foo(BaseModel):
     a: MyType

If this raises an error stating that no validator was found, you will need to define a custom data type, as detailed in either the `Pydantic <https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types>`_ or the `Scityping <https://scityping.readthedocs.io/>`_ documentation. [#new_types]_

Footnotes
---------

.. [#unpack] We may add in the future a special function name, for defining
   a post-processor which is automatically applied to results before they are
   returned. This would make a decompression function completely transparent.

.. [#almost_always] Some types are explicitely not supported, such as the
   `Generator` type. In most cases however a workaround is still possible:
   for example, one can define a class with `__iter__()` and validation methods,
   and use that instead of the built-in `Generator` type.

.. [#new_types] *Scityping* was developed as an extension of *Pydantic* to allow
   the use of (abstract) base classes in type definitions, for example defining
   a field of type `Model` which accepts any subclass of `Model`. (In plain
   Pydantic values are always *coerced* to the target type.) Whether it is best
   to define new types with either *Scityping* or *Pydantic* largely depends on
   whether this use as abstract classes is needed.
