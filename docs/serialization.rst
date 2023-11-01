*************
Serialization
*************

*smttask* relies on serialization for two core aspects of its functionality:

To generate task description files
  These are portable, plain-text JSON files which fully describe a computation. They archive the full specification of a computation (functions + parameters), and along with the environment information recorded by Sumatra, allow it to be repeated at a later.
  Task descriptions can also be sent to and executed on a different machine, for example an HPC cluster.

To compute and compare task digests.
  Task digests are obtained by hashing its task description, using a deterministic algorithm. They are used to determine output file names, and whether a task has already been executed.
  It is for this use case that serialization needs to be perfectly consistent. Any change (for example the order of JSON fields, or the byte representation of an array) will lead to a different digest, and identical tasks not being recognized as such.

Implementation
--------------

Each `Task` defines a `TaskInput` type, which is a subclass of :external:py:class:`pydantic.BaseModel` and provides a :py:meth:`json()` method which exports a model to a JSON string. :py:class:`BaseModel` already knows how to serialize most common data types, takes care of recursively serializing and deserializing nested objects, and provides hooks to configure custom JSON encoders and decoders. Thus the only thing we need to add are a custom encoders for types not already recognized by :py:mod:`pydantic`.
We do this with the scityping_ package, which provides two key features:

- Definitions for many additional, science-focused types like NumPy arrays;
- A mechanism for defining new serialization rules, for both existing and new types.

Please refer to scityping_’s documentation for more information.

The second feature is especially valuable because it allows us to build workflows which consume or produce types from external libraries, which may not a priori be serializable, without modifying the types themselves. Moreover, this can be done without worrying about the package import order, which is something very difficult to do with vanilla Pydantic.

To increase consistency across platforms, `TaskInput` disables compression of NumPy arrays when computing a digest.

.. _Pydantic: https://pydantic-docs.helpmanual.io
.. _scityping: https://scityping.readthedocs.io/en/stable/getting_started.html
