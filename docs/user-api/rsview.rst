.. _api-rsview:

Record store viewer
===================

The basic code to create a new viewer is

.. code::python

   from smttask.view import RSView
   rsview = RSView()

This offers a lazy iterator over the whole record store with filtering capabilities.
The elements of this iterator are read-only objects (“views”) of the underlying records.

See also the usage examples in the :ref:`Getting Started pages <usage-example-rsview>` and :ref:`In-depth documentation <in-depth_rsview>`.

Record store viewer
-------------------

.. autoclass:: smttask.view.recordstoreview.RecordStoreView
   :members:

Record view
-----------

.. autoclass:: smttask.view.recordview.RecordView
   :members:


List of record store filters
----------------------------

.. autosummary::

   smttask.view.recordfilter.generic_filter
   smttask.view.recordfilter.before
   smttask.view.recordfilter.after
   smttask.view.recordfilter.on
   smttask.view.recordfilter.label
   smttask.view.recordfilter.output
   smttask.view.recordfilter.outcome
   smttask.view.recordfilter.outcome_not
   smttask.view.recordfilter.outputpath
   smttask.view.recordfilter.reason
   smttask.view.recordfilter.reason_not
   smttask.view.recordfilter.script
   smttask.view.recordfilter.script_arguments
   smttask.view.recordfilter.task_name
   smttask.view.recordfilter.stdout_stderr
   smttask.view.recordfilter.tags
   smttask.view.recordfilter.tags_or
   smttask.view.recordfilter.tags_not
   smttask.view.recordfilter.user
   smttask.view.recordfilter.version

   