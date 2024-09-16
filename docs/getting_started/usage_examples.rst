**************
Usage examples
**************

.. _usage-example-rsview:

Interacting with the record store
=================================

Retrieve a particular record
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, we use ``rsview.list`` to force an iteration through all records; this stores all records in memory as a side-effect. For large stores this may take time, but allows multiple queries. (Otherwise underlying iterator may be consummable.)

.. code:: python

   from smttask import RecordStoreView as RSView
   rsview = RSView()
   rsview.list  
   rsview.last      # Last record in the store. Often, but not always, the most recent. Very fast since no comparison or iteration is required.
   rsview.first     # Complement to `.last`
   rsview.latest    # Most recent record. Compares records by date.
   rsview.earliest  # Complement to `.latest`.
   rsview.get("20240417-103729_d48d22'")  # Retrieve by record label

Filtering the record store view
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recreate the `Task` which created a record
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from smttask import RecordStoreView as RSView
   rsview = RSView()
   rec = rsview.last
   rec.task

Invalidate records
^^^^^^^^^^^^^^^^^^

Remove results for the *input* datastore, but keep them in the *output* datastore. This will cause them to be recomputed, while keeping the old results will still be accessible.

Most commonly this would be done after fixing a bug in a task, but there may be other uses as well. In the example below, we remove all 'AnalysisB' tasks peformed between the 5th and 15th of April

.. code:: python

   from smttask import RecordStoreView as RSView
   rsview = RSView()
   records = rsview.filter.after(20240405) \
                   .filter.before(20240416) \
                   .filter.task_name("AnalysisB")
   for rec in records:
     rec.invalidate()