.. SumatraTask documentation master file, created by
   sphinx-quickstart on Thu Dec 30 20:02:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SumatraTask – User Manual
=========================

This packages extends the computation tracking capabilities of `Sumatra <https://pythonhosted.org/Sumatra/>`_ with “Task” constructs (borrowed from `Luigi <https://luigi.readthedocs.io/>`_). This allows better composability of tasks, and enables a workflow based on *runfiles*. In particular, a runfile may be a full-blown Jupyter or RStudio notebook, allowing one to produce *highly reproducible, easily documented analyses*.

Getting started
---------------

.. toctree::
   :maxdepth: 1

   quickstart.md
   basics.rst
   workflows.rst
   cli.rst
   serialization.rst

Reference
---------

.. toctree::
  :maxdepth: 2
  
  user-api/user-api-index.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
