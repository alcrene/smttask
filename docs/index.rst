.. SumatraTask documentation master file, created by
   sphinx-quickstart on Thu Dec 30 20:02:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SumatraTask – User Manual
=========================

This packages extends the computation tracking capabilities of `Sumatra <https://pythonhosted.org/Sumatra/>`_ with “Task” constructs (the idea being borrowed from `Luigi <https://luigi.readthedocs.io/>`_). This allows better composability of tasks, and enables specifying a workflow *entirely in Python*. Compared to using makefiles, this is both simpler (since it doesn’t require learning a new syntax) and also more powerful: you have the full expressivity of Python at your disposable to specify the sequence of tasks to run. For example, workflows can be completely specified from within a full-blown Jupyter notebook, allowing one to produce *highly reproducible, easily documented analyses*.

Getting started
---------------

.. toctree::
   :maxdepth: 2

   getting_started/index.rst

In-depth
--------

.. toctree::
   :maxdepth: 2

   in_depth/index.rst

Reference
---------

.. toctree::
   :maxdepth: 2
  
   user-api/index.rst

.. Tutorials
.. ---------

.. .. toctree::
..    :maxdepth: 1

..    tutorials/index.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
