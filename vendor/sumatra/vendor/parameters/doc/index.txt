========================================
Parameters - hierarchical parameter sets
========================================

We consider it to be best practice to cleanly separate the parameters of a model
from the model itself. At the least, parameters should be defined in a separate
section at the start of a file. Ideally, they should be defined in a separate
file entirely. This makes version control easier, since the model code typically
changes less often than the parameters, and makes it easier to track a
simulation project, since the parameter sets can be stored in a database,
displayed in a GUI, etc.


The **Parameters** package provides Python classes to make it easier to work with
parameter sets for complex models.  In particular it provides tools for

* working with parameters for models that have a deep hierarchical structure;
* specifying that a parameter value should be drawn from a random distribution;
* specifying a range of values, for example for performing a sensitivity analysis;
* specifying the physical dimensions and range of permissible values of parameters;
* defining and iterating over multiple points in a parameter space;
* validation of parameter sets against a pre-defined schema.


Contents:

.. toctree::
   :maxdepth: 1

   installation
   parameters
   parameter_sets
   parameter_spaces
   validation
   changelog
   reference
   developers_guide

.. note:: Parameters was previously part of the NeuroTools package, but is now
          developed and distributed separately.
