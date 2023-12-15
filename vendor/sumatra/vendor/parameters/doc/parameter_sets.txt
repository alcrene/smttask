==============
Parameter sets
==============

.. testsetup::

   from parameters import ParameterSet


A problem with parameter sets for large-scale, detailed models is that the list
of parameters gets very long and unwieldy, and due to the typically hierarchical
nature of such models, the individual parameter names can also get very long,
e.g., ``v1_layer5_pyramidal_apical_dend_gbar_na``.

A solution to this is to give the parameter set a hierarchical structure as well,
which allows the top-level list of parameters to be very short (e.g. ``v1``,
``retina`` and ``lgn`` for a visual system simulation) since the top-level
parameters are themselves parameter sets.

The simplest way to implement this in Python is using nested dicts. One
disadvantage of this is that accessing deeply-nested parameters can be very
verbose, e.g. ``v1['layer5']['pyramidal']['apical_dend']['na']['gbar']``. A
second disadvantage is that it is tedious to flatten the hierarchy when
this becomes necessary, e.g. for serialisation - writing to file, etc.

For these reasons we have created a :class:`ParameterSet` class, which:

1. allows a more convenient notation;
  
2. enables subsets of the parameters, lower in the hierarchy, to be passed
   around by themselves;
  
3. provides convenient methods for reading from/writing to file and for
   determining the differences between two different parameter sets.
  
An example of the notation is ``v1.layer5.pyramidal.apical_dend.na.gbar``, which
requires only a single `.` for each level in the hierarchy rather than two
"``'``"s, a "``[``" and a "``]``". This is not much shorter than
``v1_layer5_pyramidal_apical_dend_gbar_na`` - the difference is that
``v1.layer5.pyramidal`` is itself a :class:`ParameterSet` object that can be passed
as an argument to the pyramidal cell object, which doesn't care about
``v1.layer4.spinystellate``, let alone ``retina.ganglioncell.magno.tau_m``
(while ``v1_layer5_pyramidal`` is just a :class:`NameError`).


The :class:`ParameterSet` class
-------------------------------

Creation
~~~~~~~~

:class:`ParameterSet` objects may be created from a dict:

.. doctest::

    >>> sim_params = ParameterSet({'dt': 0.11, 'tstop': 1000.0})
    
or loaded from a URL:

.. doctest::

    >>> exc_cell_params = ParameterSet("https://neuralensemble.org/svn/NeuroTools/trunk/doc/example.param")

They may be nested:

.. doctest::

    >>> inh_cell_params = ParameterSet({'tau_m': 15.0, 'cm': 0.5})
    >>> network_params = ParameterSet({'excitatory_cells': exc_cell_params, 'inhibitory_cells': inh_cell_params})
    >>> P = ParameterSet({'sim': sim_params, 'network': network_params}, label="my_params")

Note that although we show here only numerical parameter values,
:class:`Parameter`, :class:`ParameterRange` and :class:`ParameterDist` objects, as well as
strings, may also be parameter values.

.. todo:: describe references ('ref' and the :class:`ParameterReference` class)


Navigation
~~~~~~~~~~
    
Individual parameters may be accessed/set using dot notation:

.. doctest::

    >>> P.sim.dt
    0.11
    >>> P.network.inhibitory_cells.tau_m
    15.0
    >>> P.network.inhibitory_cells.cm = 0.75
    
or the usual dictionary access notation:

.. doctest::

    >>> P['network']['inhibitory_cells']['cm']
    0.75
    
or mixing the two (which may be required if some of the parameter names contain
spaces):

.. doctest::

    >>> P['network'].excitatory_cells['tau_m']
    10.0


Viewing and saving
~~~~~~~~~~~~~~~~~~

To see the entire parameter set at once, nicely formatted use the :meth:`pretty()`
method:

.. doctest::

    >>> print P.pretty()
    {
      "network": {
        "excitatory_cells": url("https://neuralensemble.org/svn/NeuroTools/trunk/doc/example.param"),
        "inhibitory_cells": {
          "tau_m": 15.0,
          "cm": 0.75,
        },
      },
      "sim": {
        "tstop": 1000.0,
        "dt": 0.11,
      },
    }

By default, if the :class:`ParameterSet` contains other :class:`ParameterSet`\s that were
loaded from URLs, these will be represented with a :func:`url()` function in the
output, but there is also the option to expand all URLs and show the full
contents:

.. doctest::

    >>> print P.pretty(expand_urls=True)
    {
      "network": {
        "excitatory_cells": {
          "tau_refrac": 0.11,
          "tau_m": 10.0,
          "cm": 0.25,
          "synI": {
            "tau": 10.0,
            "E": -75.0,
          },
          "synE": {
            "tau": 1.5,
            "E": 0.0,
          },
          "v_thresh": -57.0,
          "v_reset": -70.0,
          "v_rest": -70.0,
        },
        "inhibitory_cells": {
          "tau_m": 15.0,
          "cm": 0.75,
        },
      },
      "sim": {
        "tstop": 1000.0,
        "dt": 0.11,
      },
    }

If a :class:`ParameterSet` was loaded from a URL, it may be modified then saved back
to the same URL, provided the protocol supports writing:

.. doctest::

    >>> exc_cell_params.save()
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
      File "parameters.py", line 266, in save
        raise Exception("Saving using the %s protocol is not implemented" % scheme)
    Exception: Saving using the https protocol is not implemented
    
or saved to a different URL:

.. doctest::

    >>> exc_cell_params.save(url="file:///tmp/exc_params")

The file format is the same as that produced by the :meth:`pretty()` method.

Copying and converting
~~~~~~~~~~~~~~~~~~~~~~

A :class:`ParameterSet` can be used simply as a dictionary, but can also be
converted explicitly to a :class:`dict` if required:

.. doctest::

    >>> print sim_params.as_dict()
    {'tstop': 1000.0, 'dt': 0.11}

[need to say something about :meth:`tree_copy()`]

Iteration
~~~~~~~~~

There are several different ways to iterate over all or part of the
:class:`ParameterSet` object. :meth:`keys()`, :meth:`values()` and :meth:`items()` work as for
:class:`dict`\s. For the sake of more readable code, :meth:`names()` is provided as an
alias for :meth:`keys()` and :meth:`parameters()` as an alias for :meth:`items()`:

.. doctest::
   :options: +NORMALIZE_WHITESPACE

    >>> P.names()
    ['network', 'sim']
    >>> exc_cell_params.parameters()
    [('tau_refrac', 0.11), ('tau_m', 10.0), ('cm', 0.25),
     ('synI', {'tau': 10.0, 'E': -75.0}), ('synE', {'tau': 1.5, 'E': 0.0}),
     ('v_thresh', -57.0), ('v_reset', -70.0), ('v_rest', -70.0)]
    
To flatten nested parameter sets, i.e., the iterate recursively over all
branches of the tree, the the :meth:`flatten()` method returns a :class:`dict` with keys
created by joining the names at each hierarchical level with a separator
character ('.' by default):

.. doctest::
   :options: +NORMALIZE_WHITESPACE

    >>> network_params.flatten()
    {'excitatory_cells.synI.E': -75.0, 'excitatory_cells.v_rest': -70.0,
     'excitatory_cells.tau_refrac': 0.11, 'excitatory_cells.v_reset': -70.0,
     'excitatory_cells.v_thresh': -57.0, 'excitatory_cells.tau_m': 10.0,
     'excitatory_cells.synI.tau': 10.0, 'excitatory_cells.cm': 0.25,
     'inhibitory_cells.cm': 0.75, 'excitatory_cells.synE.tau': 1.5,
     'excitatory_cells.synE.E': 0.0, 'inhibitory_cells.tau_m': 15.0}

while the :meth:`flat()` method returns a generator which yields
``(name, value)`` tuples.:

.. doctest::

    >>> for x in network_params.flat():
    ...   print x
    ('excitatory_cells.tau_refrac', 0.11)
    ('excitatory_cells.tau_m', 10.0)
    ('excitatory_cells.cm', 0.25)
    ('excitatory_cells.synI.tau', 10.0)
    ('excitatory_cells.synI.E', -75.0)
    ('excitatory_cells.synE.tau', 1.5)
    ('excitatory_cells.synE.E', 0.0)
    ('excitatory_cells.v_thresh', -57.0)
    ('excitatory_cells.v_reset', -70.0)
    ('excitatory_cells.v_rest', -70.0)
    ('inhibitory_cells.tau_m', 15.0)
    ('inhibitory_cells.cm', 0.75)


The :class:`ParameterTable` class
---------------------------------

.. todo:: describe this
