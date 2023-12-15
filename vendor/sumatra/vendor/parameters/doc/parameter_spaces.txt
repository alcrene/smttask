================
Parameter spaces
================

.. testsetup::

   from parameters import ParameterSpace, ParameterRange

The :class:`ParameterSpace` class is a subclass of :class:`ParameterSet` that is
allowed to contain :class:`ParameterRange` and :class:`ParameterDist` objects as
parameters. This turns the single point in parameter space represented by a
:class:`ParameterSet` into a set of points. For example, the following definition
creates a set of six points in parameter space, which can be obtained in turn
using the :meth:`iter_inner()` method:

.. doctest::

    >>> PS = ParameterSpace({
    ...        'x': 999,
    ...        'y': ParameterRange([10, 20]),
    ...        'z': ParameterRange([-1, 0, 1])
    ... })
    >>> for P in PS.iter_inner():
    ...     print P
    {'y': 10, 'x': 999, 'z': -1}
    {'y': 20, 'x': 999, 'z': -1}
    {'y': 10, 'x': 999, 'z': 0}
    {'y': 20, 'x': 999, 'z': 0}
    {'y': 10, 'x': 999, 'z': 1}
    {'y': 20, 'x': 999, 'z': 1}

Putting parameter distribution objects inside a :class:`ParameterSpace` allows an
essentially infinite number of points to be generated::

    >>> PS2 = ParameterSpace({
    ...    'x': UniformDist(min=-1.0, max=1.0),
    ...    'y': GammaDist(mean=0.5, std=1.0),
    ...    'z': NormalDist(mean=-70, std=5.0)
    ... })
    >>> for P in PS2.realize_dists(n=3):
    ...     print P
    {'y': 1.81311773668, 'x': 0.883293989399, 'z': -73.5871002759}
    {'y': 0.299391158731, 'x': 0.371474054049, 'z': -68.6936045978}
    {'y': 2.90108202422, 'x': -0.388218831787, 'z': -68.6681724449}
