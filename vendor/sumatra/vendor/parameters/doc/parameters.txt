==========
Parameters
==========

.. testsetup::

   from parameters import Parameter, ParameterRange

At their simplest, individual parameters consist of a name and a value. The
value is either a simple type such as a numerical value or a string, or an
aggregate of such simple types, such as a set, list or array.

However, we may also wish to specify the physical dimensions of the parameter,
i.e., its units, and the range of permissible values.

It is also often useful to specify an object that generates numerical values or
strings, such as a random number generator, and treat that object as the
parameter.

To support all these uses, we define the :class:`Parameter` and :class:`ParameterRange`
classes, and various subclasses of the :class:`ParameterDist` abstract class, such as
:class:`GammaDist`, :class:`NormalDist` and :class:`UniformDist`.


The :class:`Parameter` class
----------------------------

Here are some examples of creating :class:`Parameter` objects:

.. doctest::

    >>> i1 = Parameter(3)
    >>> f1 = Parameter(6.2)
    >>> f2 = Parameter(-65.3, "mV")
    >>> s1 = Parameter("hello", name="message_to_the_world")
    
The parameter name, units, value and type can be accessed as attributes:

.. doctest::

    >>> i1.value
    3
    >>> f1.type
    <type 'float'>
    >>> f2.units
    'mV'
    >>> s1.name
    'message_to_the_world'

:class:`Parameter` objects are not hugely useful at the moment. The units are not
used for checking dimensional consistency, for example, and :class:`Parameter`
objects are not drop-in replacements for numerical values - you must always use
the :attr:`value` attribute to access the value, whereas it might be nice to define,
for example, a class :class:`IntegerParameter` which was a subclass of the built-in
:class:`int` type.


The :class:`ParameterRange` class
---------------------------------

When investigating the behaviour of a model or in doing sensitivity analysis, it
is often useful to run a model several times using a different value for a
certain parameter each time (also see the :meth:`iter_range_keys()` and similar
methods of the :class:`ParameterSet` class, below). The :class:`ParameterRange` class
supports this. Some usage examples:

.. doctest::

    >>> tau_m_range = ParameterRange([10.0, 15.0, 20.0], "ms", "tau_m")
    >>> tau_m_range.name
    'tau_m'
    >>> tau_m_range.next()
    10.0
    >>> tau_m_range.next()
    15.0
    >>> [2*tau_m for tau_m in tau_m_range]
    [20.0, 30.0, 40.0]


The :class:`ParameterDist` classes
----------------------------------

As with taking parameter values from a series or range, it is often useful to
pick values from a particular random distribution. Three classes are available:
:class:`UniformDist`, :class:`GammaDist` and :class:`NormalDist`. Examples::
    
    >>> ud = UniformDist(min=-1.0, max=1.0)
    >>> gd = GammaDist(mean=0.5, std=1.0)
    >>> nd = NormalDist(mean=-70, std=5.0)
    >>> ud.next()
    array([-0.56342352]) 
    >>> gd.next(3)
    array([ 0.04061142,  0.05550265,  0.23469344])
    >>> nd.next(2)
    array([-76.18506715, -68.71229944]) 



