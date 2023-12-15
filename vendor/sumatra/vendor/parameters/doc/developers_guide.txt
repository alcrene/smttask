=================
How to contribute
=================

Reporting a bug, requesting improvements
----------------------------------------

If you find a bug in Parameters or would like to suggest an improvement, go to
the `issue_tracker`_ and check whether someone else already reported the same
bug or requested the same improvement. If not, click on "New Issue" and
describe the problem, preferably including a code sample that reproduces the
problem.


Setting up a development environment
------------------------------------

We strongly suggest you use virtualenv_ to isolate your work on Parameters
from your default Python installation. It is best to create two virtualenvs,
one using Python 2.7, the other using Python 3.2 or later.

To run the tests, we suggest installing at least nose, NumPy and PyYAML. You
may also need to install SciPy.

To obtain the Parameters source code, you will need a GitHub account. You
should then fork https://github.com/NeuralEnsemble/parameters (see the
`GitHub documentation`_ if you are new to GitHub) and clone your own copy of
the repository::

  git clone git@github.com:<username>/parameters.git

You will then need either to install Parameters in your virtualenv(s) or
otherwise add it to your :envvar:`PYTHONPATH`.


Running the test suite
----------------------

The easiest way to run the test suite is to run :command:`nosetests` in the root
of the source directory or in the :file:`test` subdirectory.

Tests should be run with both Python 2.7 and some version of Python >= 3.2.


Coding standards and style
--------------------------

Parameters is intended to support Python 2.7 and all versions of Python >= 3.2,
using a single code base. For guidance on achieving this, see
`Porting to Python 3`_ and the `Python 3 Porting Guide`_.


Contributing code
-----------------

When you are happy with your changes to the code, all the tests pass with all
supported versions of Python, open a `pull request`_ on Github.


Making a release
----------------

If you are the release manager for Parameters, here is a checklist for
making a release:

* update the version numbers in :file:`setup.py`, :file:`parameters/__init__.py`,
  :file:`doc/conf.py` and :file:`doc/installation.txt`
* update :file:`doc/changelog.txt`
* run all the tests with both Python 2 and Python 3
* ``python setup.py sdist upload``
* rebuild the documentation at http://parameters.readthedocs.org/
* commit the changes, tag with release number, push to GitHub
* bump the version numbers

.. _`GitHub documentation`: http://help.github.com/fork-a-repo/
.. _`issue_tracker`: https://github.com/NeuralEnsemble/parameters/issues
.. _virtualenv: http://www.virtualenv.org/
.. _`pull request`: http://help.github.com/send-pull-requests/
.. _`Python 3 Porting Guide`: http://docs.pythonsprints.com/python3_porting/index.html
.. _`Porting to Python 3`: http://python3porting.com/noconv.html