=============
Release notes
=============

Version 0.1
-----------

The first version of Parameters was part of NeuroTools.


Version 0.2
-----------

Version 0.2 is the first stand-alone version of Parameters. The main changes
are:

* addition of parameter set validation against a predefined schema;
* Python 3 support;
* NumPy and SciPy are no longer essential, although additional functionality is
  available if they are installed;
* added support for YAML-format parameter files;
* added support for opening remote parameter files where you are accessing the
  web via a proxy server;
* added export of parameter sets in LaTeX;
* addition of references, i.e. specifying one parameter by reference to another.

Version 0.2.1
-------------

* added simple operations to references, e.g. you can specify that a given
  parameter is twice the value of another;
* fixed bug with complex nested references.
