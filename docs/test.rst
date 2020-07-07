Configuration
-------------
After installing `smttask` to your virtual environment, run

.. code::
   smt init --datapath [path/to/output/dir] --input [path/to/input/dir] [projectname]

The current implementation of `smttask` requires that the output directory
by a subdirectory of the input directory. The typical configuration is

.. code::
   cd /path/to/project/deathstar
   smt init --datapath data/run_dump --input data deathstar
