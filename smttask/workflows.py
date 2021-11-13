"""
Assembling Tasks into workflows

This module provides a pair of functions which can be used to combine multiple
tasks into containerized *workflows*, with a single set of parameters.

It borrows ideas from both `papermill`_ and NotebookScripter. The idea to have
a *workflow* module, which defines global parameters and instantiates multiple
tasks depending on those parameters. A separate module (or interactive session)
may run the workflow module multiple times with different parameters.

The working principle is very similar to NotebookScripter.

Differences with NotebookScripter

- Works with plain Python modules
  + To execute Jupyter notebooks, pair them first to a Python module with Jupytext.
- Less boilerplate
  - Workflow parameters are declared as normal global parameters, rather than
    retrieved with a ``receive_parameter`` function.
    + This makes scripts more portable and easier to read.
    + Since this is also the format used by papermill, workflows can also be
      executed with papermill, if desired.
- Fewer execution options
  + No option to run in a separate process.
  + No Jupyter-specific functionality.
- The workflow module is imported as a normal module, in addition to being
  returned.

Differences with papermill:

- Works with plain Python modules
  + To execute Jupyter notebooks, pair them first to a Python module with Jupytext.
- Works will with any arguments – not just plain types like `str` and `float`.
  (Papermill converts all arguments to strings before passing them to the
  notebook being executed.)
- Workflows are run in the same process as the calling module.
- Negligible overhead
  Papermill creates a new anonymous notebook on each call, which in my
  experience can take up to 10 seconds. This can be an issue when attempting
  to do a parameter scan by repeating a workflow multiple. (Note that if the
  workflow only *instantiates* tasks, in most cases it should complete in less
  than a second).

Usage
-----

.. code-block::
   :caption: workflow_module.py
   from smttask.workflows import set_workflow_args
   
   # Default values for workflow parameters
   a1 = 0
   b1 = 1.
   
   # If executed externally, replace the parameters with those passed as args
   # (No effect if workflow_module.py is called directly)
   set_workflow_args(__name__, globals())

   taskA = Task1(a=a1, b=b1)
   taskB = Task2(a=task1, b=...)  # Workflow => Make Task2 depend on Task1
   
.. code-block::
   :caption: 
   
   from smttask.workflows import run_workflow
   
   wf_list = [run_workflow(a1=a1) for a1 in (-1, 0, 1, 2)]
      # Uses default value for b1

   # Execute all workflows:
   for wf in wf_list:
       wf.taskB.run()

   # Alternatively, save tasks for later execution
   for wf in wf_list:
       wf.taskB.save()

.. Note::
   Only parameters already defined before the call to `set_workflow_args` will
   be replaced.

.. Note::
   The implementation uses a global variable is used to pass parameters between
   modules. The is definitely a hack, but probably no worse than the magic
   that papermill or NotebookScripter themselves use.

.. _papermill: papermill.readthedocs.io/en/latest/
.. _NotebookScripter: https://github.com/breathe/NotebookScripter
"""
from __future__ import annotations

script_args = {}
def run_workflow(module_name: str, package: str=None, **parameters
    ) -> module:
    """
    Import (or reload) a module, effectively executing it as a script.
    The imported module can retrieve parameters, which are stored in
    `wcml.utils.script_args`.

    Automatically adds the parameter `exec_environment` and sets it to
    'script' if it is not provided.
    
    The (re)imported module is returned, allowing to retrieve values defined
    in its namespace.
    
    .. Note::
       For this to work, the script must include a call to `set_workflowt_args`.

    Parameters
    ----------
    module_name: Name of the module as it appears in sys.modules
    package: If the module has not yet been imported, this is passed
        to `importlib.import_module`.
        It is required when `module_name` is relative.
    **parameters: Parameters to pass to the script
    
    Returns
    -------
    The (re)imported module.
    
    See also
    --------
    set_workflow_args
    """
    global script_args
    import importlib
    import sys
    parameters = parameters.copy()
    if 'exec_environment' not in parameters:
        parameters['exec_environment'] = 'script'
    script_args[module_name] = parameters
    if module_name in sys.modules:
        m = importlib.reload(sys.modules[module_name])
    else:
        m = importlib.import_module(module_name, package=package)
    return m

def set_workflow_args(__name__: str, globals: Dict[str,Any]):
    """
    To allow a notebook to be executed with `run_workflow`, place this
    immediately below its parameter block:
    
        retrieve_script_params(__name__, globals())
        
    Only variables already defined in the module before the call to
    `set_workflow_args` will be replaced by values  passed to `run_workflow`.
        
    .. todo:: Make __name__ and globals optional, using the stack trace to
       get values which work in most situations.
       
    See also
    --------
    run_workflow
    """
    if __name__ != "__main__":  # Make this call safe for interactive sessions
        # Running within an import
        #  - if run through `run_workflow`, there will be parameters in `script_args`
        #    which should replace the current values
        if __name__ in script_args:
            for k, v in script_args[__name__].items():
                if k in globals:
                    globals[k] = v
