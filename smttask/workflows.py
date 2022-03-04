"""
Assembling Tasks into workflows
===============================

This module provides a pair of functions which can be used to combine multiple
tasks into containerized *workflows*, with a single set of parameters.

It borrows ideas from both `papermill`_ and `NotebookScripter`_. The idea to have
a *workflow* module, which defines global parameters and instantiates multiple
tasks depending on those parameters. A separate module (or interactive session)
may run the workflow module multiple times with different parameters.

The working principle is very similar to NotebookScripter.

Differences with NotebookScripter:

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
  
- Works will with any arguments – not just plain types like `str` and `float`
  
  + (Papermill converts all arguments to strings before passing them to the
  notebook being executed.)
  
- Workflows are run in the same process as the calling module

- Negligible overhead
  
  + Papermill creates a new anonymous notebook on each call, which in my
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
   :caption: runfile.py
   
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
previously_run_workflows = {}  # Used to prevent running a workflow after it has been modified
def run_workflow(module_name: str, package: str=None,
                 exenv: str="workflow", exenv_var="exenv", **parameters
    ) -> module:
    """
    Import (or reload) a module, effectively executing it as a script.
    The imported module can retrieve parameters, which are stored in
    `wcml.utils.script_args`.

    To allow the module to detect when it is being run as a workflow, an
    "execution environment" variable is injected into its global namespace.
    The default name for this variable is ``exenv`` and the default value
    ``"workflow"``; these defaults can be changed with the `exenv_var` and
    `exenv` parameters respectively.
    
    The (re)imported module is returned, allowing to retrieve values defined
    in its namespace.
    
    .. Important::
       For this to work, the script must include a call to `set_workflow_args`.
       
    .. Note::
       Workflow files must not be modified between calls to `run_workflow`:
       Python's introspection is not 100% robust with regards to reloaded modules,
       which may break smttask's reproducibility guarantees. (In particular,
       `inspect.getsource`, which is used to serialize functions, may return
       incorrect results.)

    Parameters
    ----------
    module_name: Name of the module as it appears in sys.modules
    package: If the module has not yet been imported, this is passed
        to `importlib.import_module`.
        It is required when `module_name` is relative.
    exenv: The value to which to set the global execution environment variable.
        If `None`, no variable is injected.
    exenv_var: The name of the global execution environment variable in the
        workflow module.
        If `None`, no variable is injected.
        
    **parameters: Parameters to pass to the script
    
    Returns
    -------
    The (re)imported module.
    
    See also
    --------
    set_workflow_args
    """
    global script_args, previously_run_workflows
    import importlib
    import sys
    import inspect
    from mackelab_toolbox.utils import stableintdigest
    parameters = parameters.copy()
    if exenv is not None and exenv_var is not None:
        parameters[exenv_var] = exenv
    script_args[module_name] = parameters
    if module_name in sys.modules:
        m_old = sys.modules[module_name]
        # Do the next check before trying to reload, in case a modification causes reload to fail
        previous_hash = previously_run_workflows.get(module_name)
        if previous_hash:
            if stableintdigest(inspect.getsource(m_old)) != previous_hash:
                raise RuntimeError(f"Workflow files (here: '{module_name}') "
                                   "must not be modified between calls to `run_workflow`.")
        m = importlib.reload(m_old)
        new_hash = stableintdigest(inspect.getsource(m))
        if previous_hash is None:
            previously_run_workflows[module_name] = new_hash
        elif new_hash != previous_hash:
            # There may be redundancy between this check and the other one; not sure if one check can catch all cases
            raise RuntimeError(f"Workflow files (here: '{module_name}') "
                               "must not be modified between calls to `run_workflow`.")
    else:
        m = importlib.import_module(module_name, package=package)
        previously_run_workflows[module_name] = stableintdigest(inspect.getsource(m))
    return m

def set_workflow_args(__name__: str, globals: Dict[str,Any], existing_only: bool=False):
    """
    To allow a notebook to be executed with `run_workflow`, place this
    immediately below its parameter block:
    
        retrieve_script_params(__name__, globals())
        
    :param:existing_only: If `True`, only variables already defined in the
       module before the call to `set_workflow_args` will be replaced by values
       passed to `run_workflow`.
        
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
                if not existing_only or k in globals:
                    globals[k] = v
