import click
import logging
from .base import Task, config

@click.group()
def cli():
    pass

@cli.command()
@click.argument('taskdesc', type=click.File('r'))
@click.option('--record/--no-record', default=True,
    help="Use `--no-record` to disable recording (and thereby also the check "
         "that the version control repository is clean).")
@click.option('-v', '--verbose', count=True,
    help="Specify up to 3 times (`-vvv`) to increase the logging level which "
         "is printed. Default is to print only warning and error messages.\n"
         "default: warning and up (error, critical)\n"
         "-v: info and up\n-vv: debug and up\n-vvv: everything.")
@click.option('-q', '--quiet', count=True,
    help="Turn off warning messages. Specifying multiple times will also "
         "turn off error and critical messages.")
@click.option('--debug/--no-debug', default=False,
    help="Launch the debugger before running task.")
def run(taskdesc, record, verbose, quiet, debug):
    """Execute the Task defined in TASKDESC.

    A taskdesc can be obtained by calling `.desc.save()` on an
    instantiated task."
    """
    verbose *= 10; quiet *= 10  # Logging levels are in steps of 10
    default = logging.WARNING
    loglevel = max(min(default+quiet-verbose,
                       logging.CRITICAL),
                   logging.DEBUG)
    logging.basicConfig(level=loglevel)
    config.record = record
    task = Task.load(taskdesc)
    taskdesc.close()
    if debug:
        import pdb; pdb.set_trace()
        pass
    task.run()

if __name__ == "__main__":
    cli()
