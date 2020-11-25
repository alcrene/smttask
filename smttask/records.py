import logging
from pathlib import Path
from json import JSONDecodeError
from tqdm.auto import tqdm
import sumatra.projects
from mackelab_toolbox import smttk
from mackelab_toolbox import iotools
from smttask.base import Task
from smttask import utils, config

logger = logging.getLogger(__file__)

###############$
# Utility functions
def _rename_to_free_file(path):
    new_f, new_path = iotools.get_free_file(path, max_files=100)
    new_f.close()
    os.rename(path, new_path)
    return new_path


# TODO: Avoid monkey patching, without duplicating code from smttk
# TODO: [in smttk?] Use Sumatra's own read-only RecordView as base.

class RecordList(smttk.RecordList):
    default_project_dir = None

    def __init__(self, project=None):
        project = project or sumatra.projects.load_project(self.default_project_dir)
        super().__init__(smttk.get_records(project.record_store, project=project.name))

    def rebuild_input_datastore(self):
        inroot = Path(config.project.input_datastore.root)

        symlinks = {}
        # `symlinks` is indexed by 'inpath', and the record list iterated
        # from oldest to newest, so that newer records overwrite older ones
        logger.info("Iterating through records...")
        # for record in tqdm(self.list[::-1]):
        for record in tqdm(self.list[:2]):
            abort = False

            try:
                task = Task.from_desc(record.parameters)
            except Exception:  # Tasks might raise any kind of exception
                logger.debug(f"Skipped record {record.label}: Task could not be recreated.")
                continue
            # Recompute the task digests
            relpaths = task.outputpaths

            # Get the file path associated to each output name
            # Abort if any of the names are missing, or if they cannot be unambiguously resolved
            output_paths = {}
            for nm in task.outputpaths:
                # Although the computed output paths may differ from the
                # recorded ones, the variable names should still be the same
                # Get the output path associated to this name
                paths = [path for path in record.outputpath
                              if nm in Path(path).stem.split(
                                  '_', task.digest.count('_') )[-1]  # 'split' removes digest(s)
                        ]
                if len(paths) == 0:
                    logger.debug(f"No output file containing {nm} is associated to record {record.label}.")
                    abort = True
                    break
                elif len(paths) >= 2:
                    logger.debug(f"Record {record.label} has multiple output files containing {nm}.")
                    abort = True
                    break
                output_paths[nm] = Path(paths[0])
            if abort:
                continue

            # Compute the new symlinks
            for nm, relpath in relpaths.items():
                outpath = output_paths[nm].resolve()  # Raises error if the path does not exist
                inpath = inroot/relpath.with_suffix(outpath.suffix)
                symlinks[inpath] = utils.relative_path(inpath.parent, outpath)

        # Create all the symlinks
        # Iterate through `symlinks` and create them the links defined therein.
        # If a file already exists where we want to place a link, we do the
        # following:
        #   - If it's a link that already points to the right location, do nothing
        #   - If it's a link that points to another location, replace it
        #   - If it's an actual file, append a number to its filename before
        #     creating the link.
        logger.info("Creating symlinks...")
        num_created_links = 0
        for inpath, relpath in tqdm(symlinks.items()):
            src = inpath.parent/relpath
            if inpath.is_symlink():
                if inpath.resolve() == src.resolve():
                    # Present link is the same we want to create; don't do anything
                    continue
                else:
                    # Remove the deprecated link
                    inpath.unlink()
                    logger.debug(f"Removed deprecated link '{inpath} -> {inpath.absolute()}'")

            if inpath.exists():
                assert not inpath.is_symlink()
                # Rename the path so as to not lose data
                renamed_path = rename_to_free_file(move['new path'])
                logger.debug(f"Previous file '{inpath}' was renamed to '{renamed_path}'.")
            else:
                # Make sure the directory hierarchy exists
                inpath.parent.mkdir(exist_ok=True)

            inpath.symlink_to(relpath)
            logger.debug(f"Added link '{inpath}' -> {relpath}")
            num_created_links += 1

        logger.info(f"Created {num_created_links} new links in {inroot}.")

    # Shorthand
    rebuild_links = rebuild_input_datastore

# We are going to monkey patch the new RecordView into smttk.RecordView
orig_RecordView = smttk.RecordView
class RecordView(orig_RecordView):
    """
    Adds a method to recovering task output, based on the task output format
    (i.e. [task_digest]__[var name].json).
    """
    # Within `get_output`, these are tried in sequence, and the first
    # to load successfully is returned.
    # At present these must all be subtypes of pydantic.BaseModel
    data_models = []

    def get_output(self, name="", data_models=()):
        """
        Load the output data associated the provided name.
        (The association to `name` is done by matching the output path.)
        `name` should be such that that exactly one output path matches;
        if the record produced only one output, `name` is not required.

        After having found the output file path, the method attempts to
        load it with the provided data models; the first to succeed is returned.
        A list of data models can be provided via the `data_models`, but
        in general it is more convenient to set a default list with the class variable
        `RecordView.data_models`. Models passed as arguments have precedence.
        """
        data_models = list(data_models) + self.data_models
        if not data_models:
            raise TypeError("`get_output` requires at least one data model, "
                            "given either as argument or by setting the class "
                            "attribute `RecordView.data_models`.")
        # TODO: Allow to specify extension, but still match files with _1, _2… suffix added by iotools ?
        # TODO: If name does not contain extension, force match to be right before '.', allowing for _1, _2… suffix ?
        # if '.' not in name:
        #     name += '.'
        paths = []
        for path in self.outputpath:
            if name in path:
                paths.append(path)
        if len(paths) == 0:
            raise FileNotFoundError(f"The record {self.label} does not have an "
                                    f"output file with name '{name}'")
        elif len(paths) > 1:
            paths_str = '\n'.join(paths)
            raise ValueError(f"The record {self.label} has multiple files with "
                             f"the name '{name}':\n{paths_str}")
        else:
            for F in data_models:
                try:
                    return F.parse_file(paths[0])
                except JSONDecodeError:
                    pass
            raise JSONDecodeError(f"The file at location {paths[0]} is unrecognized "
                                  f"by any of the following types: {data_models}.")


smttk.RecordView = RecordView
