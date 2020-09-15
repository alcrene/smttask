import sumatra.projects
from json import JSONDecodeError
from mackelab_toolbox import smttk

# TODO: Avoid monkey patching, without duplicating code from smttk
# TODO: [in smttk?] Use Sumatra's own read-only RecordView as base.

class RecordList(smttk.RecordList):
    default_project_dir = None

    def __init__(self, project=None):
        project = project or sumatra.projects.load_project(self.default_project_dir)
        super().__init__(smttk.get_records(project.record_store, project=project.name))

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
