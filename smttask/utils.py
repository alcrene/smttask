import os
import os.path
from pathlib import Path
from typing import Any
from .config import config

# Copied from pydantic.utils
def lenient_issubclass(cls: Any, class_or_tuple) -> bool:
    """
    Equivalent to issubclass, but allows first argument to be non-type
    (in which case the result is ``False``).
    """
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)

def relative_path(src, dst, through=None, resolve=True):
    """
    Like pathlib.Path.relative_to, with the difference that `dst` does not
    need to be a subpath of `src`.

    In typical use, `src` would be point to directory, and `dst` to a file.

    Parameters
    ----------
    src: Path-like
        Returned path starts from here.
    dst: Path-like
        Returned path points to here. If ``relpath`` is the returned path, then
        `dst` points to the same location as concatenating `src` and ``relpath``.
    through: Path-like
        Generally does not need to be provided; by default it is obtained with
        `os.path.commonpath`. When provided, the returned path always goes
        through `through`, even when unnecessary.
    resolve: bool
        Whether to normalize both `src` and `dst` with `Path.resolve`.
        It is hard to construct an example where doing this has an undesirable
        effect, so leaving to ``True`` is recommended.

    Examples
    --------
    >>> from smttask.utils import relative_path
    >>> pout = Path("/home/User/data/output/file")
    >>> pin  = Path("/home/User/data/file")
    >>> relative_path(pin, pout, through="/home/User/data")
    PosixPath('../output/file')
    """
    src=Path(src); dst=Path(dst)
    if resolve:
        src = src.resolve()
        dst = dst.resolve()
    if through is None:
        through = os.path.commonpath([src, dst])
    if through != str(src):
        dstrelpath = dst.relative_to(through)
        srcrelpath  = src.relative_to(through)
        depth = len(srcrelpath.parents)
        uppath = Path('/'.join(['..']*depth))
        return uppath.joinpath(dstrelpath)
    else:
        return dst.relative_to(src)

import tempfile
class unique_process_num():
    """
    Sets the environment variables SMTTASK_PROCESS_NUM to the smallest integer
    not already assigned to an smttask process.
    Assigned numbers are tracked by files
    """
    def __enter__(self):
        tmpdir = Path(tempfile.gettempdir())
        file = None
        for n in range(config.max_processes):
            try:
                fpath = tmpdir/f"smttask_process-{n}.lock"
                file = open(fpath, 'x')
            except OSError:
                pass
            else:
                break
        if file is None:
            raise RuntimeError("Smttask: The maximum number of processes have "
                               "already been assigned. If there are stale "
                               "lock files, they can be removed from "
                               f"{tmpdir}/smttask_process-N.lock")
        self.file = file
        self.fpath = fpath
        os.environ["SMTTASK_PROCESS_NUM"] = str(n)
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        os.remove(self.fpath)
