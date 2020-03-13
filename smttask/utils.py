from pathlib import Path

def relative_path(src, dst, through=None):
    """
    Like pathlib.Path.relative_to, with the additional possibility to
    define a common parent.
    When provided, the returned path always goes through `through`, even when
    unnecessary.

    Examples
    --------
    >>> from smttask.utils import relative_path
    >>> pout = Path("/home/User/data/output/file")
    >>> pin  = Path("/home/User/data/file")
    >>> relative_path(pin, pout, through="/home/User/data")
    PosixPath('../output/file')
    """
    src=Path(src); dst=Path(dst)
    if through is not None:
        dstrelpath = dst.relative_to(through)
        srcrelpath  = src.relative_to(through)
        depth = len(srcrelpath.parents) - 1
            # The number of '..' we need to prepend to the link
            # The last parent is the cwd ('.') and so doesn't count
        uppath = Path('/'.join(['..']*depth))
        return uppath.joinpath(dstrelpath)
    else:
        return dst.relative_to(src)
