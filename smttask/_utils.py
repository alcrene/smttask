"""
Elements of `utils` which don't depend on any other module within smttask,
and therefore can be imported anywhere without causing import cycles.

This is a private module used internally to solve import cycles;
external code that uses these functions should import them from *smttask.utils*.
"""
from __future__ import annotations

import os
import os.path
import logging
from pathlib import Path
from typing import Union, Any, List

logger = logging.getLogger(__name__)

__all__ = ["NO_VALUE", "lenient_issubclass", "relative_path",
           "parse_duration_str", "sync_one_way"]

#################
# Constants

import mackelab_toolbox as mtb
import mackelab_toolbox.utils

NO_VALUE = mtb.utils.sentinel("value not provided")  # Default value in function signatures

#################
# Misc

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

# Surprisingly, dateutils doesn't seem to provide this functionality
from decimal import Decimal
def parse_duration_str(duration_string) -> Decimal:
    """
    Parse a human readable string indicating a duration in hours, minutes, seconds.
    Returns the number of seconds as an Decimal.

    Examples::
    >>> parse_duration_str("1min")                     # 60
    >>> parse_duration_str("1m")                       # 60
    >>> parse_duration_str("1 minutes")                # 60
    >>> parse_duration_str("1h23m2s")                  # 60**2 + 23*60 + 2
    >>> parse_duration_str("1day 1hour 23minutes 2seconds") # 24*60**2 + 60**2 + 23*60 + 2

    Unusual but also valid::
    >>> parse_duration_str("1 min 1 min")              # 120

    Loosely based on: https://gist.github.com/Ayehavgunne/ac6108fa8740c325892b
    """
    duration_string = duration_string.lower()
    durations = {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    duration_multipliers = {'days': 24*60*60, 'hours': 60*60, 'minutes': 60, 'seconds': 1}
    num_str = []     # Accumulate numerical characters to parse
    mul_str = []
    parsed_num = None  # Numerical value after parsing
    def add_amount(amount, multiplier_str):
        if amount is None:
            raise ValueError(f"No amount specified for interval '{multiplier_str}'.")
        key = [k for k in durations if k.startswith(multiplier_str)]
        if not len(key) == 1:
            raise ValueError(f"'{multiplier_str}' is not a valid interval specifier. "
                             f"Accepted values are: {durations.keys()}.")
        durations[key[0]] += amount
    for character in duration_string:
        if character.isnumeric() or character == '.':
            if mul_str:
                # Starting a new amount – add the previous one to the totals
                add_amount(parsed_num, ''.join(mul_str))
                mul_str = []
            num_str.append(character)
        elif character.isalpha():
            if num_str:
                # First character of an interval specifier
                parsed_num = Decimal(''.join(num_str))
                num_str = []
            mul_str.append(character)
    if parsed_num or mul_str:
        add_amount(parsed_num, ''.join(mul_str))
        parsed_num = None
    return sum(durations[k]*duration_multipliers[k] for k in durations)


#################
# Sumatra

def sync_one_way(src: Union[RecordStore, str, Path],
                 target: Union[RecordStore, str, Path],
                 project_name: str
    ) -> List[str]:
    """
    Merge the records from `src` into `target`.
    Equivalent to Sumatra's RecordStore.sync(), except that only the `target`
    store is updated.

    Where the two stores have the same label (within a project) for
    different records, those records will not be synced. The function
    returns a list of non-synchronizable records (empty if the sync worked
    perfectly).
    """
    from sumatra.recordstore import get_record_store
    if isinstance(src, (str, Path)):
        src = get_record_store(str(src))
    if isinstance(target, (str, Path)):
        target = get_record_store(str(target))
    
    # NB: Copied almost verbatim from sumatra.recordstore.base
    src_labels = set(src.labels(project_name))
    target_labels = set(target.labels(project_name))
    only_in_src = src_labels.difference(target_labels)
    # only_in_target = target_labels.difference(src_labels)
    in_both = src_labels.intersection(target_labels)
    non_synchronizable = []
    for label in in_both:
        if src.get(project_name, label) != target.get(project_name, label):
            non_synchronizable.append(label)
    for label in only_in_src:
        target.save(project_name, src.get(project_name, label))
    # for label in only_in_target:
    #     src.save(project_name, target.get(project_name, label))
    return non_synchronizable
