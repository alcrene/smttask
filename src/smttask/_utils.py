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

__all__ = ["Singleton", "NO_VALUE", "flatten", "lenient_issubclass", "relative_path",
           "parse_duration_str", "sync_one_way", "clone_conda_project"]

#################
# Singleton (Copied from mackelab_toolbox.utils)

class Singleton(type):
    """Singleton metaclass
    Based on the pattern for numpy._globals._NoValue

    Although singletons are usually an anti-pattern, I've found them useful in
    a few cases, notably for a configuration class storing dynamic attributes
    in the form of properties.
    Before using a singleton, consider these alternate, more
    pythonic options:

      - Enum
        For defining unique constants
      - SimpleNamespace
        For defining configuration containers

    Example
    -------
    >>> from smttask.utils import Singleton
    >>> import sys
    >>>
    >>> class Config(metaclass=Singleton):
    >>>     def num_modules(self):
    >>>         return len(sys.modules)
    >>> config = Config()

    Attempting to create a new instance just returns the original one.
    >>> config2 = Config()
    >>> config is config2  # True
    """
    def __new__(metacls, name, bases, dct):
        cls = super().__new__(metacls, name, bases, dct)
        cls.__instance = None
        # We need to patch __clsnew__ into __new__.
        # 1. Don't overwrite cls.__new__ if one of the parents is already a Singleton
        #    (Otherwise, the child will try to assign two or more different __new__
        #     functions to __super_new)
        if any(isinstance(supercls, metacls) for supercls in cls.mro()[1:]):
            pass
        # 2. Don't overwrite cls.__new__ if it exists
        else:
            for supercls in cls.mro():
                # Ensure we don't assign __clsnew__ to __super_new, other we get
                # infinite recursion
                if supercls.__new__ != metacls.__clsnew__:
                    assert not hasattr(cls, f"_{metacls.__name__}__super_new"), "Multiple Singleton metaclasses have clashed in an unexpected way."
                    cls.__super_new = supercls.__new__
                    break
        cls.__new__ = metacls.__clsnew__
        return cls
    @staticmethod
    def __clsnew__(cls, *args, **kwargs):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = cls.__super_new(cls, *args, **kwargs)
        return cls.__instance


#################
# Constants

class NO_VALUE_CLASS(metaclass=Singleton):
    def __repr__(self):
        return "<value not provided>"
NO_VALUE = NO_VALUE_CLASS()  # Default value in function signatures

# ## Iteration utilities ########################

# %%
import itertools
from collections.abc import Iterable
from typing import Tuple

def flatten(*l, terminate=None):
    """
    Flatten any Python iterable. Taken from https://stackoverflow.com/a/2158532.

    Parameters
    ----------
    l: iterable
        Iterable to flatten
    terminate: tuple of types
        Tuple listing types which should not be expanded. E.g. if l is a nested
        list of dictionaries, specifying `terminate=dict` or `terminate=(dict,)`
        would ensure that the dictionaries are treated as atomic elements and
        not expanded.
        By default terminates on the types listed in `terminating_types`.
    """
    from .config import config  # Placed inside function to prevent import cycles

    # Normalize `terminate` argument
    if terminate is None:
        terminate = config.terminating_types
    if not isinstance(terminate, Iterable):
        terminate = (terminate,)
    else:
        terminate = tuple(terminate)
    # Flatten `l`
    for el in l:
        if (isinstance(el, Iterable)
            and not isinstance(el, terminate)):
            for ell in el:
                yield from flatten(ell, terminate=terminate)
        else:
            yield el

#################
# Misc
from parameters import ParameterSet as BaseParameterSet

def lenient_issubclass(cls: Any, class_or_tuple) -> bool:
    """
    Equivalent to issubclass, but allows first argument to be non-type
    (in which case the result is ``False``).
    """
    # Copied from pydantic.utils
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)

def is_parameterset(value) -> bool:
    # NB: For now we vendorize our patched version of `parameters` (or rather, we use the one from our vendorized `sumatra`)
    #     This is installed _instead_ of the official package.
    return isinstance(value, BaseParameterSet)

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

    Loosely based on: https://gist.github.com/Ayehavgunne/ac6108fa8740c325892b
    """
    duration_string = duration_string.lower()
    durations = {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    duration_multipliers = {'days': 24*60*60, 'hours': 60*60, 'minutes': 60, 'seconds': 1}
    num_str = []     # Accumulate numerical characters to parse
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
    from .vendor.sumatra.sumatra.recordstore import get_record_store
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


#################
# Project management

import os
import distutils.core
import tempfile
import shutil
import subprocess
from typing import Union, Sequence
from pathlib import Path
from configparser import ConfigParser
from pydantic import validate_arguments

# When performing an editable install (`pip install -e`), easy_install is
# still used under the hood. This means that the directory is added to a
# `easy_install.pth` file in the site-packages directory.
# THE PROBLEM: For reasons I don't fully understand, when installing the
# cloned package with `pip install -e`, it can happen that the path to the
# source package is also added / conserved in this file. This can cause
# the source package to shadow the clone, and is very confusing/infuriating
# to debug.
# To ensure that this never happens, after cloning a project, we run this
# function to purge any trace of the source project from easy_install.pth.
remove_src_pkg = '''
import site
from pathlib import Path
from reorder_editable import Editable
pkg = str(Path("{src_pkg}").expanduser().resolve())
for sitedir in site.getsitepackages() + [site.getusersitepackages()]:
    path = Path(sitedir)/"easy-install.pth"
    if path.exists():
        ed = Editable(location=path)
        ed.write_lines([line for line in ed.read_lines()
                        if line != pkg])
'''

run_env_readme = \
"""Some or all of the Python environments in this directory were created by 
cloning a SumatraTask project (typically with `smttask project clone`).
This allows project code to be modified for further development, while keeping 
a clean version for task execution. Clones can also be used to run code using
older versions of a project from a previous git commit.
Except for the project code contained here, cloned environments use the same
packages as the environment they were cloned from.

The following environments were created by cloning:
"""
# List of environments is appended by the cloning function

# TODO: Provide interface to allow:
#  - Creating cloned environments automatically, by specifying only `--clone SHA` to `smttask run`
#  - Recreating sets of environments
#    + Challenge: The commit of the cloned project may not exist in the source
#      project (either because it is a hotfix, or a rebase)
#    + Challenge: Dependency versions may also differ (either a different
#      pip/conda version, or a different commit if the dependency is a repo)
# Possibilities:
#  - Store config templates in a `.cloning` directory of the source project
#  - Clone projects to `.local/smttask/projects`

@validate_arguments
def clone_conda_project(source: Path, dest: Path,
                        source_env: str, dest_env: str,
                        envs_dir: Path="~/.local/smttask/envs",
                        config_files: Sequence[Path]=(),
                        install_ipykernel: bool=True):
    """
    Parameters
    ----------
    source: The project to clone.
    dest: The location into which to clone the project; it may also be empty.
      WIP: If the location already exists, it must be a clone of the source.
    source_env: The name (not path) of the conda environment used in the source
      project.
    dest_env: The name (not path) of the conda environment used in the dest
      project. Each clone requires a different virtual environment, since
      the project code is installed into the environment.
    envs_dir: The directory into which to save the cloned environment.
    config_files: Extra configuration files to copy over; these may be names
      or actual files. May be used for two related purposes:
      - To copy a configuration file which isn't tracked with version control.
      - To update said configuration file with values from a template file.
      LIMITATION: This assumes 1) that configuration files are located at the top
      level of the project; 2) that they are not tracked with version control;
      and 3) that they are compatible with the `configparser` module.
      For each file path ``path/to/conf.cfg``, we do the following:
      - Load ``source/conf.cfg``, if it exists
      - Load ``path/to/conf.cfg``, if it exists, overwriting options
      - Write the result to ``dest/conf.cfg``.
      `FileExistsError` is raised if ``dest/conf.cfg`` already exists.
    install_ipykernel: If True (default), install an IPython kernel making
      the cloned environment available from within Jupyter.
    """
    # UNRESOLVED: If dest exists because it was copied from another machine,
    #  the paths in .git and .smt/project may not point to the right locations
    #  How should we deal with this ?
    source_abs = source.expanduser().resolve()
    dest_abs = dest.expanduser().resolve()
    config_files = [Path(path).expanduser().resolve() for path in config_files]
    
    if "/" in source_env:
        raise ValueError("`source_env` should be an environment name, not a "
                         f"path. Received '{source_env}'.")
    if "/" in dest_env:
        raise ValueError("`dest_env` should be an environment name, not a "
                         f"path. Received '{dest_env}'.")

    target_already_exists = False
    if dest_abs.exists():
        # Allow empty directories
        try:
            next(dest_abs.iterdir())  # Using iterator ensures we don't unnecessarily iterate over all files
        except StopIteration:
            # SUCCESS: directory is empty
            pass
        else:
            target_already_exists = True
            # TODO: Check that `source_abs` is in the upstream
            # # FAIL: directory contains something
            # raise FileExistsError(f"Cannot clone project to location '{dest}': "
            #                       "file or directory already exists.")
    
    # Clone the project
    if target_already_exists:
        logger.info(f"Skipping the cloning of {source} to {dest}: destination already exists.")
    else:
        logger.info(f"Cloning repo {source} to {dest} ...")
        subprocess.run(["git", "clone", source_abs, dest_abs])
    
    # Add symlink for data directory
    data_dir = dest_abs/"data"
    if not data_dir.exists():
        data_dir.symlink_to(source_abs/"data")
    
    # Clone the conda environment
    logger.info(f"Cloning the '{source_env}' environment to '{envs_dir/dest_env}' ...")
    env_dir_abs = envs_dir.expanduser().resolve()
    subprocess.run(["conda", "create",
                    "--prefix", env_dir_abs/dest_env,
                    "--clone", source_env])

    # Switch to the new project directory to avoid clobbering the original project directory
    # (E.g. source installs will clone repositories to ./src/)
    os.chdir(dest_abs)

    # Add the environment directory if it isn't already in conda's list
    # NB: This must be done before any call using `conda run -n <dest_env>`
    #     (In particular, it must be done before installing packages not installed with conda)
    out = subprocess.run(["conda", "config", "--show", "envs_dirs"],
                         capture_output=True, text=True)
    envs = (p.strip(" -") for p in out.stdout.split("\n") if p.startswith("  -"))
    if str(env_dir_abs) not in envs:
        subprocess.run(["conda", "config", "--append", "envs_dirs", env_dir_abs])
        
    # Install packages that weren't installed with conda
    # NB: `pip freeze` exports conda packages as file:///home/conda/feedstock_root/…,
    out = subprocess.run(["conda", "run", "-n", source_env,
                          "pip", "freeze"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr + "\n\n" + out.stdout)
    reqs = out.stdout
    # Get already installed packages
    out = subprocess.run(["conda", "run", "-n", dest_env,
                          "pip", "list", "--format", "freeze",
                          "--exclude", "pip", "--exclude", "distribute", "--exclude", "setuptools", "--exclude", "wheel"],  # `pip freeze` excludes these from its output
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr + "\n\n" + out.stdout)
    pkgs = out.stdout.split("\n")
    def get_req_name(req):
        "Extract a package name from a line exported by `pip freeze`"
        req = req.strip()
        # Remove elements that come before package name
        # + "-e <pkg>…"
        if req.startswith("-e"):
            req = req[2:].strip()
        # Possible markers for the end of a package name:
        # + "<pkg> @ file://…"
        # + "<pkg>==<version>"
        # + "git+https://git@github.com/user/repo@SHA#egg=<pkg name>"
        try:
            # NB: Specs with 'egg=' are dealt with below
            i = next(i for i in (req.find(sep) for sep in (" ", "==")) if i > -1)
        except StopIteration:
            i = None
        name = req[:i]
        if "egg=" in name:
            name = name[name.rfind("egg=")+4:]
        elif i == -1:  # `req` didn't match any of the expected formats
            raise RuntimeError(f"Unexpected requirement format:\n{req}")
        return name.replace("_", "-")  # '_' and '-' are sometimes interchanged, because one is invalid for paths, the other for URLs. Standardize to '-' for matching
    installed_pkgs = [get_req_name(pkg) for pkg in pkgs if pkg.strip()]  # `pkgs` may include empty lines
        # Assumption: If the package was cloned, then its version must match
        
    # Get the name of this package, so we can exclude it from the list of packages to install
    # (It is installed later
    _dist = distutils.core.run_setup("setup.py", stop_after="config")  # DEVNOTE: "config" option is mostly a guess
    this_pkg_name = _dist.get_name()
    
    def special_subs(req):
        """
        Hard coded special cases, to deal with pip/setuptools idiosyncracies:
        When packages are installed with `pip install` (instead of through a
        requirements file), the recorded dependency is not always compatible
        with installing from a requirements file.
        """
        subs = {
            "git+git@github.com:" : "git+https://git@github.com/"
        }
        for k, v in subs.items():
            if k in req:
                req = req.replace(k, v, 1)
                break
        return req
    reqs = [special_subs(req) for req in reqs.split("\n")
            if req.strip() and get_req_name(req) not in installed_pkgs + [this_pkg_name]]
            #if "file:///home/conda" not in req]
    if reqs:
        reqs_formatted = "\n".join((f"  {req}" for req in reqs))
        logger.info("Installing additional packages that weren't installed "
                    f"with conda:\n{reqs_formatted}")
        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            f.write("\n".join(reqs))
            req_file = f.name
        out = subprocess.run(["conda", "run", "-n", dest_env,
                              "pip", "install", "-r", req_file],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"{out.stderr}\n\n{out.stdout}\n\nYou may want to inspect "
                               f"the automatically generated requirements file:\n  {req_file}")
        os.remove(req_file)
    # Make a development install of the cloned repo
    subprocess.run(["conda", "run", "-n", dest_env,
                    "pip", "install", "-e", dest_abs])
    if out.returncode != 0:
        raise RuntimeError(out.stderr + "\n\n" + out.stdout)
                    
    # Add/append to README explaining the purpose of cloned environments
    readme_path = envs_dir.expanduser().resolve()/"README"
    try:
        with open(readme_path, "x") as f:
            f.write(run_env_readme.strip())
            f.write(f"\n - {dest_env}")
            
    except FileExistsError:
        # Before appending, check that README file is what we think it is
        with open(readme_path, "r") as f:
            txt = f.read()
            append = (txt.startswith(run_env_readme.strip())
                      and txt.rsplit("\n", 2)[-2].startswith(" -"))
        if append:
            with open(readme_path.exists(), "a") as f:
                f.write(f"\n - {dest_env}")

    # Install the kernel so this environment can be used with IPython
    if install_ipykernel:
        subprocess.run(["conda", "run", "-n", dest_env,
                        "python", "-m", "ipykernel", "install", "--user",
                        "--name", dest_env, "--display-name", f"Python ({dest_env})"])

    # Add .smt project file to the new project, which points to the new repo but the old data
    if not (dest_abs/".smt").exists():
        (dest_abs/".smt").mkdir()
        shutil.copy(source_abs/".smt/project", dest_abs/".smt/project")
        subprocess.run(["smt", "configure", "--repository", "."],
                       cwd=dest_abs)
                   
    # Copy over any extra config file
    for path in config_files:
        path = path
        cfg = ConfigParser()
        cfg.read(source_abs/path.name)
        cfg.read(path)
        with open(dest_abs/path.name, 'x') as f:
            cfg.write(f)

    # Purge any trace of the source package from the cloned environment
    # (I'm not sure why this happens, but it may be related to copying cloned repos across machines)
    subprocess.run(["conda", "run", "-n", dest_env,
                    "python", "-c", remove_src_pkg.format(src_pkg=source_abs)])


######################## Normalization to bytes ################################
# The code below implements a generic _tobytes function, which tries its       #
# best to convert a wide variety of inputs into a bytes sequence which can     #
# be hashed. This functionality is currently only really used by               #
# `workflows.SeedGenerator` (and there, limited to `_normalize_entropy`);      #
# the rest of smttask only needs support for hashing  str and int.             #
# There is also the possibility that we could reimplement the                  #
# functionality below using Serializable.reduce; this would avoid the need for #
# all the type cases. OTOH, Serializable.reduce needs to allow bi-directional  #
# conversion, whereas here this is not an issue, making the logic here         #
# ultimately simpler and faster.                                               #
######################## ###################### ################################

from enum import Enum
from typing import Any
from collections.abc import Mapping, Sequence, Collection, Iterable
from .hashing import stablehash, stableintdigest

def universal_stablehash(o: Any) -> '_hashlib.HASH':
    """
    Like `stablehash`, but with much more flexibility in the type of `o`.
    The value is passed through `_tobytes` to normalize it to bytes before
    hashing, allowing most values to be used to produce a unique, stable hash.
    See `_tobytes` for details.
    Additional value->bytes rules for different types can be added to the
    `_byte_converters` dictionary.
    """
    return stablehash(_tobytes(o))
def universal_stableintdigest(o: Any, byte_len=4) -> int:
    return stableintdigest(_tobytes(o), byte_len)

# Extra functions to converting values to bytes specialized to specific types
# This is the mechanism to use to add support for types outside your own control
# (i.e. for which it is not possible to add a __bytes__ method).
class TypeDict:
    """
    A dictionary using types as keys. A key will match any of its subclasses;
    if there are multiple possible matches, the earliest in the dictionary
    takes precedence.
    """
    def __getitem__(self, key):
        if not isinstance(key, type):
            raise TypeError("TypeDict keys must be types")
        for k, v in self.items():
            if issubclass(key, k):
                return v
        raise KeyError(f"Type {key} is not a subclass of any of this "
                       "TypeDict's type keys.")

    def __setitem__(self, key, value):
        if not isinstance(key, type):
            raise TypeError("TypeDict keys must be types")
        return super().__setitem__(key, value)

    def __contains__(self, key):
        return (isinstance(key, type) and
                any(issubclass(key, k) for k in self))

_byte_converters = TypeDict()

def _tobytes(o) -> bytes:
    """
    Utility function for converting an object to bytes. This is used to
    generate unique, stable digests for arbitrary sequences of objects:

    1. Different inputs should, with high probability, return different byte
       sequences.
    2. The same inputs should always return the same byte sequence, even when
       executed in a new session (in order to satisfy the 'stable' description).
       Note that this precludes using an object's `id`, which is sometimes
       how `hash` is implemented.
       It also precludes using `hash` or `__hash__`, since that function is
       randomly salted for each new Python session.
    3. It is NOT necessary for the input to be reconstructable from
       the returned bytes.

    ..Note:: To avoid overly complicated byte sequences, the distinction
       guarantee is not preserved across types. So `_tobytes(b"A")`,
       `_tobytes("A")` and `_tobytes(65)` all return `b'A'`.
       So multiple inputs can return the same byte sequence, as long as they
       are unlikely to be used in the same location to mean different things.

    **Supported types**
    - None
    - bytes
    - str
    - int
    - float
    - Enum
    - type
    - dataclasses, as long as their fields are supported types.
      + NOTE: At present we allow hashing both frozen and non-frozen dataclasses
    - Any object implementing a ``__bytes__`` method
    - Mapping
    - Sequence
    - Collection
    - Any object for which `bytes(o)` does not raise an exception

    TODO?: Restrict to immutable objects ?

    Raises
    ------
    TypeError:
        - If `o` is a consumable Iterable.
        - If `o` is of a type for which `_to_bytes` is not implemented.
    """
    # byte converters for specific types
    if o is None:
        # TODO: Would another value more appropriately represent None ? E.g. \x00 ?
        return b""
    elif isinstance(o, bytes):
        return o
    elif isinstance(o, str):
        return o.encode('utf8')
    elif isinstance(o, int):
        l = ((o + (o<0)).bit_length() + 8) // 8  # Based on https://stackoverflow.com/a/54141411
        return o.to_bytes(length=l, byteorder='little', signed=True)
    elif isinstance(o, float):
        return o.hex().encode('utf8')
    elif isinstance(o, Enum):
        return _tobytes(o.value)
    elif isinstance(o, type):
        return _tobytes(f"{o.__module__}.{o.__qualname__}")
    elif is_dataclass(o):
        # DEVNOTE: To restrict this to immutable dataclasses, check `o.__dataclass_params__.frozen`
        return _tobytes(tuple((f.name, getattr(o, f.name)) for f in fields(o)))
    # Generic byte encoders. These methods may not be ideal for each type, or
    # even work at all, so we first check if the type provides a __bytes__ method.
    elif hasattr(o, '__bytes__'):
        return bytes(o)
    elif type(o) in _byte_converters:
        return _byte_converters[type(o)](o)
    elif isinstance(o, Mapping) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(k) + _tobytes(v) for k,v in o.items())
    elif isinstance(o, Sequence) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in o)
    elif isinstance(o, Collection) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in sorted(o))
    elif isinstance(o, Iterable) and not isinstance(o, terminating_types):
        raise ValueError("Cannot compute a stable hash for a consumable Iterable.")
    else:
        try:
            return bytes(o)
        except TypeError:
            # As an ultimate fallback, attempt to use the same decomposition
            # that pickle would
            try:
                state = o.__getstate__()
            except Exception:
                raise TypeError("smttask.hashing._tobytes does not know how "
                                f"to convert values of type {type(o)} to bytes. "
                                "One way to solve this is may be to add a "
                                "`__bytes__` method to that type. If that is "
                                "not possible, you may also add a converter to "
                                "smttask.hashing._byte_converters.")
            else:
                return _tobytes(state)