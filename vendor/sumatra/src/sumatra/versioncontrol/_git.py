"""
Defines the Sumatra version control interface for Git.

Classes
-------

GitWorkingCopy
GitRepository


:copyright: Copyright 2006-2015 by the Sumatra team, see doc/authors.txt
:license: BSD 2-clause, see LICENSE for details.
"""

import logging
import git
import os
import shutil
import tempfile
from distutils.version import LooseVersion
from configparser import NoSectionError, NoOptionError
try:
    from git.errors import InvalidGitRepositoryError, NoSuchPathError
except:
    from git.exc import InvalidGitRepositoryError, NoSuchPathError
from .base import Repository, WorkingCopy, VersionControlError
from ..core import component


logger = logging.getLogger("Sumatra")


def check_version():
    if not hasattr(git, "Repo"):
        raise VersionControlError(
            "GitPython not installed. There is a 'git' package, but it is not "
            "GitPython (https://pypi.python.org/pypi/GitPython/)")
    minimum_version = '0.3.5'
    if LooseVersion(git.__version__) < LooseVersion(minimum_version):
        raise VersionControlError(
            "Your Git Python binding is too old. You require at least "
            "version {0}. You can install the latest version e.g. via "
            "'pip install -U gitpython'".format(minimum_version))


def findrepo(path):
    check_version()
    try:
        repo = git.Repo(path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        return
    else:
        return os.path.dirname(repo.git_dir)


############## Additions for supporting dirty directories ####################
#                                                                            #
#    Dirty directories are directories which don’t cause `is_dirty` to       #
#    return True, even when they have uncommitted changes.                   #
#    This is meant to support including things like Jupyter Notebooks        #
#    or additional notes in the same directory as the project code.          #
#    Obviously users must then take care that they don’t put                 #
#    reproducibility-critical code in these directories.                     #
#                                                                            #
##############################################################################

from pathlib import Path
from typing import Any
import git
from git.repo.base import defenc, finalize_process

def get_dirty_files(repo, ignored_paths: list[Path], *args):
    """
    Returns the result of ``repo.git.diff(*args)``, excluding any files
    which are in the `ignored_path`.
    """
    lines = repo.git.diff(*args).split("\n")
    removelst = []
    for line in lines:
        # See the git-diff man page (section RAW OUTPUT FORMAT) for the return format
        diffinfo = line.split()
        if len(diffinfo) == 6:
            src = Path(diffinfo[5])
            removelst.append(any(src.is_relative_to(path) for path in ignored_paths))
        elif len(diffinfo) == 7:
            # Some diffs involve moving a file. We only ignore the line if BOTH the source and destination paths are under an ignored path
            src, dst = Path(diffinfo[5]), Path(diffinfo[6])
            removelst.append(any(src.is_relative_to(path) for path in ignored_paths)
                             and any(dst.is_relative_to(path) for path in ignored_paths))
    return "\n".join(line for line, remove in zip(lines, removelst) if not remove)


# Copied from GitPython.repo.base:_get_untracked_files
# CHANGED: The line with `src.is_relative_to`
def get_untracked_files(repo, ignored_paths: list[Path], *args: Any, **kwargs: Any) -> list[str]:
    # make sure we get all files, not only untracked directories
    proc = repo.git.status(*args, porcelain=True, untracked_files=True, as_process=True, **kwargs)
    # Untracked files prefix in porcelain mode
    prefix = "?? "
    untracked_files = []
    for line in proc.stdout:
        line = line.decode(defenc)
        if not line.startswith(prefix):
            continue
        filename = line[len(prefix) :].rstrip("\n")
        # Special characters are escaped
        if filename[0] == filename[-1] == '"':
            filename = filename[1:-1]
            # WHATEVER ... it's a mess, but works for me
            filename = filename.encode("ascii").decode("unicode_escape").encode("latin1").decode(defenc)
        # The two lines below are what we add to ignore certain paths
        src = Path(filename)
        if any(src.is_relative_to(path) for path in ignored_paths):
            continue
        untracked_files.append(filename)
    finalize_process(proc)
    return untracked_files


# Copied from GitPython.repo.base:is_dirty
# CHANGED: Use `get_dirty_file` instead of `repo.git.diff(*args)`
# CHANGED: Use our patched version of `get_untracked_files`
def is_dirty(
    self,
    index: bool = True,
    working_tree: bool = True,
    untracked_files: bool = False,
    submodules: bool = True,
    path: git.PathLike|None = None,
    ignored_paths = ()
) -> bool:
    """
    :return:
        ``True``, the repository is considered dirty. By default it will react
        like a git-status without untracked files, hence it is dirty if the
        index or the working copy have changes."""
    if self._bare:
        # Bare repositories with no associated working directory are
        # always considered to be clean.
        return False

    ignored_paths = [Path(path) for path in ignored_paths]

    default_args = ["--abbrev=40", "--full-index", "--raw"]
    if not submodules:
        default_args.append("--ignore-submodules")
    if path:
        default_args.extend(["--", str(path)])
    if index:
        # diff index against HEAD
        if git.osp.isfile(self.index.path) and len(get_dirty_files(self, ignored_paths, "--cached", *default_args)):
            return True
    # END index handling
    if working_tree:
        # diff index against working tree
        if len(get_dirty_files(self, ignored_paths, *default_args)):
            return True
    # END working tree handling
    if untracked_files:
        if len(get_untracked_files(self, ignored_paths, path, ignore_submodules=not submodules)):
            return True
    # END untracked files
    return False


############ End additions for supporting dirty directories ##################

@component
class GitWorkingCopy(WorkingCopy):
    """
    An object which allows various operations on a Git working copy.
    """
    name = "git"

    def __init__(self, path=None):
        check_version()
        WorkingCopy.__init__(self, path)
        self.path = findrepo(self.path)
        self.repository = GitRepository(self.path)

    @property
    def exists(self):
        return bool(self.path and findrepo(self.path))

    def current_version(self):
        head = self.repository._repository.head
        try:
            return head.commit.hexsha
        except AttributeError:
            return head.commit.sha

    def use_version(self, version):
        logger.debug("Using git version: %s" % version)
        # if version != 'master':      # AR: I don’t see why excluding the 'master' branch from checking is a good idea; I’ve been using SumatraTask with a repo with no 'master' branch and have not run into issues.
        assert not self.has_changed()  #     For reference, the commit where the guard against 'master' was added was # 2e42f6c
        g = git.Git(self.path)         # NB: We call has_changed() with no ignored dirs: Since we will be changing revision, we can’t allow any exceptions
        g.checkout(version)

    def use_latest_version(self):
        self.use_version('master')  # note that we are assuming all code is in the 'master' branch

    def status(self):
        raise NotImplementedError()

    def has_changed(self, ignored_paths=()):
        # return self.repository._repository.is_dirty()  # Version which does not support dirty directories
        return is_dirty(self.repository._repository, ignored_paths=ignored_paths)

    def diff(self):
        """Difference between working copy and repository."""
        g = git.Git(self.path)
        return g.diff('HEAD', color='never')

    def reset(self):
        """Resets all uncommitted changes since the commit. Destructive, be
        careful with use"""
        g = git.Git(self.path)
        g.reset('HEAD', '--hard')

    def patch(self, diff):
        """Resets all uncommitted changes since the commit. Destructive, be
        careful with use"""
        assert not self.has_changed(), "Cannot patch dirty working copy"
        # Create temp patch file
        if diff[-1] != '\n':
            diff = diff + '\n'
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(diff)
            temp_file_name = temp_file.name
        try:
            g = git.Git(self.path)
            g.apply(temp_file_name)
        finally:
            os.remove(temp_file_name)

    def content(self, digest, filename):
        """Get the file content from repository."""
        repo = git.Repo(self.path)
        for _,blob in repo.index.iter_blobs():
            if blob.name == os.path.basename(filename):
                file_content = repo.git.show('%s:%s' %(digest, blob.path))
                return file_content
        return 'File content not found.'

    def contains(self, path):
        """Does the repository contain the file with the given path?"""
        return path in self.repository._repository.git.ls_files().split()

    def get_username(self):
        config = self.repository._repository.config_reader()
        try:
            username, email = (config.get('user', 'name'), config.get('user', 'email'))
        except (NoSectionError, NoOptionError):
            return ""
        return "%s <%s>" % (username, email)


def move_contents(src, dst):
    for file in os.listdir(src):
        src_path = os.path.join(src, file)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(dst, file))
        else:
            shutil.copy2(src_path, dst)
    shutil.rmtree(src)


@component
class GitRepository(Repository):
    name = "git"
    use_version_cmd = "git checkout"
    apply_patch_cmd = "git apply"

    def __init__(self, url, upstream=None):
        check_version()
        Repository.__init__(self, url, upstream)
        self.__repository = None
        self.upstream = self.upstream or self._get_upstream()

    @property
    def exists(self):
        try:
            self._repository
        except VersionControlError:
            pass
        return bool(self.__repository)

    @property
    def _repository(self):
        if self.__repository is None:
            try:
                self.__repository = git.Repo(self.url)
            except (InvalidGitRepositoryError, NoSuchPathError) as err:
                raise VersionControlError("Cannot access Git repository at %s: %s" % (self.url, err))
        return self.__repository

    def checkout(self, path="."):
        """Clone a repository."""
        path = os.path.abspath(path)
        g = git.Git(path)
        if self.url == path:
            # already have a repository in the working directory
            pass
        else:
            tmpdir = os.path.join(path, "tmp_git")
            g.clone(self.url, tmpdir)
            move_contents(tmpdir, path)
        self.__repository = git.Repo(path)

    def get_working_copy(self, path=None):
        path = self.url if path is None else path
        return GitWorkingCopy(path)

    def _get_upstream(self):
        if self.exists:
            config = self._repository.config_reader()
            if config.has_option('remote "origin"', 'url'):
                return config.get('remote "origin"', 'url')
