"""
Defines the Sumatra version control interface for Bazaar.

Classes
-------

BazaarWorkingCopy
BazaarRepository


:copyright: Copyright 2006-2015 by the Sumatra team, see doc/authors.txt
:license: BSD 2-clause, see LICENSE for details.
"""
from builtins import str

from bzrlib.branch import Branch
from bzrlib.workingtree import WorkingTree
from bzrlib import diff
from bzrlib.errors import NotBranchError

import os
try:
    from StringIO import StringIO  # bazaar does not handle unicode
except ImportError:  # Python3
    from io import StringIO

from .base import VersionControlError
from .base import Repository, WorkingCopy
from ..core import component


@component
class BazaarWorkingCopy(WorkingCopy):
    name = "bazaar"

    def __init__(self, path=None):
        WorkingCopy.__init__(self, path)
        try:
            self.workingtree = WorkingTree.open(self.path)
        except NotBranchError:
            pass
        else:
            self.repository = BazaarRepository(self.workingtree.branch.user_url)
            # self.repository.working_copy = self
            self._current_version = self.repository._repository.revno()

    @property
    def exists(self):
        return self.path and os.path.exists(os.path.join(self.path, ".bzr"))

    def _get_revision_tree(self, version):
        version = int(version)
        revision_id = self.workingtree.branch.get_rev_id(version)
        return self.workingtree.branch.repository.revision_tree(revision_id)

    def current_version(self):
        return str(self._current_version)

    def use_version(self, version):
        self.use_latest_version()
        assert not self.has_changed()
        rev_tree = self._get_revision_tree(version)
        self.workingtree.revert(old_tree=rev_tree)
        self._current_version = version

    def use_latest_version(self):
        self.workingtree.update()
        self.workingtree.revert()
        self._current_version = self.repository._repository.revno()

    def status(self):
        current_tree = self._get_revision_tree(self._current_version)
        delta = self.workingtree.changes_from(current_tree, want_unversioned=True, want_unchanged=True)
        modified = set(i[0] for i in delta.modified)
        removed = set(i[0] for i in delta.removed)
        unknown = set(i[0] for i in delta.unversioned)
        added = set(i[0] for i in delta.added)
        clean = set(i[0] for i in delta.unchanged)
        missing = set([])
        return {'modified': modified, 'removed': removed,
                'missing': missing, 'unknown': unknown,
                'added': added, 'clean': clean}

    def has_changed(self):
        return self.workingtree.has_changes()

    def diff(self):
        """Difference between working copy and repository."""
        iostream = StringIO()
        diff.show_diff_trees(self.workingtree.basis_tree(), self.workingtree, iostream)
        # textstream
        return iostream.getvalue()

    def get_username(self):
        config = self.workingtree.branch.get_config()
        return config.username()


@component
class BazaarRepository(Repository):
    name = "bazaar"
    use_version_cmd = "bzr update -r"
    apply_patch_cmd = "bzr patch"

    def __init__(self, url, upstream=None):
        Repository.__init__(self, url, upstream)
        self.url = url
        self.__repository = None
        # use bzrlib.info.gather_location_info to get upstream?

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
                self.__repository = Branch.open(self.url)
            except Exception as err:
                raise VersionControlError("Cannot access Bazaar repository at %s: %s" % (self.url, err))
        return self.__repository

    def checkout(self, path="."):
        """Clone a repository."""
        path = os.path.abspath(path)
        self._repository.create_checkout(path, lightweight=True)
        # self.working_copy = BazaarWorkingCopy(path)

    def get_working_copy(self, path=None):
        return BazaarWorkingCopy(path)
