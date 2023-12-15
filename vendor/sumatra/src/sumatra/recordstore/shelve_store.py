"""
Handles storage of simulation/analysis records based on the Python standard
shelve module.

:copyright: Copyright 2006-2015 by the Sumatra team, see doc/authors.txt
:license: BSD 2-clause, see LICENSE for details.
"""
from builtins import str

import os
import shutil
import shelve
from datetime import datetime
from .base import RecordStore
from ..core import component


@component
class ShelveRecordStore(RecordStore):
    """
    Handles storage of simulation/analysis records based on the Python standard
    :mod:`shelve` module.

    The advantage of this record store is that it has no dependencies. The
    disadvantages are that it allows only local access and does not support
    the *smtweb* interface.
    """

    def __init__(self, shelf_name=".smt/records"):
        shelf_name = os.path.expanduser(shelf_name)
        self._shelf_name = shelf_name
        # Some shelve backends add an extension to the filename, and more than one
        # file may be created. So that the file(s) can be deleted, we need to try
        # to discover the full filename(s).
        dir = os.path.dirname(os.path.abspath(shelf_name))
        initial_dir_contents = set(os.listdir(dir))
        self.shelf = shelve.open(shelf_name)
        self._shelf_files = set(os.listdir(dir)).difference(initial_dir_contents)

    def __del__(self):
        if hasattr(self, "shelf"):
            self.shelf.close()

    def __str__(self):
        return "Record store using the shelve package (database file=%s)" % self._shelf_name

    def __getstate__(self):
        return {'shelf_name': self._shelf_name}

    def __setstate__(self, state):
        self.__init__(**state)

    def list_projects(self):
        return [str(key) for key in self.shelf.keys()]

    def has_project(self, project_name):
        return project_name in self.shelf

    def save(self, project_name, record):
        if project_name in self.shelf:
            records = self.shelf[project_name]
        else:
            records = {}
        records[record.label] = record
        self.shelf[project_name] = records
        self.shelf.sync()  # At least some backends don't immediately write to disk, even when writeback=False

    def get(self, project_name, label):
        return self.shelf[project_name][label]

    def list(self, project_name, tags=None):
        if project_name in self.shelf:
            if tags:
                if not isinstance(tags, list):
                    tags = [tags]
                records = [record for record in self.shelf[project_name].values()
                           if any([tag in record.tags for tag in tags])]
            else:
                records = list(self.shelf[project_name].values())
        else:
            records = []
        return records

    def labels(self, project_name, tags=None):
        if project_name in self.shelf:
            if tags:
                if not isinstance(tags, list):
                    tags = [tags]
                lbls = [label for label, record in self.shelf[project_name].items()
                        if any([tag in record.tags for tag in tags])]
            else:
                lbls = list(self.shelf[project_name].keys())
        else:
            lbls = []
        return lbls

    def delete(self, project_name, label):
        records = self.shelf[project_name]
        records.pop(label)
        self.shelf[project_name] = records

    def delete_by_tag(self, project_name, tag):
        for_deletion = [record for record in self.shelf[project_name].values() if tag in record.tags]
        for record in for_deletion:
            self.delete(project_name, record.label)
        return len(for_deletion)

    def most_recent(self, project_name):
        most_recent = None
        most_recent_timestamp = datetime.min
        for record in self.shelf[project_name].values():
            if record.timestamp > most_recent_timestamp:
                most_recent_timestamp = record.timestamp
                most_recent = record.label
        return most_recent

    def clear(self):
        for path in self._shelf_files:
            os.remove(path)

    @classmethod
    def accepts_uri(cls, uri):
        return os.path.exists(uri) or os.path.exists(uri + ".db") or os.path.splitext(uri)[1] == ".shelf"

    def backup(self):
        """
        Copy the database file
        """
        shutil.copy2(self._shelf_name, self._shelf_name + ".backup")

    def remove(self):
        """
        Delete the database entirely.
        """
        self.backup()
        os.remove(self._shelf_name)
