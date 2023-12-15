"""
The datastore module provides an abstraction layer around data storage,
allowing different methods of storing simulation/analysis results (local
filesystem, remote filesystem, database, etc.) to provide a common interface.

Currently, only local filesystem storage is supported.

Classes
-------

FileSystemDataStore - provides methods for accessing files stored on a local file
                      system, under a given root directory.
ArchivingFileSystemDataStore - provides methods for accessing files written to
                      a local file system then archived as .tar.gz.
MirroredFileSystemDataStore - provides methods for accessing files written to
                      a local file system then mirrored to a web server

Functions
---------

get_data_store() - return a DataStore object based on a class name and
                   constructor arguments.


:copyright: Copyright 2006-2015 by the Sumatra team, see doc/authors.txt
:license: BSD 2-clause, see LICENSE for details.
"""
from .base import DataStore, DataKey, IGNORE_DIGEST
from .filesystem import FileSystemDataStore
from .archivingfs import ArchivingFileSystemDataStore
from .mirroredfs import MirroredFileSystemDataStore
try:
    from .davfs import DavFsDataStore
except ImportError:
    pass
from ..core import get_registered_components


def get_data_store(type, parameters):
    cls = get_registered_components(DataStore)[type]
    return cls(**parameters)
