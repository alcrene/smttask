"""
Provides a single method, `describe`, for producing a unique and reproducible
description of a variable. Similar `repr`, but the result may not be a string.

This module is essentially one big if-else statement.
"""
from warnings import warn
import scipy as sp
import scipy.stats
import scipy.stats._multivariate as _mv
from numbers import Number

from parameters import ParameterSet
from sumatra.datastore.filesystem import DataFile

from .base import PlainArg, File, TaskBase

dist_warning = """Task was not tested on inputs of type {}.
Descriptions of distribution tasks need to be
special-cased because they simply include the memory address; the
returned description is this not reproducible.
""".replace('\n', ' ')

def describe(v):
    if isinstance(v, PlainArg):
        return str(v)
    elif isinstance(v, (tuple, list)):
        return type(v)(describe(u) for u in v)
    elif isinstance(v, ParameterSet):
        # TODO: We want to use recursion through ParameterSets, but where ?
        return v
    elif isinstance(v, TaskBase):
        return v.desc
    elif isinstance(v, type):
        s = repr(v)
        if '<locals>' in s:
            warn(f"Type {s} is dynamically generated and thus not reproducible.")
        return s
    elif isinstance(v, File):
        return v.desc()
    elif isinstance(v, DataFile):
        return File.desc(v.full_path)

    # scipy.stats Distribution types
    elif isinstance(v,
        (_mv.multi_rv_generic, _mv.multi_rv_frozen)):
        if isinstance(v, _mv.multivariate_normal_gen):
            return "multivariate_normal"
        elif isinstance(v, _mv.multivariate_normal_frozen):
            return f"multivariate_normal(mean={v.mean()}, cov={v.cov()})"
        else:
            warn(dist_warning.format(type(v)))
            return repr(v)
    elif isinstance(v, _mv.multi_rv_frozen):
        if isinstance(v, _mv.multivariate_normal_gen):
            return f"multivariate_normal)"
        else:
            warn(dist_warning.format(type(v)))
            return repr(v)

    else:
        warn("Task was not tested on inputs of type {}. "
             "Please make sure task digests are unique "
             "and reproducible.".format(type(v)))
        return repr(v)
