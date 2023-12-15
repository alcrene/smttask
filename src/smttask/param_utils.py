"""
Utilities for working with parameter sets

Copied from portions of mackelab_toolbox.parameters
"""

from .config import config
from ._utils import is_parameterset

__all__ = ["dfdiff", "ParameterComparison", "digest"]

###########################
# Comparing ParameterSets
###########################

#    Find differences between two or more sets.
#    Differences are typically returned in a Pandas DataFrame, allowing
#    simple visualization and further inspection.
#    Hierarchical sets can be compared up to a user-specified depth.
#

# Two main functions, which should be merged at some point.
#   + ParameterComparison
#         Actually a class.
#         Only method which can take records, or more than two sets
#         Can be converted to a Pandas DataFrame
#   + dfdiff
#         A bit more straightforward, but less flexible.

from collections import namedtuple
from typing import Optional, Sequence

def dfdiff(pset1: config.ParameterSet, pset2: config.ParameterSet,
           name1: str='pset 1', name2: str='pset 2',
           ignore: Optional[Sequence[str]]=None):
    """
    Uses `pset1`'s `diff` method to compare `pset1` and `pset2`.
    Falls back to `pset2`'s method if `pset1` doesn't have one, and to
    `NTParameterSet.diff` if neither pset does.

    Parameters
    ----------
    pset1, pset2: ParameterSet | dict
        The parameter sets to compare.
        Dicts are also supported, but not recommended because they don't
        provide a `diff` method.
    name1, name2: str (default: 'pset 1', 'pset 2')
        Labels for each parameter set. These are the column labels in the
        return DataFrame.
    ignore: Tuple[str] | List[str]
        List of parameter names to exclude from the comparison, even if they
        differ.

        .. note:: Names not evaluated hierarchically, but compared at each
           level. So ``['timestamp']`` will ignore any parameter named
           'timestamp' at any level, and ``['task.timestamp']`` will not ignore
           anything (because '.' is invalid in a key name of a hierarchical
           parameter set).
    """

    if not hasattr(config.ParameterSet, 'diff'):
        raise RuntimeError(
            "`dfdiff` requires Sumatra's NTParameterSet. Make sure "
            "you can execute `import sumatra.parameters`.")

    pset1 = params_to_lists(pset1)  # params_to_lists returns same type as pset
    pset2 = params_to_lists(pset2)
    if not hasattr(pset1, 'diff'):
        if hasattr(pset2, 'diff'):
            # Use pset2's diff method, since pset1 doesn't have one
            pset1, pset2 = pset2, pset1
        else:
            # Cast to a type which has a diff method
            pset1 = config.ParameterSet(pset1)
    diff = pset1.diff(pset2)
    df = psets_to_dataframe(**{name1:diff[0], name2:diff[1]})
    if ignore:
        df.loc[[idx for idx in df.index
                if not any(s == d for d in idx for s in ignore)]]
    return df

def psets_to_dataframe(*args, **psets):
    from itertools import count, chain

    # Add unlabelled parameter sets to psets with default names
    if len(args) > 0:
        psets = psets.copy()
        newpsets = {'pset'+str(i): p for i, p in zip(count(1), args)}
        if len(set(newpsets).intersection(psets)) != 0:
            raise ValueError("Don't use 'pset' to label a parameter set, or pass "
                             "all parameter sets as keywords to avoid label "
                             "clashes.")
        psets.update(newpsets)

    d = {lbl: {tuple(k.split('.')) : v for k, v in config.ParameterSet(pi).flatten().items()}
     for lbl, pi in psets.items()}
    # Make sure all keys have same length, otherwise we get all NaNs (ind -> 'inner dict')
    indlens = (len(ik) for ik in
               chain.from_iterable(ind.keys() for ind in d.values()))
    klen = max(chain([0], indlens))
    if klen == 0:
        d = {}
    else:
        d = {ok: {k + ('–',)*(klen-len(k)): v for k, v in od.items()}
             for ok, od in d.items()}

    return pd.DataFrame(d)

ParamRec = namedtuple('ParamRec', ['label', 'parameters'])
    # Data structure for associating a name to a parameter set

class ParameterComparison:
    """
    Example
    -------
    >>>  testparams = ParameterSet("path/to/file")
    >>>  records = smttask.view.RecordStoreView(project='project').list
    >>>  cmp = ParameterComparison([testparams] + records, ['test params'])
    >>>  cmp.dataframe(depth=3)
    """
    def __init__(self, params, labels=None):
        """
        Parameters
        ----------
        params: iterable of ParameterSet's or Sumatra records
        labels: list or tuple of strings
            Names for the elements of `params` which are parameter sets. Records
            don't need a specified name since we use their label.
        """
        self.records = make_paramrecs(params, labels)
        self.comparison = structure_keys(get_differing_keys(self.records))

    def _get_colnames(self, depth=1, param_summary=None):
        if param_summary is None:
            param_summary = self.comparison
        if depth == 1:
            colnames = list(param_summary.keys())
        else:
            nonnested_colnames = [key for key, subkeys in param_summary.items() if subkeys is None]
            nested_colnames = itertools.chain(*[ [key+"."+colname for colname in self._get_colnames(depth-1, param_summary[key])]
                                                 for key, subkeys in param_summary.items() if subkeys is not None])
            colnames = nonnested_colnames + list(nested_colnames)
        return colnames

    def _display_param(self, record, name):
        if self.comparison[name] is None:
            try:
                display_value = record.parameters[name]
            except KeyError:
                display_value = "–"
        else:
            display_value = "<+>"
        return display_value

    def dataframe(self, depth=1, maxcols=50):
        """
        Remark
        ------
        Changes the value of pandas.options.display.max_columns
        (to ensure all parameter keys are shown)
        """
        colnames = self._get_colnames(depth)
        columns = [ [self._display_param(rec, name) for name in colnames]
                    for rec in self.records ]
        pd.options.display.max_columns = max(len(colnames), maxcols)
        return pd.DataFrame(data=columns,
             index=[rec.label for rec in self.records],
             columns=colnames)

def make_paramrecs(params, labels=None):
    """
    Parameters
    ----------
    params: iterable of ParameterSets or sumatra Records
    labels: list or tuple of strings
        Names for the elements of `params` which are parameter sets. Records
        don't need a specified name since we use the label.
    """
    if labels is None:
        labels = []
    i = 0
    recs = []
    for p in tqdm(params, leave=False):
        if not isinstance(p, dict) and hasattr(p, 'parameters'):
            if isinstance(p.parameters, str):
                # Parameters were simply stored as a string
                params = config.ParameterSet(p.parameters)
            else:
                assert is_parameterset(p.parameters)
                params = p.parameters
            recs.append(ParamRec(p.label, params))
        elif is_parameterset(params):
            raise TypeError("Wrap ParameterSets in a list so they are passed "
                            "as a single argument.")
        else:
            if isinstance(p, (str,dict)):
                p = config.ParameterSet(p)
            if is_parameterset(p):
                raise TypeError("Each element of `params` must be a "
                                f"ParameterSet. Received '{type(p)}'.")
            if i >= len(labels):
                raise ValueError("A label must be given for each element of "
                                 "`params` which is not a Sumatra record.")
            recs.append(ParamRec(labels[i], p))
            i += 1
    assert(i == len(labels)) # Check that we used all names
    return recs

def _isndarray(a):
    """
    Test if a is an Numpy array, without having to import Numpy.
    Will also return True if `a` looks like an array. (Specifically,
    if it implements the `all` and `any` methods.)
    """
    # A selection of attributes sufficiently specific
    # for us to treat `a` as a Numpy array.
    array_attrs = {'all', 'any'}
    return array_attrs.issubset(dir(a))

def _dict_diff(a, b):
    """
    Ported from sumatra.parameters to allow comparing arrays and lists.
    """
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    intersection = a_keys.intersection(b_keys)
    difference1 = a_keys.difference(b_keys)
    difference2 = b_keys.difference(a_keys)
    result1 = dict([(key, a[key]) for key in difference1])
    result2 = dict([(key, b[key]) for key in difference2])
    # Now need to check values for intersection....
    for item in intersection:
        if isinstance(a[item], dict):
            if not isinstance(b[item], dict):
                result1[item] = a[item]
                result2[item] = b[item]
            else:
                d1, d2 = _dict_diff(a[item], b[item])
                if d1:
                    result1[item] = d1
                if d2:
                    result2[item] = d2
        else:
            if _isndarray(a[item]) or _isndarray(b[item]):
                equal = (a[item] == b[item]).all()
            elif isinstance(a[item], Iterable):
                equal = ( isinstance(b[item], Iterable)
                          and all(x == y for x, y in zip(a[item], b[item]))
                          and len(list(a[item])) == len(list(b[item])) )
                    # len() == len() tests for different length generators
            else:
                equal = (a[item] == b[item])
            if not equal:
                result1[item] = a[item]
                result2[item] = b[item]
    if len(result1) + len(result2) == 0:
        assert a == b, "Error in _dict_diff()"
    return result1, result2

def get_differing_keys(records):
    """
    Parameters
    ----------
    records: ParamRec instances
    """
    assert(isinstance(records, Iterable))
    assert(all(isinstance(rec, ParamRec) for rec in records))

    diffpairs = {(i,j) : _param_diff(records[i].parameters, records[j].parameters,
                                     records[i].label, records[j].label)
                  for i, j in itertools.combinations(range(len(records)), 2)}
    def get_keys(diff):
        assert(bool(hasattr(diff, 'key')) != bool(hasattr(diff, 'keys'))) # xor
        if hasattr(diff, 'key'):
            return set([diff.key])
        elif hasattr(diff, 'keys'):
            return set(diff.keys)
    differing_keys = set().union( *[ get_keys(diff)
                                     for diffpair in diffpairs.values() # all difference types between two pairs
                                     for difftype in diffpair.values()  # keys, nesting, type, value
                                     for diff in difftype ] )

    return differing_keys

def structure_keys(keys):
    #keys = sorted(keys)
    roots = set([key.split('.')[0] for key in keys])
    tree = {root: [] for root in roots}
    for root in roots:
        for key in keys:
            if '.' in key and key.startswith(root):
                tree[root].append('.'.join(key.split(".")[1:]))

    return config.ParameterSet(
        {root: None if subkeys == [] else structure_keys(subkeys)
         for root, subkeys in tree.items()})

KeyDiff = namedtuple('KeyDiff', ['name1', 'name2', 'keys'])
NestingDiff = namedtuple('NestingDiff', ['key', 'name'])
TypeDiff = namedtuple('TypeDiff', ['key', 'name1', 'name2'])
ValueDiff = namedtuple('ValueDiff', ['key', 'name1', 'name2'])
diff_types = {'keys': KeyDiff,
              'nesting': NestingDiff,
              'type': TypeDiff,
              'value': ValueDiff}
def _param_diff(params1, params2, name1="", name2=""):
    diffs = {key: set() for key in diff_types.keys()}
    keys1, keys2 = set(params1.keys()), set(params2.keys())
    if keys1 != keys2:
        diffs['keys'].add( KeyDiff(name1, name2, frozenset(keys1.symmetric_difference(keys2))) )
    def diff_vals(val1, val2):
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            return (val1 != val2).any()
        else:
            return val1 != val2
    for key in keys1.intersection(keys2):
        if is_parameterset(params1[key]):
            if not is_parameterset(params2[key]):
                diffs['nesting'].add((key, name1))
            else:
                for diffkey, diffval in _param_diff(params1[key], params2[key],
                                                    name1 + "." + key, name2 + "." + key).items():
                    # Prepend key to all nested values
                    if hasattr(diff_types[diffkey], 'key'):
                        diffval = {val._replace(key = key+"."+val.key) for val in diffval}
                    if hasattr(diff_types[diffkey], 'keys') and len(diffval) > 0:
                        iter_type = type(next(iter(diffval)).keys)  # Assumes all key iterables have same type
                        diffval = {val._replace(keys = iter_type(key+"."+valkey for valkey in val.keys))
                                   for val in diffval}
                    # Update differences dictionary with the nested differences
                    diffs[diffkey].update(diffval)
        elif is_parameterset(params2[key]):
            diffs['nesting'].add(NestingDiff(key, name2))
        elif type(params1[key]) != type(params2[key]):
            diffs['type'].add(TypeDiff(key, name1, name2))
        elif diff_vals(params1[key], params2[key]):
            diffs['value'].add(ValueDiff(key, name1, name2))

    return diffs


###########################
# Using ParameterSets as consistent identifiers
# (i.e. making file names from parameters)
###########################

from collections import OrderedDict
import numpy as np

# We use the string representation of arrays to compute the hash,
# so we need to make sure it's standardized. The values below
# are the NumPy defaults.
_digest_printoptions = {
    'precision': 8,
    'edgeitems': 3,
    'formatter': None,
    'infstr': 'inf',
    'linewidth': 75,
    'nanstr': 'nan',
    'suppress': False,
    'threshold': 1000,
    'floatmode': 'maxprec',
    'sign': '-',
    'legacy': False}
_new_printoptions = {'1.14': ['floatmode', 'sign', 'legacy']}
    # Lists of printoptions keywords, keyed by the NumPy version where they were introduced
    # This allows removing keywords when using an older version
_remove_whitespace_for_digests = True
_type_compress = OrderedDict((
    (np.floating, np.float64),
    (np.integer, np.int64)
))
    # When normalizing types (currently only in `digest`), numpy types
    # matching the key (left) are converted to the type on the right.
    # First matching entry is used, so more specific types should come first.

def normalize_type(value):
    """
    Apply the type conversions given by `_type_compress`. This reduces the space
    of possible types, helping make digests more consistent.
    """
    if isinstance(value, np.ndarray):
        for cmp_dtype, conv_dtype in _type_compress.items():
            if np.issubdtype(value.dtype, cmp_dtype):
                return value.astype(conv_dtype)
    # No conversion match was found: return value unchanged
    return value

def digest(params, suffix=None, convert_to_arrays=True):
    """
    Generate a unique name by hashing a parameter file.

    ..Note:
    Parameters whose names start with '_' are ignored. This means that two
    parameter sets A, B with `A['_x'] == 1` and `B['_x'] == 2` will be
    assigned the same name.

    ..Debugging:
    If two parameter sets should give the same digest but don't, check the
    value of `debug_store['digest']['hashed_string']`. This module-wide
    stores the most recently hashed string representation of a parameter set.
    Digests will be the same if and only if these string represenations
    are the same.

    Parameters
    ----------
    params: ParameterSet or iterable of ParameterSets
        Digest will be based on these parameters. Parameter keys starting with
        an underscore are ignored.
        Can also be give a list of parameter sets; the name in this case will
        depend on the order of the list.
        Could also arbitrarily nested lists of parameter sets.

    suffix: str or None (default none)
        If not None, an underscore ('_') and then the value of `suffix` are
        appended to the calculated digest

    convert_to_arrays: bool (default True)
        If true, the parameters are normalized by using the result of
        `params_to_arrays(params)` to calculate the digest.
    """
    if isinstance(params, dict) and not isinstance(params, config.ParameterSet):
        # TODO: Any reason this implicit conversion should throw a warning ?
        params = config.ParameterSet(params)
    if (not is_parameterset(params) and not instance(params, str)
        and isinstance(params, Iterable)):
        # Get a hash for each ParameterSet, and rehash them together
        basenames = [p.digest() if hasattr(p, 'digest')
                     else digest(p, None, convert_to_arrays)
                     for p in params]
        basename = hashlib.sha1(bytes(''.join(basenames), 'utf-8')).hexdigest()
        basename += '_'

    else:
        if not is_parameterset(params):
            logger.warning("'digest()' requires an instance of ParameterSet. "
                           "Performing an implicit conversion.")
            params = config.ParameterSet(params)
        if convert_to_arrays:
            # Standardize the parameters by converting them all to arrays
            # -> `[1, 0]` and `np.array([1, 0])` should give same file name
            params = params_to_arrays(params)
        if params == '':
            basename = ""
        else:
            if (np.__version__ < '1.14'
                and _digest_printoptions['legacy'] != '1.13'):
                logger.warning(
                    "You are running Numpy v{}. Numpy's string representation "
                    "algorithm was changed in v.1.14, meaning that computed "
                    "digests will not be consistent with those computed on "
                    "more up-to-date systems. To ensure consistent digests, "
                    "either update to 1.14, or set  `smttask.param_utils._digest_printoptions['legacy']` "
                    "to '1.13'. Note that setting the 'legacy' option may not "
                    "work in all cases.".format(np.__version__))
            # Remove printoptions that are not supported in this Numpy version
            printoptions = _digest_printoptions.copy()
            for version in (v for v in _new_printoptions if v > np.__version__):
                for key in _new_printoptions[version]:
                    del printoptions[key]

            # Standardize the numpy print options, which affect output from str()
            stored_printoptions = np.get_printoptions()
            np.set_printoptions(**printoptions)

            # HACK Force dereferencing of '->' in my ParameterSet
            #      Should be innocuous for normal ParameterSets
            def dereference(paramset):
                for key in paramset:
                    paramset[key] = paramset[key]
                    if is_parameterset(paramset[key]):
                        dereference(paramset[key])
            dereference(params)

            # We need a sorted dictionary of parameters, so that the hash is consistent
            # Also remove keys starting with '_'
            # Types need to be normalized, because if we save values as Python
            # plain types, this can throw away some Numpy type information.
            # To make sure digests are consistent when we read the parameters
            # back, we use one type per Python type (1 for floats, 1 for ints)
            flat_params = params.flatten()
                # flatten avoids need to sort recursively
            sorted_params = OrderedDict()
            for key in sorted(flat_params):
                if key[0] != '_':
                    val = flat_params[key]
                    if hasattr(val, 'digest'):
                        sorted_params[key] = \
                            val.digest() if hasattr(val.digest, '__call__') \
                            else val.digest
                    else:
                        sorted_params[key] = normalize_type(val)

            # Now that the parameterset is standardized, hash its string repr
            s = repr(sorted_params)
            if _remove_whitespace_for_digests:
                # Removing whitespace makes the result more reliable; e.g. between
                # v1.13 and v1.14 Numpy changed the amount of spaces between some elements
                s = ''.join(s.split())
            debug_store['digest'] = {'hashed_string': s}
            basename = hashlib.sha1(bytes(s, 'utf-8')).hexdigest()
            basename += '_'
            # Reset the saved print options
            np.set_printoptions(**stored_printoptions)
    if isinstance(suffix, str):
        suffix = suffix.lstrip('_')
    if suffix is None or suffix == "":
        assert(len(basename) > 1 and basename[-1] == '_')
        return basename[:-1] # Remove underscore
    elif isinstance(suffix, str):
        return basename + suffix
    elif isinstance(suffix, Iterable):
        assert(len(suffix) > 0)
        return basename + '_'.join([str(s) for s in suffix])
    else:
        return basename + str(suffix)

def params_to_arrays(params):
    """
    Recursively apply `np.array()` to all values in a ParameterSet. This allows
    arrays to be specified in files as nested lists, which are more readable.
    Also converts dictionaries to parameter sets.
    """
    # TODO: Don't erase _url attribute
    ParamType = type(params)
        # Allows to work with types derived from ParameterSet, for example Sumatra's
        # NTParameterSet
    for name, val in params.items():
        if isinstance(val) or is_parameterset(val):
            params[name] = params_to_arrays(val)
        elif (not isinstance(val, str)
            and isinstance(val, Iterable)
            and all(isinstance(v, Number) for v in flatten(val))):
                # The last condition leaves objects like ('lin', 0, 1) as-is;
                # otherwise they would be casted to a single type
            params[name] = np.array(val)
    return ParamType(params)

def params_to_lists(params):
    """
    Recursively call `tolist()` on all NumPy array values in a ParameterSet.
    This allows exporting arrays as nested lists, which are more readable
    and properly imported (array string representations drop the comma
    separating elements, which causes import to fail).
    """
    # TODO: Don't erase _url attribute
    ParamType = type(params)
        # Allows to work with types derived from ParameterSet, for example Sumatra's
        # NTParameterSet
    for name, val in params.items():
        if isinstance(val, dict) or is_parameterset(val):
            params[name] = params_to_lists(val)
        elif isinstance(val, np.ndarray):
            params[name] = val.tolist()
    return ParamType(params)
params_to_nonarrays = params_to_lists
