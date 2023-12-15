import pytest
import pickle
import json
import numpy as np
from dataclasses import dataclass, fields, asdict, replace
from typing import Tuple
from scipy import stats

import scityping
import scityping.scipy  # Load serializers for scipy distributions
from smttask.hashing import stablehexdigest
from smttask.workflows import ParamColl, expand, SeedGenerator
# from smttask.workflows import KW_ONLY  # Dataclass field types

scityping.config.safe_packages.add(__name__)

class RerunThisTest(RuntimeError):
    def __init__(self, msg=None):
        if not msg:
            msg = ("This test verifies consistency across multiple runs: "
                   "it needs to run again before succeeding.\n"
                   "You likely want to rerun it with `pytest --lf`")
        super().__init__(msg)

# Test Parameter collections
# (Default param values are required for tests to run on Python <3.10)
@dataclass(frozen=True)
class DataParamset(ParamColl):
    L: int    = 100
    λ: float  = 1
    σ: float  = 1
    δy: float = 0.1

@dataclass(frozen=True)
class ModelParamset(ParamColl):
    λ: float  = 1
    σ: float  = 1

# Test plain dataclasses
@dataclass(frozen=True)
class ModelDataclass:
    λ: float  = 1
    σ: float  = 1

def test_ParamColl():
    data_params = DataParamset(
        L=400,
        λ=1,
        σ=1,
        δy=expand([-1, -0.3, 0, 0.3, 1])
    )
    model_params = ModelParamset(
        λ=expand(np.logspace(-1, 1, 10)),
        σ=expand(np.linspace(0.1, 3, 8))
    )
    model_params_aligned = ModelParamset(
        λ=expand(np.logspace(-1, 1, 10)),
        σ=expand(np.linspace(0.1, 3, 10))
    )

    #
    # Iterating over ParamColl returns the keys
    assert len(list(data_params)) == len(data_params.keys()) == 4
    assert list(data_params) == ["L", "λ", "σ", "δy"]

    # Expanding a list
    assert list(data_params.inner()) == list(data_params.outer())
    assert len(list(data_params.outer())) == len(data_params.δy) == data_params.outer_len == 5

    # Expanding an array + Non-aligned doesn't allow inner() iterator
    assert len(list(model_params.outer())) == 10*8
    with pytest.raises(ValueError):
        next(model_params.inner())

    # Expanding an array + Aligned expanded params allows inner() iterator
    assert len(list(model_params_aligned.inner())) == len(model_params_aligned.λ) == model_params_aligned.inner_len == 10
    assert len(list(model_params_aligned.outer())) == model_params_aligned.outer_len == 10*10

    map_fields = [field.name for field in fields(data_params)
                  if not field.name.startswith("_")
                     and field.name not in {"paramseed"}]
    assert dict(**data_params) == {k: getattr(data_params, k)
                                   for k in map_fields}

    # Slicing inner() and outer() works as advertised
    assert len(list(model_params_aligned.inner(2, 8))) == 6
    assert len(list(model_params_aligned.inner(2, 8, 2))) == 3
    assert len(list(model_params_aligned.outer(5,20)))   == 15
    assert len(list(model_params_aligned.outer(5,20,5))) == 3


    ## Param collections with random generators ##

    # Random params require a seed

    with pytest.raises(TypeError):
        DataParamset(
            L=400,
            λ=expand(stats.norm()),
            σ=1,
            δy=expand([-1, -0.3, 0, 0.3, 1])
        )


    data_params = DataParamset(
        L=400,
        λ=expand(stats.norm()),
        σ=expand([1, 0.2, 0.05]),
        δy=expand([-1, -0.3, 0, 0.3, 1]),
        paramseed=314
    )

    # σ and δy are not aligned: cannot do inner product
    with pytest.raises(ValueError):
        list(data_params.inner())  # `list()` is required to trigger error
    assert len(list(data_params.outer())) == data_params.outer_len == 3*5

    # Now align σ and δy: can do both inner and outer product
    data_params = replace(data_params, σ=expand([10, 0, 1, 0.2, 0.05]))
    assert len(list(data_params.inner())) == data_params.inner_len == 5
    assert len(list(data_params.outer())) == data_params.outer_len == 5*5

    # Randomly generated values are reproducible
    ival1 = tuple(p.λ for p in data_params.inner())
    ival2 = tuple(p.λ for p in data_params.inner())
    assert ival1 == ival2

    oval1 = tuple(p.λ for p in data_params.outer())
    oval2 = tuple(p.λ for p in data_params.outer())
    assert oval1 == oval2

    # Values change if seed changes
    data_params = replace(data_params, paramseed = 628)
    ival3 = tuple(p.λ for p in data_params.inner())
    oval3 = tuple(p.λ for p in data_params.outer())
    assert ival1 != ival3
    assert oval1 != oval3

    # When the only expandable values are random, only `inner` is possible
    data_params = replace(data_params,
                          σ=stats.lognorm(1),       # NB: This is just a scalar value (not expandable)
                          δy = expand(stats.uniform(-1, 1)))
    with pytest.raises(ValueError):
        list(zip(data_params.outer(), range(10)))
    σvals = [p.σ for p, _ in zip(data_params.inner(), range(100))]  # Infinite iterate: use range(100) to truncate
    assert len(σvals) == 100  # All generated values are different

    # # Can set the inner length to convert from infinite to finite collection
    # data_params.inner_len = 10
    # σvals2 = [p.σ for p in data_params.inner()]
    # assert len(σvals2) == 10
    # assert σvals2 == σvals[:10]

def test_ParamColl_reproducible_rng(pytestconfig):
    """
    Test that the seed leads to reproducible parameter sets across runs.
    IMPORTANT: This test needs to be run twice, in *separate* processes.
       The easiest way is to run `pytest` twice in a row.
       To restart the test, delete the file `emdd-test-paramcoll-reproducible.txt`
    """
    model_params = ModelParamset(
        λ=expand(stats.norm()),
        σ=expand(stats.norm()),
        paramseed=314
    )
    s = str([dict(**p) for p in model_params.inner(1)])

    s2 = pytestconfig.cache.get("test-paramcoll-reproducible-rng", None)
    if s2 is None:
        pytestconfig.cache.set("test-paramcoll-reproducible-rng", s)
        raise RerunThisTest()
    elif s != s2:
        # Clear the cache first, so that the next recreates it with the
        # (hopefully fixed) code
        pytestconfig.cache.set("test-paramcoll-reproducible-rng", None)
        # We know this test will fail, but using `assert` produces a more informative message
        assert s == s2, "Random ParamColls are not consistent across processes, "\
                        "even with fixed seed."

def test_ParamColl_reproducible_digests(pytestconfig):
    from tasks import Square_x  # Any task will do
    task = Square_x(x=2)

    fixed_model = ModelParamset(
        λ=3,
        σ=1.1
    )
    stat_model = ModelParamset(
        λ=expand(stats.norm()),
        σ=expand(stats.norm()),
        paramseed=314
    )

    # cf. Task.compute_hashed_digest
    json_fixed = task.taskinputs.__config__.json_dumps(fixed_model, default=task.taskinputs.digest_encoder)
    json_stat  = task.taskinputs.__config__.json_dumps(stat_model, default=task.taskinputs.digest_encoder)
    digests = [stablehexdigest(json_fixed)[:8], stablehexdigest(json_stat)[:8]]
    digests2 = pytestconfig.cache.get("test-paramcoll-reproducible-digests", None)
    if digests2 is None:
        pytestconfig.cache.set("test-paramcoll-reproducible-digests", digests)
        raise RerunThisTest()
    elif digests != digests2:
        # Clear the cache first, so that the next recreates it with the
        # (hopefully fixed) code
        pytestconfig.cache.set("test-paramcoll-reproducible-digests", None)
        # We know this test will fail, but using `assert` produces a more informative message
        assert digests == digests2, \
            "Digests (hashes) for ParamColl objects are not consistent across runs." 


def test_nested_ParamColl():

    model_data_in_paramcoll = DataParamset(
        L=expand([100, 200, 300]),
        λ=ModelParamset(
            λ=expand([1.1, 1.2, 1.3])
        ),
        σ=expand(stats.poisson(1)),
        paramseed=312
    )
    model_data_in_dataclass = DataParamset(
        L=expand([100, 200, 300]),
        λ=ModelDataclass(
            λ=expand([1.1, 1.2, 1.3])
        ),
        σ=expand(stats.poisson(1)),
        paramseed=312
    )
    # Nested dataclass was converted to a ParamColl
    assert isinstance(model_data_in_paramcoll.λ, ParamColl)
    assert isinstance(model_data_in_dataclass.λ, ParamColl)

    assert model_data_in_paramcoll.inner_len \
        == model_data_in_dataclass.inner_len \
        == len(list(model_data_in_dataclass.inner())) \
        == 3

    assert model_data_in_paramcoll.outer_len \
        == model_data_in_dataclass.outer_len \
        == len(list(model_data_in_dataclass.outer())) \
        == 9  # NB: Random variables don't increase the size of an outer product

    assert list(map(asdict, model_data_in_paramcoll.inner())) \
        == list(map(asdict, model_data_in_dataclass.inner()))
    assert list(map(asdict, model_data_in_paramcoll.outer())) \
        == list(map(asdict, model_data_in_dataclass.outer()))

    # Inner length is updated recursively
    model_rv = DataParamset(
        L=expand(stats.poisson(100)),
        λ=ModelDataclass(
            λ=expand(stats.norm())
        ),
        σ=expand(stats.poisson(1)),
        paramseed=312
    )
    # Without setting the length, the param coll would return values forever
    assert model_rv.inner_len == np.inf
    # model_rv = replace(model_rv, inner_len=17)
    assert len(list(model_rv.inner(17))) == 17

    # Pickling works
    # Note that the difficult test is pickling a ParamColl with auto-generated nested class
    # (i.e. `model_data_in_dataclass`), since plain pickle can’t deal with dynamically generated classes
    # (They need to be importable)
    # Easy: All importable ParamColl types
    pickle_data = pickle.dumps(model_data_in_paramcoll, protocol=3)
    model_data_in_paramcoll2 = pickle.loads(pickle_data)
    assert list(map(asdict, model_data_in_paramcoll.inner())) \
        == list(map(asdict, model_data_in_paramcoll2.inner()))

    # Hard: Dynamically created ParamColl types, which are not importable
    pickle_data = pickle.dumps(model_data_in_dataclass, protocol=3)
    model_data_in_dataclass2 = pickle.loads(pickle_data)
    assert list(map(asdict, model_data_in_dataclass.inner())) \
        == list(map(asdict, model_data_in_dataclass2.inner()))

    # Harder: Dynamic ParamColl, not as instances, but as plain types
    PColl = type(model_data_in_dataclass.λ)
    assert isinstance(PColl, type) and issubclass(PColl, ParamColl)
    pickle_data = pickle.dumps(PColl)
    PColl2 = pickle.loads(pickle_data)
    assert PColl is PColl2



def test_SeedGenerator():
    class SeedGen(SeedGenerator):
        data: Tuple[int, int]
        noise: int
    seedgen = SeedGen(123456789)
    seeds1a = seedgen(1)
    seeds1b = seedgen(1)
    seeds2 = seedgen(2)
    assert (seeds1a.data == seeds1b.data).all() and seeds1a.noise == seeds1b.noise
    assert (seeds1a.data != seeds2.data).all() and seeds1a.noise != seeds2.noise
    assert len(seeds2.data) == 2
    assert np.isscalar(seeds2.noise)

    # Changing the entropy changes the seed values
    seedgen = SeedGen(987654321)
    seeds1c = seedgen(1)
    assert (seeds1a.data != seeds1c.data).any()

    # We also allow using non-integer values for the keys. This allows to easily
    # generate different keys for different parameter values.
    # Below we check that both the values and the order will change the seed.
    seeds4a = seedgen.data("b", 4.3)
    seeds4b = seedgen.data(4.3, "b")
    seeds4c = seedgen.data(4.3, "a")
    assert (seeds4a != seeds4b).any()
    assert (seeds4a != seeds4c).any()
    assert (seeds4b != seeds4c).any()
