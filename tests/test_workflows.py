from dataclasses import dataclass, asdict
from typing import Tuple
import numpy as np
from scipy import stats
import pytest

from smttask.workflows import ParamColl, expand, SeedGenerator

def test_ParamColl():

    # (Default param values are required for tests to run on Python <3.10)
    @dataclass
    class DataParamset(ParamColl):
        L: int    = 100
        λ: float  = 1
        σ: float  = 1
        δy: float = 0.1

    @dataclass
    class ModelParamset(ParamColl):
        λ: float  = 1
        σ: float  = 1

    #
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

    assert dict(**data_params) == {k: v for k,v in asdict(data_params).items()
                                      if not k.startswith("_") and k not in {"seed", "inner_len"}}

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
            λ=stats.norm(),
            σ=1,
            δy=expand([-1, -0.3, 0, 0.3, 1])
        )


    data_params = DataParamset(
        L=400,
        λ=stats.norm(),
        σ=expand([1, 0.2, 0.05]),
        δy=expand([-1, -0.3, 0, 0.3, 1]),
        seed=314
    )

    # σ and δy are not aligned: cannot do inner product
    with pytest.raises(ValueError):
        list(data_params.inner())  # `list()` is required to trigger error
    assert len(list(data_params.outer())) == data_params.outer_len == 3*5

    # Now align σ and δy: can do both inner and outer product
    data_params.σ = expand([10, 0, 1, 0.2, 0.05])
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
    data_params.seed = 628
    ival3 = tuple(p.λ for p in data_params.inner())
    oval3 = tuple(p.λ for p in data_params.outer())
    assert ival1 != ival3
    assert oval1 != oval3

    # When there are only random and scalar values, only `inner` is possible
    data_params.σ = stats.lognorm(1)
    data_params.δy = stats.uniform(-1, 1)
    with pytest.raises(ValueError):
        list(zip(data_params.outer(), range(10)))
    σvals = [p.σ for p, _ in zip(data_params.inner(), range(100))]  # Infinite iterate: use range(100) to truncate
    assert len(σvals) == 100  # All generated values are different

    # Can set the inner length to convert from infinite to finite collection
    data_params.inner_len = 10
    σvals2 = [p.σ for p in data_params.inner()]
    assert len(σvals2) == 10
    assert σvals2 == σvals[:10]

# Test that the seed leads to reproducible parameter sets across runs
# IMPORTANT: This test needs to be run twice, in *separate* processes. The easiest way to do that is to 
def test_ParamColl_reproducible():
    @dataclass
    class ModelParamset(ParamColl):
        λ: float = 1
        σ: float = 1

    model_params = ModelParamset(
        λ=stats.norm(),
        σ=stats.norm(),
        seed=314
    )
    s = str([dict(**p) for p in model_params.inner(1)])

    fname = "emdd-test-paramcoll-reproducible.txt"
    try:
        with open(fname, 'r') as f:
            s2 = f.read()
    except FileNotFoundError:
        with open(fname, 'w') as f:
            f.write(s)
    else:
        assert s == s2, "Random ParamColls are not consistent across processes, even with fixed " \
            "seed. To catch this error, the `test_ParamColl_reproducible` test needs to be run " \
            "*twice* (which is why this error may have been missed on a first run). " \
            f"Once the problem is fixed, you need to delete the artifact `{fname}` then run the " \
            "test again twice."

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
