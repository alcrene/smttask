import pytest
from pydantic import BaseModel
from pydantic.fields import SHAPE_GENERIC

from smttask.typing import (separate_outputs, SeparateOutputs,
                            PureFunction, PartialPureFunction)

from pathlib import Path
import functools
import sys
import os
import smttask
testroot = Path(smttask.__file__).parent.parent/"tests"
os.chdir(testroot)
from utils_for_testing import clean_project

def test_separate_output():

    class Foo(BaseModel):
        a: separate_outputs(int, lambda: ["a"])

    assert Foo.__fields__['a'].shape == SHAPE_GENERIC
    assert issubclass(Foo.__fields__['a'].type_, SeparateOutputs)


    # No conversion
    r = Foo(a=(1,)).a
    assert r == (1,) and isinstance(r, tuple) and isinstance(r[0], int)
    assert isinstance(r, SeparateOutputs)
    # list  -> tuple
    r = Foo(a=[1,]).a
    assert r == (1,) and isinstance(r, tuple) and isinstance(r[0], int)
    # float -> int
    r = Foo(a=(1.,)).a
    assert r == (1,) and isinstance(r, tuple) and isinstance(r[0], int)
    # list -> tuple, float -> int
    r = Foo(a=[1.,]).a
    assert r == (1,) and isinstance(r, tuple) and isinstance(r[0], int)


    class Foo2(BaseModel):
        a: int
    class Bar(BaseModel):
        foos: separate_outputs(Foo2, lambda: ["foo"])

    r = Bar(foos=(Foo2(a=3.),))
    assert isinstance(r.foos[0].a, int) and r.foos[0].a == 3

def test_pure_functions():

    # Add test directory to import search path
    projectroot = testroot/"test_project"
    projectpath = str(projectroot.absolute())
    if str(projectpath) not in sys.path:
        sys.path.insert(0, projectpath)
    # Clear the runtime directory and cd into it
    clean_project(projectroot)
    os.makedirs(projectroot/"data", exist_ok=True)
    os.chdir(projectroot)

    from tasks import AddPureFunctions

    def f1(x): return x+1
    def f2(p): return 1.5**p
    def g1(x, a): return x+a
    def g2(x, p): return x**p

    # Test Function arithmetic (see CompositePureFunction)
    xlst = (-1.2, 0.5, 3)
    pure_f1 = PureFunction(f1)
    pure_f2 = PureFunction(f2)
    with pytest.raises(TypeError):
        # Fails because f2 is not pure
        h = pure_f1 + f2
    # Test all ops, include reversed versions
    h = pure_f1 + pure_f2
    assert [h(x) for x in xlst] == [f1(x) + f2(x) for x in xlst]
    h = pure_f1 + 5
    assert [h(x) for x in xlst] == [f1(x) + 5 for x in xlst]
    h = 9.2 + pure_f1
    assert [h(x) for x in xlst] == [9.2 + f1(x) for x in xlst]
    h = pure_f1 - pure_f2
    assert [h(x) for x in xlst] == [f1(x) - f2(x) for x in xlst]
    h = pure_f1 - 5
    assert [h(x) for x in xlst] == [f1(x) - 5 for x in xlst]
    h = 9.2 - pure_f1
    assert [h(x) for x in xlst] == [9.2 - f1(x) for x in xlst]
    h = pure_f1 * pure_f2
    assert [h(x) for x in xlst] == [f1(x) * f2(x) for x in xlst]
    h = pure_f1 * 5
    assert [h(x) for x in xlst] == [f1(x) * 5 for x in xlst]
    h = 9.2 * pure_f1
    assert [h(x) for x in xlst] == [9.2 * f1(x) for x in xlst]
    h = pure_f1 / pure_f2
    assert [h(x) for x in xlst] == [f1(x) / f2(x) for x in xlst]
    h = pure_f1 / 5
    assert [h(x) for x in xlst] == [f1(x) / 5 for x in xlst]
    h = 9.2 / pure_f1
    assert [h(x) for x in xlst] == [9.2 / f1(x) for x in xlst]

    task1 = AddPureFunctions(f1=f1,
                             f2=f2,
                             g1=functools.partial(g1, a=1),
                             g2=functools.partial(g2, x=1.5),
                             f3=h
                             )

    assert task1.digest == 'b3f5fddcf8'
    assert task1.desc.json() == '{"taskname": "AddPureFunctions", "module": "tasks", "inputs": {"digest": "b3f5fddcf8", "hashed_digest": "b3f5fddcf8", "unhashed_digests": {}, "f1": "def f1(x):\\n    return (x + 1)", "f2": "def f2(p):\\n    return (1.5 ** p)", "g1": ["PartialPureFunction", "def g1(x, a):\\n    return (x + a)", {"a": 1}], "g2": ["PartialPureFunction", "def g2(x, p):\\n    return (x ** p)", {"x": 1.5}], "f3": ["CompositePureFunction", "truediv", [9.2, "def f1(x):\\n    return (x + 1)"]]}, "reason": null}'

    task1.run()

    # Check that serialize->deserialize works
    from mackelab_toolbox.serialize import config as serialize_config
    serialize_config.trust_all_inputs = True
    task2 = smttask.Task.from_desc(task1.desc.json())
    assert task1.run()(0.5, 2) == task2.run()(0.5, 2)

    output = task1.Outputs.parse_result(task1.run(), _task=task1)

    assert output.json() == '{"": "def h(x, p):\\n    return ((((f1(x) + f2(p)) + g1(x)) + g2(p=p)) + f3(x))"}'

def wip_test_pure_functions_ufunc():
    import numpy as np
    
    # We need a way to serialize plain NumPy ufuncs
    # The line below currently does not work
    # Even better might be to support serialization of ufuncs directly, since they are
    # already pure; requiring to wrap them with PureFunction is needlessly complicated
    # (We could turn off the safety warning for functions imported from numpy as well)
    PureFunction(np.exp)
