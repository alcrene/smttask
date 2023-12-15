import pytest
from pydantic import BaseModel, ValidationError
from pydantic.fields import SHAPE_GENERIC

from scityping import Type
from scityping.functions import PureFunction, PartialPureFunction
from smttask.typing import separate_outputs, SeparateOutputs

from pathlib import Path
import functools
import sys
import os
import typing
from typing import List
import smttask
# testroot = Path(smttask.__file__).parent.parent.parent/"tests"
testroot = Path(__file__).parent; assert testroot.stem == "tests"
os.chdir(testroot)
from utils_for_testing import clean_project
from types_for_testing import MyGenericModel

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
    pure_g1 = PureFunction(g1)
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

    with pytest.raises(ValidationError): # f1 is non-pure
        task1 = AddPureFunctions(f1=f1,
                                 f2=pure_f2,
                                 g1=functools.partial(pure_g1, a=1),
                                 g2=PureFunction(functools.partial(g2, x=1.5)),
                                 f3=h
                                 )
    with pytest.raises(ValidationError): # g2 is non-pure
        task1 = AddPureFunctions(f1=pure_f1,
                                 f2=pure_f2,
                                 g1=functools.partial(pure_g1, a=1),
                                 g2=functools.partial(g2, x=1.5),
                                 f3=h
                                 )
    task1 = AddPureFunctions(f1=pure_f1,
                             f2=pure_f2,
                             g1=functools.partial(pure_g1, a=1),
                             g2=PureFunction(functools.partial(g2, x=1.5)),
                             f3=h
                             )


    assert task1.digest == "bee821c7d0"
    assert task1.desc.json() == '{"taskname": "AddPureFunctions", "module": "tasks", "inputs": {"digest": "bee821c7d0", "hashed_digest": "bee821c7d0", "unhashed_digests": {}, "f1": ["scityping.functions.PureFunction", {"func": "def f1(x):\\n    return x + 1"}], "f2": ["scityping.functions.PureFunction[[int], float]", {"func": "def f2(p):\\n    return 1.5 ** p"}], "g1": ["scityping.functions.PartialPureFunction", {"func": ["scityping.functions.PureFunction", {"func": "def g1(x, a):\\n    return x + a"}], "args": [], "kwargs": {"a": 1}}], "g2": ["scityping.functions.PartialPureFunction", {"func": "def g2(x, p):\\n    return x ** p", "args": [], "kwargs": {"x": 1.5}}], "f3": ["scityping.functions.CompositePureFunction", {"opname": "truediv", "terms": [9.2, ["scityping.functions.PureFunction", {"func": "def f1(x):\\n    return x + 1"}]]}]}, "reason": null}'

    task1.run()

    # Check that serialize->deserialize works
    import scityping
    scityping.config.trust_all_inputs = True
    task2 = smttask.Task.from_desc(task1.desc.json())
    assert task1.run()(0.5, 2) == task2.run()(0.5, 2)

    output = task1.Outputs.parse_result(task1.run(), _task=task1)

    with open("/home/alex/tmp/test-digests.txt", 'w') as f:
        f.write(output.json())
    assert output.json() == '{"": ["scityping.functions.PureFunction", {"func": "@PureFunction\\ndef h(x, p):\\n    return f1(x) + f2(p) + g1(x) + g2(p=p) + f3(x)"}]}'


test_pure_functions()