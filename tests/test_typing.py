from pydantic import BaseModel
from pydantic.fields import SHAPE_GENERIC

from smttask.typing import (separate_outputs, SeparateOutputs,
                            PureFunction, PurePartialFunction)

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

    task1 = AddPureFunctions(f1=f1,
                             f2=f2,
                             g1=functools.partial(g1, a=1),
                             g2=functools.partial(g2, x=1.5)
                             )

    assert task1.digest == '8488292d4c'
    assert task1.desc.json() == '{"taskname": "AddPureFunctions", "module": "tasks", "inputs": {"digest": "8488292d4c", "hashed_digest": "8488292d4c", "unhashed_digests": {}, "f1": "def f1(x):\\n    return (x + 1)", "f2": "def f2(p):\\n    return (1.5 ** p)", "g1": ["PurePartialFunction", "def g1(x, a):\\n    return (x + a)", {"a": 1}], "g2": ["PurePartialFunction", "def g2(x, p):\\n    return (x ** p)", {"x": 1.5}]}, "reason": null}'

    task1.run()

    output = task1.Outputs.parse_result(task1.run(), _task=task1)

    assert output.json() == '{"": "def h(x, p):\\n    return (((f1(x) + f2(p)) + g1(x)) + g2(p))"}'