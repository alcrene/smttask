from pydantic import BaseModel
from pydantic.fields import SHAPE_GENERIC

from smttask.typing import separate_outputs, SeparateOutputs

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
