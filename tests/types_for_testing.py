import typing
from dataclasses import dataclass
from typing import Generic
try:
    from pydantic.generics import GenericModel
except ImportError:
    from scityping.pydantic import GenericModel

@dataclass
class Point:
    x: float
    y: float
    
T = typing.TypeVar("MyType")
class MyGenericModel(GenericModel, Generic[T]):
    a: T
