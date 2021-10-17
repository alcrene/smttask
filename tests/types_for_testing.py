import typing
from dataclasses import dataclass
from pydantic.generics import GenericModel, Generic

@dataclass
class Point:
    x: float
    y: float
    
T = typing.TypeVar("MyType")
class MyGenericModel(GenericModel, Generic[T]):
    a: T
