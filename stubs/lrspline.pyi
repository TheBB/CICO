from collections.abc import Iterator
from typing import IO

from numpy import ndarray

class Element:
    id: int
    def start(self) -> tuple[float, ...]: ...
    def end(self) -> tuple[float, ...]: ...

class ElementView:
    def __iter__(self) -> Iterator[Element]: ...
    def __len__(self) -> int: ...

class LRSplineObject:
    dimension: int
    pardim: int
    controlpoints: ndarray
    elements: ElementView
    def corners(self) -> ndarray: ...
    @staticmethod
    def read_many(stream: IO) -> list[LRSplineObject]: ...
    def __len__(self) -> int: ...
    def element_at(self, *args: float) -> Element: ...
    def clone(self) -> LRSplineObject: ...
    def __call__(self, *args: ndarray) -> ndarray: ...
