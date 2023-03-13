from itertools import count, islice
from typing import Generic, Iterator, List, Optional, Tuple, TypeVar, overload

from attrs import define
from numpy import floating

from .. import api, util
from .passthrough import Passthrough


@overload
def islice_flag(stop: Optional[int], /) -> Iterator[bool]:
    ...


@overload
def islice_flag(start: Optional[int], stop: Optional[int], step: Optional[int], /) -> Iterator[bool]:
    ...


def islice_flag(*args):
    counter = islice(count(), *args)
    try:
        next_index = next(counter)
    except StopIteration:
        return

    for i in count():
        yield i == next_index
        if i == next_index:
            try:
                next_index = next(counter)
            except StopIteration:
                return


T = TypeVar("T")


@overload
def islice_group(it: Iterator[T], stop: Optional[int], /) -> Iterator[List[T]]:
    ...


@overload
def islice_group(
    it: Iterator[T], start: Optional[int], stop: Optional[int], step: Optional[int], /
) -> Iterator[List[T]]:
    ...


def islice_group(it, *args):
    accum = []
    for item, flag in zip(it, islice_flag(*args)):
        accum.append(item)
        if flag:
            yield accum
            accum = []


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
S = TypeVar("S", bound=api.Step)


@define
class GroupedStep(Generic[S]):
    index: int
    steps: List[S]

    @property
    def value(self) -> Optional[float]:
        return self.steps[-1].value


class GroupedTimeSource(Passthrough[F, S, Z, F, GroupedStep[S], Z]):
    def topology(self, step: GroupedStep[S], field: F, zone: Z) -> api.Topology:
        return self.source.topology(step.steps[-1], field, zone)

    def field_data(self, step: GroupedStep[S], field: F, zone: Z) -> util.FieldData[floating]:
        return self.source.field_data(step.steps[-1], field, zone)

    def field_updates(self, step: GroupedStep[S], field: F) -> bool:
        return any(self.source.field_updates(s, field) for s in step.steps)


class StepSlice(GroupedTimeSource[F, S, Z]):
    arguments: Tuple[Optional[int]]

    def __init__(self, source: api.Source[F, S, Z], arguments: Tuple[Optional[int]]):
        super().__init__(source)
        self.arguments = arguments

    def steps(self) -> Iterator[GroupedStep[S]]:
        for i, times in enumerate(islice_group(self.source.steps(), *self.arguments)):
            yield GroupedStep(i, times)


class LastTime(GroupedTimeSource[F, S, Z]):
    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(instantaneous=True)

    def steps(self) -> Iterator[GroupedStep[S]]:
        steps = list(self.source.steps())
        yield GroupedStep(0, steps)
