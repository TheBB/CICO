from __future__ import annotations

import logging
from functools import reduce
from operator import itemgetter
from typing import Dict, Iterator, List, MutableMapping, Optional, Set, Tuple, TypeVar, cast

from numpy import floating

from ..api import Basis, Field, Shape, Source, SourceProperties, Step, Zone
from ..topology import Topology
from ..typing import Point
from ..util import FieldData, bisect
from .passthrough import PassthroughBFS


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=Field)
S = TypeVar("S", bound=Step)
Z = TypeVar("Z", bound=Zone)


class KeyZones(PassthroughBFS[B, F, S, Z, Zone]):
    manager: ZoneManager

    def __init__(self, source: Source):
        super().__init__(source)
        self.manager = ZoneManager()

    def validate_source(self) -> None:
        assert not self.source.properties.globally_keyed

    @property
    def properties(self) -> SourceProperties:
        return super().properties.update(
            globally_keyed=True,
        )

    def zones(self) -> Iterator[Zone]:
        for zone in self.source.zones():
            yield self.manager.lookup(zone)

    def topology(self, timestep: S, basis: B, zone: Zone) -> Topology:
        return self.source.topology(timestep, basis, cast(Z, zone))

    def field_data(self, timestep: S, field: F, zone: Zone) -> FieldData[floating]:
        return self.source.field_data(timestep, field, cast(Z, zone))


class ZoneManager:
    lut: VertexDict[Set[int]]
    shapes: Dict[int, Shape]

    def __init__(self):
        self.lut = VertexDict()
        self.shapes = dict()

    def lookup(self, zone: Zone) -> Zone:
        if zone.global_key is not None:
            assert self.shapes[zone.global_key] == zone.shape
            return cast(Zone, zone)

        keys = reduce(lambda x, y: x & y, (self.lut.get(pt, set()) for pt in zone.coords))
        assert len(keys) < 2

        if keys:
            key = next(iter(keys))
            assert self.shapes[key] == zone.shape
        else:
            key = len(self.shapes)
            assert key not in self.shapes
            self.shapes[key] = zone.shape
            for pt in zone.coords:
                self.lut.setdefault(pt, set()).add(key)
            logging.debug(f"Local zone '{zone.local_key}' associated with new global zone {key}")

        return Zone(
            shape=zone.shape,
            coords=zone.coords,
            local_key=zone.local_key,
            global_key=key,
        )


Q = TypeVar("Q")


class VertexDict(MutableMapping[Point, Q]):
    rtol: float
    atol: float

    _keys: List[Optional[Point]]
    _values: List[Optional[Q]]

    lut: Dict[int, List[Tuple[int, float]]]

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol
        self._keys = []
        self._values = []
        self.lut = dict()

    def _bounds(self, key: float):
        if key >= self.atol:
            return (
                (key - self.atol) / (1 + self.rtol),
                (key + self.atol) / (1 - self.rtol),
            )

        if key <= -self.atol:
            return (
                (key - self.atol) / (1 - self.rtol),
                (key + self.atol) / (1 + self.rtol),
            )

        return (
            (key - self.atol) / (1 - self.rtol),
            (key + self.atol) / (1 - self.rtol),
        )

    def _candidate(self, key: Point) -> int:
        candidates = None
        for coord, k in enumerate(key):
            lut = self.lut.setdefault(coord, [])
            minval, maxval = self._bounds(k)
            lo = bisect.bisect_left(lut, minval, key=itemgetter(1))
            hi = bisect.bisect_left(lut, maxval, key=itemgetter(1))
            if candidates is None:
                candidates = {i for i, _ in lut[lo:hi]}
            else:
                candidates &= {i for i, _ in lut[lo:hi]}
        if candidates is None:
            raise KeyError(key)
        for c in candidates:
            if self._keys[c] is not None:
                return c
        raise KeyError(key)

    def _insert(self, key: Point, value: Q) -> None:
        newindex = len(self._values)
        for coord, v in enumerate(key):
            lut = self.lut.setdefault(coord, [])
            bisect.insort(lut, (newindex, v), key=itemgetter(1))
        self._keys.append(key)
        self._values.append(value)

    def __setitem__(self, key: Point, value: Q) -> None:
        try:
            c = self._candidate(key)
            self._values[c] = value
        except KeyError:
            self._insert(key, value)

    def __getitem__(self, key: Point) -> Q:
        c = self._candidate(key)
        return cast(Q, self._values[c])

    def __delitem__(self, key: Point) -> None:
        try:
            i = self._candidate(key)
        except KeyError:
            return
        self._keys[i] = None
        self._values[i] = None

    def __iter__(self) -> Iterator[Point]:
        for key in self._keys:
            if key is not None:
                yield key

    def __len__(self) -> int:
        return len(self._values)
