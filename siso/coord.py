from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, cast

import erfa
import numpy as np
from attrs import define
from numpy import floating
from typing_extensions import Self

from . import api, util
from .util import FieldData, coord


systems: util.Registry[api.CoordinateSystem] = util.Registry()
ellpsoids: util.Registry[Ellipsoid] = util.Registry()


def find_system(code: str) -> api.CoordinateSystem:
    name, *params = code.split(":")
    if name in systems:
        return systems[name].make(params)
    return Named.make((code,))


@systems.register
@define
class Generic(api.CoordinateSystem):
    name: ClassVar[str] = "Generic"

    @classmethod
    def make(cls, params: Sequence[str]) -> Generic:
        assert not params
        return cls()

    @classmethod
    def default(cls) -> Generic:
        return cls()

    @property
    def parameters(self) -> Tuple[str, ...]:
        return cast(Tuple[str], ())


@systems.register
@define
class Named(api.CoordinateSystem):
    name: ClassVar[str] = "Named"
    identifier: str

    @classmethod
    def make(cls, params: Sequence[str]) -> Named:
        (name,) = params
        return cls(name)

    @classmethod
    def default(cls) -> Named:
        assert False

    @property
    def parameters(self) -> Tuple[str, ...]:
        if self.identifier:
            return (self.identifier,)
        return cast(Tuple[str], ())

    def fits_system_name(self, code: str) -> bool:
        return code.casefold() == self.identifier.casefold()


@systems.register
@define
class Geodetic(api.CoordinateSystem):
    name: ClassVar[str] = "Geodetic"
    ellipsoid: Ellipsoid

    @classmethod
    def make(cls, params: Sequence[str]) -> Geodetic:
        assert len(params) < 2
        if params:
            return cls(ellpsoids[params[0]]())
        return cls(Wgs84())

    @classmethod
    def default(cls) -> Geodetic:
        return cls(Wgs84())

    @property
    def parameters(self) -> Tuple[str, ...]:
        return (self.ellipsoid.name,)


@systems.register
@define
class Utm(api.CoordinateSystem):
    name: ClassVar[str] = "UTM"
    zone_number: int
    zone_letter: str

    @classmethod
    def make(cls, params: Sequence[str]) -> Utm:
        (zone,) = params
        try:
            i = next(i for i in range(len(zone)) if not zone[i].isnumeric())
        except StopIteration:
            raise ValueError(zone)
        zone_number = int(zone[:i])
        zone_letter = zone[i:].upper()
        if len(zone_letter) > 1:
            zone_letter = "N" if zone_letter.startswith("N") else "M"
        return cls(zone_number, zone_letter)

    @classmethod
    def default(cls) -> Utm:
        assert False

    @property
    def parameters(self) -> Tuple[str, ...]:
        return (str(self.zone_number), self.zone_letter)

    @property
    def northern(self) -> bool:
        return self.zone_letter >= "N"


@systems.register
class Geocentric(api.CoordinateSystem):
    name = "Geocentric"
    parameters = cast(Tuple[str], ())

    @classmethod
    def make(cls, params: Sequence[str]) -> Self:
        assert not params
        return cls()

    @classmethod
    def default(cls) -> Geocentric:
        return cls()


class Ellipsoid(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def semi_major_axis(self) -> float:
        ...

    @property
    @abstractmethod
    def flattening(self) -> float:
        ...


@ellpsoids.register
@define
class SphericalEarth(Ellipsoid):
    name = "Sphere"
    flattening: float = 0.0
    semi_major_axis: float = 6_371_008.8


class ErfaEllipsoid(Ellipsoid):
    erfa_code: ClassVar[int]

    @property
    def semi_major_axis(self) -> float:
        return erfa.eform(self.erfa_code)[0]

    @property
    def flattening(self) -> float:
        return erfa.eform(self.erfa_code)[1]


@ellpsoids.register
@define
class Wgs84(ErfaEllipsoid):
    erfa_code = 1
    name = "WGS84"


@ellpsoids.register
@define
class Grs80(ErfaEllipsoid):
    erfa_code = 2
    name = "GRS80"


@ellpsoids.register
@define
class Wgs72(ErfaEllipsoid):
    erfa_code = 3
    name = "WGS72"


T = TypeVar("T", bound=api.CoordinateSystem)
S = TypeVar("S", bound=api.CoordinateSystem)


CoordConverter = Callable[[T, S, FieldData[floating]], FieldData[floating]]
VectorConverter = Callable[[T, S, FieldData[floating], FieldData[floating]], FieldData[floating]]
ConversionPath = List[api.CoordinateSystem]


NEIGHBORS: Dict[str, List[str]] = {}
COORD_CONVERTERS: Dict[Tuple[str, str], CoordConverter] = {}
VECTOR_CONVERTERS: Dict[Tuple[str, str], VectorConverter] = {}


def register_coords(
    src: Type[api.CoordinateSystem], tgt: Type[api.CoordinateSystem]
) -> Callable[[CoordConverter[T, S]], CoordConverter[T, S]]:
    def decorator(conv: CoordConverter) -> CoordConverter:
        NEIGHBORS.setdefault(src.name, []).append(tgt.name)
        COORD_CONVERTERS[(src.name, tgt.name)] = conv
        return conv

    return decorator


def register_vectors(
    src: Type[api.CoordinateSystem], tgt: Type[api.CoordinateSystem]
) -> Callable[[VectorConverter[T, S]], VectorConverter[T, S]]:
    def decorator(conv: VectorConverter) -> VectorConverter:
        VECTOR_CONVERTERS[(src.name, tgt.name)] = conv
        return conv

    return decorator


def conversion_path(src: api.CoordinateSystem, tgt: api.CoordinateSystem) -> Optional[ConversionPath]:
    if src == tgt:
        return []
    if isinstance(src, (Generic, Named)) and isinstance(tgt, Generic):
        return []

    visited: Dict[str, str] = {}
    queue: deque[str] = deque((src.name,))

    def construct_backpath() -> ConversionPath:
        path = [tgt]
        name = visited[tgt.name]
        while name != src.name:
            path.append(systems[name].default())
            name = visited[name]
        path.append(src)
        return path[::-1]

    while queue:
        system = queue.popleft()
        for neighbor in NEIGHBORS.get(system, []):
            if neighbor in visited or neighbor == src.name:
                continue
            visited[neighbor] = system
            if neighbor == tgt.name:
                return construct_backpath()
            queue.append(neighbor)

    return None


def optimal_system(
    systems: Sequence[api.CoordinateSystem], target: api.CoordinateSystem
) -> Optional[Tuple[int, ConversionPath]]:
    optimal: Optional[Tuple[int, ConversionPath]] = None

    for i, system in enumerate(systems):
        new_path = conversion_path(system, target)
        if new_path is None:
            continue
        if optimal is None:
            optimal = i, new_path
        _, prev_path = optimal
        if len(new_path) < len(prev_path):
            optimal = i, new_path

    return optimal


def convert_coords(
    src: api.CoordinateSystem,
    tgt: api.CoordinateSystem,
    data: FieldData[floating],
) -> FieldData[floating]:
    return COORD_CONVERTERS[(src.name, tgt.name)](src, tgt, data)


def convert_vectors(
    src: api.CoordinateSystem,
    tgt: api.CoordinateSystem,
    data: FieldData[floating],
    coords: FieldData[floating],
) -> FieldData[floating]:
    return VECTOR_CONVERTERS[(src.name, tgt.name)](src, tgt, data, coords)


@register_coords(Geodetic, Geocentric)
def _(src: Geodetic, tgt: Geocentric, data: FieldData[floating]) -> FieldData[floating]:
    lon, lat, height = data.components
    return FieldData(
        erfa.gd2gce(
            src.ellipsoid.semi_major_axis,
            src.ellipsoid.flattening,
            np.deg2rad(lon),
            np.deg2rad(lat),
            height,
        )
    )


@register_vectors(Geodetic, Geocentric)
def _(
    src: Geodetic, tgt: Geocentric, data: FieldData[floating], coords: FieldData[floating]
) -> FieldData[floating]:
    return data.spherical_to_cartesian_vector_field(coords)


@register_coords(Geodetic, Utm)
def _(src: Geodetic, tgt: Utm, data: FieldData[floating]) -> FieldData[floating]:
    lon, lat, *rest = data.components
    x, y = coord.lonlat_to_utm(lon, lat, tgt.zone_number, tgt.zone_letter)
    return FieldData.concat(x, y, *rest)


@register_vectors(Geodetic, Utm)
def _(src: Geodetic, tgt: Utm, data: FieldData[floating], coords: FieldData[floating]) -> FieldData[floating]:
    lon, lat, *_ = coords.components
    in_x, in_y, *rest = data.components
    out_x, out_y = coord.lonlat_to_utm_vf(lon, lat, in_x, in_y, tgt.zone_number, tgt.zone_letter)
    return FieldData.concat(out_x, out_y, *rest)


@register_coords(Utm, Geodetic)
def _(src: Utm, tgt: Geodetic, data: FieldData[floating]) -> FieldData[floating]:
    x, y, *rest = data.components
    converter = coord.UtmConverter(tgt.semi_major_axis, tgt.flattening, src.zone_number, src.northern)
    lon, lat = converter.to_lonlat(x, y, src.zone_number, src.zone_letter)
    return FieldData.concat(lon, lat, *rest)


@register_vectors(Utm, Geodetic)
def _(src: Utm, tgt: Geodetic, data: FieldData[floating], coords: FieldData[floating]) -> FieldData[floating]:
    x, y, *_ = coords.components
    in_x, in_y, *rest = data.components
    converter = coord.UtmConverter(src.semi_major_axis, src.flattening, tgt.zone_number, tgt.northern)
    out_x, out_y = converter.to_lonlat_vf(x, y, in_x, in_y, src.zone_number, src.zone_letter)
    return FieldData.concat(out_x, out_y, *rest)
