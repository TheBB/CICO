from pathlib import Path
from typing import Generic, Iterator, List, TypeVar

from numpy import floating

from .. import api, coord
from ..api import Coords, Shape, Zone
from ..impl import Basis, Field, Step
from ..util import FieldData


T = TypeVar("T", bound=api.Topology)


class PureGeometry(api.Source[Basis, Field, Step, T, Zone[int]], Generic[T]):
    filename: Path
    corners: List[Coords]
    controlpoints: List[FieldData[floating]]
    topologies: List[T]

    def __init__(self, filename: Path):
        self.filename = filename
        self.corners = []
        self.topologies = []
        self.controlpoints = []

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=True,
            globally_keyed=True,
            single_basis=True,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        return

    def use_geometry(self, geometry: Field) -> None:
        return

    def bases(self) -> Iterator[Basis]:
        yield Basis("mesh")

    def basis_of(self, field: Field) -> Basis:
        return Basis("mesh")

    def fields(self, basis: Basis) -> Iterator[Field]:
        return
        yield

    def geometries(self, basis: Basis) -> Iterator[Field]:
        yield Field("Geometry", type=api.Geometry(self.controlpoints[0].num_comps, coords=coord.Generic()))

    def steps(self) -> Iterator[Step]:
        yield Step(index=0)

    def zones(self) -> Iterator[Zone[int]]:
        for i, (corners, topology) in enumerate(zip(self.corners, self.topologies)):
            shape = [Shape.Line, Shape.Quatrilateral, Shape.Hexahedron][topology.pardim - 1]
            yield Zone(
                shape=shape,
                coords=corners,
                key=i,
            )

    def topology(self, timestep: Step, basis: Basis, zone: Zone[int]) -> T:
        return self.topologies[zone.key]

    def field_data(self, timestep: Step, field: Field, zone: Zone[int]) -> FieldData[floating]:
        return self.controlpoints[zone.key]
