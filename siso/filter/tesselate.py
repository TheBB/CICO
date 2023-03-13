from typing import TypeVar

from numpy import floating

from .. import api
from ..topology import Topology
from ..util import FieldData
from .passthrough import Passthrough


Z = TypeVar("Z", bound=api.Zone)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)


class Tesselate(Passthrough[F, S, Z, F, S, Z]):
    nvis: int

    def __init__(self, source: api.Source[F, S, Z], nvis: int):
        super().__init__(source)
        self.nvis = nvis

    def validate_source(self) -> None:
        assert not self.source.properties.tesselated

    @property
    def properties(self) -> api.SourceProperties:
        return super().properties.update(
            tesselated=True,
        )

    def topology(self, timestep: S, field: F, zone: Z) -> Topology:
        topology = self.source.topology(timestep, field, zone)
        tesselator = topology.tesselator(self.nvis)
        return tesselator.tesselate_topology(topology)

    def field_data(self, timestep: S, field: F, zone: Z) -> FieldData[floating]:
        topology = self.source.topology(timestep, field, zone)
        tesselator = topology.tesselator(self.nvis)
        field_data = self.source.field_data(timestep, field, zone)
        return tesselator.tesselate_field(topology, field, field_data)
