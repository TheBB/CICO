from typing import Dict, Tuple, TypeVar

from numpy import floating

from .. import api
from ..topology import DiscreteTopology
from ..util import FieldData
from .passthrough import PassthroughBFSZ


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
T = TypeVar("T", bound=api.Topology)
Z = TypeVar("Z", bound=api.Zone)


class Discretize(PassthroughBFSZ[B, F, S, Z, T, DiscreteTopology]):
    nvis: int
    mappers: Dict[Tuple[B, Z], api.FieldDataFilter]

    def __init__(self, source: api.Source[B, F, S, T, Z], nvis: int):
        super().__init__(source)
        self.nvis = nvis
        self.mappers = {}

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(
            discrete_topology=True,
        )

    def topology(self, step: S, basis: B, zone: Z) -> DiscreteTopology:
        topology = self.source.topology(step, basis, zone)
        discrete, mapper = topology.discretize(self.nvis)
        self.mappers[(basis, zone)] = mapper
        return discrete

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        data = self.source.field_data(step, field, zone)
        basis = self.source.basis_of(field)
        mapper = self.mappers[(basis, zone)]
        q = mapper(field, data)
        return q
