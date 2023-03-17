from typing import TypeVar, cast

from ..api import Basis, Field, Step, Zone
from ..topology import DiscreteTopology, UnstructuredTopology
from .passthrough import Passthrough


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=Field)
S = TypeVar("S", bound=Step)
Z = TypeVar("Z", bound=Zone)


class ForceUnstructured(Passthrough[B, F, S, Z, B, F, S, Z]):
    def validate_source(self) -> None:
        assert self.source.properties.tesselated

    def topology(self, timestep: S, basis: B, zone: Z) -> UnstructuredTopology:
        topology = cast(DiscreteTopology, self.source.topology(timestep, basis, zone))
        if not isinstance(topology, UnstructuredTopology):
            return UnstructuredTopology(
                num_nodes=topology.num_nodes,
                cells=topology.cells,
                celltype=topology.celltype,
            )
        return topology
