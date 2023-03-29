from functools import reduce
from typing import Iterator, List

from attrs import define
from numpy import floating

from .. import api
from ..api import B, F, S, T, Z
from ..util import FieldData
from .passthrough import PassthroughBSTZ, WrappedField


@define
class RecombinedField(WrappedField[F]):
    sources: List[F]
    name: str

    def __post_init__(self):
        assert all(src.cellwise == self.sources[0].cellwise for src in self.sources)
        assert all(src.type == self.sources[0].type for src in self.sources)

    @property
    def wrapped_field(self) -> F:
        return self.sources[0]

    @property
    def type(self) -> api.FieldType:
        return reduce(lambda x, y: x.concat(y), (s.type for s in self.sources))

    @property
    def splittable(self) -> bool:
        if len(self.sources) == 1:
            return self.sources[0].splittable
        return False


class Recombine(PassthroughBSTZ[B, S, T, Z, F, RecombinedField[F]]):
    recombinations: List[api.RecombineFieldSpec]

    def __init__(self, source: api.Source, recombinations: List[api.RecombineFieldSpec]):
        super().__init__(source)
        self.recombinations = recombinations

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(
            recombine_fields=[],
        )

    def use_geometry(self, geometry: RecombinedField[F]) -> None:
        self.source.use_geometry(geometry.wrapped_field)

    def basis_of(self, field: RecombinedField[F]) -> B:
        return self.source.basis_of(field.wrapped_field)

    def geometries(self, basis: B) -> Iterator[RecombinedField]:
        for field in self.source.geometries(basis):
            yield RecombinedField(name=field.name, sources=[field])

    def fields(self, basis: B) -> Iterator[RecombinedField]:
        in_fields = {field.name: field for field in self.source.fields(basis)}

        for field in in_fields.values():
            yield RecombinedField(name=field.name, sources=[field])

        for spec in self.recombinations:
            if all(src in in_fields for src in spec.source_names):
                yield RecombinedField(
                    name=spec.new_name, sources=[in_fields[src] for src in spec.source_names]
                )

    def field_data(self, timestep: S, field: RecombinedField[F], zone: Z) -> FieldData[floating]:
        return FieldData.concat(self.source.field_data(timestep, src, zone) for src in field.sources)

    def field_updates(self, timestep: S, field: RecombinedField[F]) -> bool:
        return any(self.source.field_updates(timestep, src) for src in field.sources)
