from typing import Dict, Tuple, TypeVar

from numpy import floating

from .. import api, util
from ..coord import ConversionPath, convert_coords, convert_vectors
from ..util import FieldData
from .passthrough import PassthroughAll


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
T = TypeVar("T", bound=api.Topology)
Z = TypeVar("Z", bound=api.Zone)


class CoordTransform(PassthroughAll[B, F, S, T, Z]):
    """Coordinate transform filter.

    This filter converts geometries and vector fields from one coordinate system
    (defined by the source geometry), to another.

    Parameters:
    - source: data source to draw from.
    - path: the coordinate system conversion path to use. Generated either by
        `siso.coord.conversion_path` or `siso.coord.optimal_system`.
    """

    path: ConversionPath

    # Conversion of vector fields requires (in general) knowledge of the
    # coordinates of the points those vector fields are defined on. We store
    # them in this dict, mapping coordinate system name and zone to nodes.
    cache: Dict[Tuple[str, Z], FieldData[floating]]

    def __init__(self, source: api.Source[B, F, S, T, Z], path: ConversionPath):
        super().__init__(source)
        self.path = path
        self.cache = {}

    def field_data(self, timestep: S, field: F, zone: Z) -> FieldData[floating]:
        indata = self.source.field_data(timestep, field, zone)

        # Coordinate conversion leaves scalar fields alone.
        if not field.is_geometry and not field.is_vector:
            return indata

        # For geometries, convert and store in cache.
        if field.is_geometry:
            for a, b in util.pairwise(self.path):
                self.cache[a.name, zone] = indata
                indata = convert_coords(a, b, indata)
            return indata

        # For vectors, use cached data to convert.
        for a, b in util.pairwise(self.path):
            indata = convert_vectors(a, b, indata, self.cache[a.name, zone])

        return indata
