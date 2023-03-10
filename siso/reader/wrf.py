from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Iterator, Optional, Tuple

import numpy as np
from netCDF4 import Dataset
from numpy import floating, integer
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from .. import api, util
from ..coord import Generic, Geodetic, SphericalEarth
from ..field import Field
from ..timestep import TimeStep
from ..topology import CellType, DiscreteTopology, StructuredTopology, UnstructuredTopology
from ..util import FieldData
from ..zone import Shape, Zone


class FieldDimensionality(Enum):
    Planar = auto()
    Volumetric = auto()
    Unknown = auto()


class NetCdf(api.Source[Field, TimeStep, Zone]):
    filename: Path
    dataset: Dataset

    volumetric: bool
    valid_domains: Tuple[FieldDimensionality, ...]
    staggering: api.Staggering
    periodic: bool
    geodetic: bool

    longitude_name: ClassVar[str]
    latitude_name: ClassVar[str]
    height_name: ClassVar[str]

    def __init__(self, path: Path):
        self.filename = path

    def __enter__(self) -> Self:
        self.dataset = Dataset(self.filename, "r").__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.dataset.__exit__(*args)

    @property
    def num_timesteps(self) -> int:
        return len(self.dataset.dimensions["Time"])

    @property
    def num_latitude(self) -> int:
        return len(
            self.dataset.dimensions[
                "south_north" if self.staggering == api.Staggering.Inner else "south_north_stag"
            ]
        )

    @property
    def num_longitude(self) -> int:
        return len(
            self.dataset.dimensions[
                "west_east" if self.staggering == api.Staggering.Inner else "west_east_stag"
            ]
        )

    @property
    def num_vertical(self) -> int:
        return len(
            self.dataset.dimensions[
                "bottom_top" if self.staggering == api.Staggering.Inner else "bottom_top_stag"
            ]
        )

    @property
    def num_planar(self) -> int:
        return self.num_latitude * self.num_longitude

    @property
    def wrf_planar_nodeshape(self) -> Tuple[int, ...]:
        return (self.num_latitude, self.num_longitude)

    @property
    def wrf_planar_cellshape(self) -> Tuple[int, ...]:
        return tuple(s - 1 for s in self.wrf_planar_nodeshape)

    @property
    def wrf_nodeshape(self) -> Tuple[int, ...]:
        planar_shape = self.wrf_planar_nodeshape
        if not self.volumetric:
            return planar_shape
        return (self.num_vertical, *planar_shape)

    @property
    def wrf_cellshape(self) -> Tuple[int, ...]:
        return tuple(s - 1 for s in self.wrf_nodeshape)

    def field_domain(self, name: str) -> FieldDimensionality:
        try:
            time, *dimensions = self.dataset[name].dimensions
        except IndexError:
            return FieldDimensionality.Unknown

        if time != "Time":
            return FieldDimensionality.Unknown

        try:
            x, y = dimensions
            assert x.startswith("south_north")
            assert y.startswith("west_east")
            return FieldDimensionality.Planar
        except (AssertionError, ValueError):
            pass

        try:
            x, y, z = dimensions
            assert x.startswith("bottom_top")
            assert y.startswith("south_north")
            assert z.startswith("west_east")
            return FieldDimensionality.Volumetric
        except (AssertionError, ValueError):
            pass

        return FieldDimensionality.Unknown

    def field_data_raw(
        self,
        name: str,
        index: int,
        extrude_if_volumetric: bool = True,
        include_poles_if_periodic: bool = True,
    ) -> FieldData[floating]:
        time, *dimensions = self.dataset[name].dimensions
        assert time == "Time"
        assert len(dimensions) in (2, 3)
        dimensions = list(dimensions)
        data = self.dataset[name][index, ...]

        # Handle staggering
        for dim, dim_name in enumerate(dimensions):
            if dim_name.endswith("_stag") and self.staggering == api.Staggering.Inner:
                data = util.unstagger(data, dim)
                dimensions[dim] = dim_name[:-5]
            elif not dim_name.endswith("_stag") and self.staggering == api.Staggering.Outer:
                data = util.stagger(data, dim)
                dimensions[dim] = f"{dim}_stag"

        if len(dimensions) == 3 and not self.volumetric:
            index = len(self.dataset.dimensions["soil_layers_stag"]) - 1
            data = data[index, ...]
            dimensions = dimensions[1:]

        south: Optional[np.ndarray] = None
        north: Optional[np.ndarray] = None
        if include_poles_if_periodic and self.periodic:
            if name in (self.longitude_name, self.latitude_name):
                south = util.angular_mean(data[0])
                north = util.angular_mean(data[-1])
            else:
                south = np.mean(data[..., 0, :], axis=-1)
                north = np.mean(data[..., -1, :], axis=-1)

        if len(dimensions) == 3:
            data = data.reshape(self.num_vertical, -1)
        else:
            data = data.flatten()

        if include_poles_if_periodic and self.periodic:
            to_append = np.array([south, north]).T
            data = np.append(data, to_append, axis=-1)

        if len(dimensions) == 2 and extrude_if_volumetric and self.volumetric:
            newdata = np.zeros_like(data, shape=(self.num_vertical,) + data.shape)
            newdata[...] = data
            data = newdata

        return FieldData(data.reshape(-1, 1))

    def periodic_planar_topology(self) -> FieldData[integer]:
        cells = [util.structured_cells(self.wrf_planar_cellshape, pardim=2)]

        nodemap = util.nodemap((self.num_latitude, 2), (self.num_longitude, self.num_longitude - 1))
        cells.append(util.structured_cells((self.num_latitude - 1, 1), pardim=2, nodemap=nodemap))

        south_pole_id = self.num_planar
        nodemap = util.nodemap((2, self.num_longitude + 1), (south_pole_id, 1), periodic=(1,))
        nodemap[1] = nodemap[1, 0]
        cells.append(util.structured_cells((1, self.num_longitude), pardim=2, nodemap=nodemap))

        north_pole_id = self.num_planar + 1
        nodemap = util.nodemap(
            (2, self.num_longitude + 1), (-self.num_longitude - 1, 1), periodic=(1,), init=north_pole_id
        )
        nodemap[0] = nodemap[0, 0]
        cells.append(util.structured_cells((1, self.num_longitude), pardim=2, nodemap=nodemap))

        return FieldData.join(cells)

    def periodic_volumetric_topology(self) -> FieldData[integer]:
        cells = [util.structured_cells(self.wrf_cellshape, pardim=3)]

        cells[0] += cells[0] // self.num_planar * 2
        num_horizontal = self.num_planar + 2

        nodemap = util.nodemap(
            (self.num_vertical, self.num_latitude, 2),
            (num_horizontal, self.num_longitude, self.num_longitude - 1),
        )
        cells.append(
            util.structured_cells(
                (self.num_vertical - 1, self.num_latitude - 1, 1), pardim=3, nodemap=nodemap
            )
        )

        south_pole_id = self.num_planar
        nodemap = util.nodemap(
            (self.num_vertical, 2, self.num_longitude + 1), (num_horizontal, south_pole_id, 1), periodic=(2,)
        )
        nodemap[:, 1] = (nodemap[:, 1] - south_pole_id) // num_horizontal * num_horizontal + south_pole_id
        cells.append(
            util.structured_cells((self.num_vertical - 1, 1, self.num_longitude), pardim=3, nodemap=nodemap)
        )

        north_pole_id = self.num_planar + 1
        nodemap = util.nodemap(
            (self.num_vertical, 2, self.num_longitude + 1),
            (num_horizontal, -self.num_longitude - 1, 1),
            periodic=(2,),
            init=north_pole_id,
        )
        nodemap[:, 0] = (nodemap[:, 0] - north_pole_id) // num_horizontal * num_horizontal + north_pole_id
        cells.append(
            util.structured_cells((self.num_vertical - 1, 1, self.num_longitude), pardim=3, nodemap=nodemap)
        )

        return FieldData.join(cells)

    def rotation(self, with_intrinsic: bool = True) -> Rotation:
        intrinsic = 0.0
        if with_intrinsic:
            intrinsic = 360 * np.ceil(self.num_longitude / 2) / self.num_longitude
        return Rotation.from_euler(
            "ZYZ", [-self.dataset.STAND_LON, -self.dataset.MOAD_CEN_LAT, intrinsic], degrees=True
        )

    def use_geometry(self, geometry: Field) -> None:
        self.geodetic = geometry.name == "Geodetic"

    def zones(self) -> Iterator[Zone]:
        corners = FieldData.concat(
            self.field_data_raw(
                self.longitude_name, 0, extrude_if_volumetric=False, include_poles_if_periodic=False
            ),
            self.field_data_raw(
                self.latitude_name, 0, extrude_if_volumetric=False, include_poles_if_periodic=False
            ),
        ).corners(self.wrf_planar_nodeshape)

        yield Zone(
            shape=Shape.Hexahedron,
            coords=corners,
            local_key="0",
        )

    def fields(self) -> Iterator[Field]:
        yield Field("Generic", type=api.Geometry(ncomps=3, coords=Generic()))

        yield Field(
            "Geodetic",
            type=api.Geometry(ncomps=3, coords=Geodetic(SphericalEarth(semi_major_axis=6370000.0))),
        )

        for variable in self.dataset.variables:
            if self.field_domain(variable) in self.valid_domains:
                yield Field(variable, type=api.Scalar())

    def topology(self, timestep: TimeStep, field: Field, zone: Zone) -> DiscreteTopology:
        if self.periodic:
            num_nodes = self.num_planar + 2
            if self.volumetric:
                num_nodes *= self.num_vertical
                return UnstructuredTopology(
                    num_nodes, self.periodic_volumetric_topology(), celltype=CellType.Hexahedron
                )
            else:
                return UnstructuredTopology(
                    num_nodes, self.periodic_planar_topology(), celltype=CellType.Quadrilateral
                )
        else:
            celltype = CellType.Hexahedron if self.volumetric else CellType.Quadrilateral
            return StructuredTopology(self.wrf_cellshape, celltype)

    def field_data(self, timestep: TimeStep, field: Field, zone: Zone) -> FieldData[floating]:
        if not field.is_geometry and not field.is_vector:
            return self.field_data_raw(field.name, timestep.index)
        assert field.is_geometry
        return self.geometry(timestep.index)

    def height(self, index: int) -> FieldData[floating]:
        if self.volumetric:
            return (self.field_data_raw("PH", index) + self.field_data_raw("PHB", index)) / 9.81
        return self.field_data_raw(self.height_name, 0)

    @lru_cache(maxsize=1)
    def geometry(self, index: int) -> FieldData[floating]:
        if self.geodetic:
            return FieldData.concat(
                self.field_data_raw(self.longitude_name, index),
                self.field_data_raw(self.latitude_name, index),
                self.height(index),
            )

        x = np.zeros(self.wrf_nodeshape, dtype=float)
        y = np.zeros(self.wrf_nodeshape, dtype=float)
        x[...] = np.arange(self.num_longitude)[..., np.newaxis, :] * self.dataset.DX
        y[...] = np.arange(self.num_latitude)[..., :, np.newaxis] * self.dataset.DY

        return FieldData.concat(
            FieldData(x.reshape(-1, 1)),
            FieldData(y.reshape(-1, 1)),
            self.height(index),
        )


class Wrf(NetCdf):
    longitude_name = "XLONG"
    latitude_name = "XLAT"
    height_name = "HGT"

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with Dataset(path, "r") as f:
                assert "WRF" in f.TITLE
            return True
        except (AssertionError, OSError):
            return False

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=False,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        self.volumetric = settings.dimensionality.out_is_volumetric()

        self.valid_domains = (FieldDimensionality.Volumetric,)
        if settings.dimensionality.in_allows_planar():
            self.valid_domains += (FieldDimensionality.Planar,)

        self.staggering = settings.staggering
        self.periodic = settings.periodic

    def timesteps(self) -> Iterator[TimeStep]:
        for index in range(self.num_timesteps):
            time = self.dataset["XTIME"][index] * 60
            yield TimeStep(index=index, time=time)

    def fields(self) -> Iterator[Field]:
        yield from super().fields()
        yield Field("WIND", type=api.Vector(3, api.VectorInterpretation.Flow), splittable=False)

    def field_data(self, timestep: TimeStep, field: Field, zone: Zone) -> FieldData[floating]:
        if field.name == "WIND":
            return self.wind(timestep.index)
        return super().field_data(timestep, field, zone)

    @lru_cache(maxsize=1)
    def wind(self, index: int) -> FieldData[floating]:
        local = FieldData.concat(
            self.field_data_raw("U", index, include_poles_if_periodic=False),
            self.field_data_raw("V", index, include_poles_if_periodic=False),
            self.field_data_raw("W", index, include_poles_if_periodic=False),
        )

        if not self.geodetic:
            return local

        lonlat = FieldData.concat(
            self.field_data_raw(self.longitude_name, index, include_poles_if_periodic=False),
            self.field_data_raw(self.latitude_name, index, include_poles_if_periodic=False),
        )

        points = (
            lonlat.spherical_to_cartesian()
            .rotate(self.rotation(with_intrinsic=True).inv())
            .cartesian_to_spherical(with_radius=False)
        )

        vectors = local.spherical_to_cartesian_vector_field(points).numpy(-1, *self.wrf_planar_nodeshape)

        south: Optional[np.ndarray] = None
        north: Optional[np.ndarray] = None
        if self.periodic:
            south = np.mean(vectors[:, 0, ...], axis=-2)[:, np.newaxis, :]
            north = np.mean(vectors[:, -1, ...], axis=-2)[:, np.newaxis, :]

        vectors = vectors.reshape(-1, self.num_planar, 3)

        if self.periodic:
            assert south is not None
            assert north is not None
            vectors = np.append(vectors, south, axis=1)
            vectors = np.append(vectors, north, axis=1)

        vectors = vectors.reshape(-1, 3)

        lonlat = FieldData.concat(
            self.field_data_raw(self.longitude_name, index),
            self.field_data_raw(self.latitude_name, index),
        )

        return FieldData(vectors).rotate(self.rotation()).cartesian_to_spherical_vector_field(lonlat)


class GeoGrid(NetCdf):
    longitude_name = "XLONG_M"
    latitude_name = "XLAT_M"
    height_name = "HGT_M"

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with Dataset(path, "r") as f:
                assert "GEOGRID" in f.TITLE
            return True
        except (AssertionError, OSError):
            return False

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=True,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        self.volumetric = False
        self.valid_domains = (FieldDimensionality.Planar,)
        self.staggering = settings.staggering
        self.periodic = settings.periodic

    def timesteps(self) -> Iterator[TimeStep]:
        yield TimeStep(index=0, time=0.0)
