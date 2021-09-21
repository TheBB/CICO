from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

from . import config
from .coords import Coords, Converter, graph, CoordinateConversionError, Local
from .fields import Field, CombinedField, SimpleField, SourcedField, PatchData, FieldData, FieldPatches, Geometry
from .geometry import GeometryManager, Patch, PatchKey, UnstructuredTopology

from .typing import StepData, Array2D
from typing import ContextManager, Iterable, Tuple, Optional, Dict

import numpy as np
import treelog as log



# Abstract base classes
# ----------------------------------------------------------------------


class Sink(ABC):

    @abstractmethod
    def step(self, stepdata: StepData) -> ContextManager['StepFilter']:
        pass


class StepSink(ABC):

    @abstractmethod
    def geometry(self, field: Field) -> ContextManager['FieldFilter']:
        pass

    @abstractmethod
    def field(self, field: Field) -> ContextManager['FieldFilter']:
        pass


class FieldSink(ABC):

    @abstractmethod
    def __call__(self, patch: PatchData, data: FieldData):
        pass


class Source(ABC):

    @abstractmethod
    def steps(self) -> Iterable[Tuple[int, StepData]]:
        """Iterate over all steps with associated data."""
        pass

    @abstractmethod
    def fields(self) -> Iterable[Field]:
        """Iterate over all fields."""
        pass



# LastStep
# ----------------------------------------------------------------------


class LastStepFilter(Source):

    src: Source

    def __init__(self, src: Source):
        """Filter that only returns the last timestep."""
        self.src = src
        config.require(multiple_timesteps=False)

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        for step in self.src.steps():
            pass
        yield step

    def fields(self) -> Iterable[Field]:
        yield from self.src.fields()



# StepSlice
# ----------------------------------------------------------------------


class StepSliceFilter(Source):

    src: Source
    params: Tuple[Optional[int]]

    def __init__(self, src: Source, *args: Optional[int]):
        """Filter that acts as a time array slice."""
        self.src = src
        self.params = args

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        for stepid, stepdata in islice(self.src.steps(), *self.params):
            yield stepid, stepdata

    def fields(self) -> Iterable[Field]:
        yield from self.src.fields()



# Tesselator
# ----------------------------------------------------------------------


class TesselatorFilter(Source):

    src: Source
    manager: GeometryManager

    def __init__(self, src: Source):
        """Filter that tesselates all patches, providing either structured or
        unstructured output.  It also enumerates patches and
        guarantees integer patch keys.
        """
        self.src = src
        self.manager = GeometryManager()

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield from self.src.steps()

    def fields(self) -> Iterable[Field]:
        for field in self.src.fields():
            yield TesselatedField(field, self.manager)


class TesselatedField(SourcedField):

    manager: GeometryManager

    def __init__(self, src: Field, manager: GeometryManager):
        self.src = src
        self.manager = manager

    def decompositions(self) -> Iterable[Field]:
        for field in self.src.decompositions():
            yield TesselatedField(field, self.manager)

    def patches(self, stepid: int, force: bool = False, coords: Optional[Coords] = None) -> FieldPatches:
        for patchdata, fielddata in self.src.patches(stepid, force=force, coords=coords):
            key = patchdata.key if isinstance(patchdata, Patch) else patchdata[0].key

            if self.is_geometry:
                patchid = self.manager.update(key, fielddata)
                topo = patchdata.topology.tesselate()
                data = patchdata.topology.tesselate_field(fielddata)
                yield Patch((patchid,), topo), data

            elif isinstance(self.src, SimpleField):
                patchid = self.manager.global_id(key)
                yield Patch((patchid,)), patchdata.topology.tesselate_field(fielddata, cells=self.cells)

            elif isinstance(self.src, CombinedField):
                patchid = self.manager.global_id(key)
                data = np.hstack([p.topology.tesselate_field(d, cells=self.cells) for p, d in zip(patchdata, fielddata)])
                yield Patch((patchid,)), data

            else:
                raise TypeError(f"Unable to find corresponding geometry patch in field {self.name}")



# MergeTopologies
# ----------------------------------------------------------------------


class MergeTopologiesFilter(Source):

    src: Source
    indices: Dict[PatchKey, Tuple[int, int]]
    last_step: int
    next_index: int
    topology: Optional[UnstructuredTopology]

    def __init__(self, src: Source):
        """Filter that combines all topologies into one."""
        self.src = src
        self.indices = dict()
        self.last_step = -1
        self.next_index = 0
        self.topology = None

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield from self.src.steps()

    def fields(self) -> Iterable[Field]:
        for field in self.src.fields():
            yield MergeTopologiesField(field, self)

    def set_indices(self, stepid: int, key: PatchKey, nnodes: int):
        if stepid != self.last_step:
            self.indices.clear()
            self.last_step = stepid
            self.next_index = 0

        self.indices[key] = (self.next_index, self.next_index + nnodes)
        self.next_index += nnodes

    def get_indices(self, patches: Iterable[Patch]) -> Iterable[Tuple[int, int]]:
        return [self.indices[patch.key] for patch in patches]


class MergeTopologiesField(SourcedField):

    manager: MergeTopologiesFilter

    def __init__(self, src: Source, manager: MergeTopologiesFilter):
        self.src = src
        self.manager = manager

    def decompositions(self):
        for field in self.src.decompositions():
            yield MergeTopologiesField(field, self.manager)

    def patches(self, stepid: int, force: bool = False, coords: Optional[Coords] = None) -> FieldPatches:
        # TODO: Find a way to get this information without the data
        if self.is_geometry:
            topo = None
            for patch, data in self.src.patches(stepid, force=force, coords=coords):
                self.manager.set_indices(stepid, patch.key, len(data))
                if topo is None:
                    topo = patch.topology
                else:
                    assert isinstance(topo, UnstructuredTopology)
                    assert isinstance(patch.topology, UnstructuredTopology)
                    topo = UnstructuredTopology.join(topo, patch.topology)
            if topo is not None:
                self.manager.topology = topo

        nnodes = self.manager.next_index
        total_data = None
        for patch, data in self.src.patches(stepid, force=force, coords=coords):
            if total_data is None:
                total_data = data
            else:
                total_data = np.vstack((total_data, data))

        if total_data is not None:
            yield Patch((0,), self.manager.topology), total_data



# CoordinateTransform
# ----------------------------------------------------------------------


class CoordinateTransformFilter(Source):

    src: Source
    target: Coords
    converter: Optional[Converter]
    source_coords: Optional[Coords]

    def __init__(self, src: Source, target: Coords):
        self.src = src
        self.target = target
        self.converter = None

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield from self.src.steps()

    def fields(self) -> Iterable[Field]:
        for fld in self.src.fields():
            if fld.is_geometry:
                try:
                    converter = graph.path(fld.coords, self.target)
                    path = ' -> '.join(str(k) for k in [fld.coords, *converter.path[1:-1], self.target])
                    log.debug(f"Coordinate conversion path: {path}")
                    yield CoordinateTransformGeometryField(fld, self)
                except CoordinateConversionError:
                    log.warning(f"Skipping {fld.name}: {fld.coords} not convertable to {self.target}")
                    continue
            else:
                yield CoordinateTransformField(fld, self)


class CoordinateTransformGeometryField(SourcedField):

    manager: CoordinateTransformFilter

    def __init__(self, src: Field, manager: CoordinateTransformFilter):
        self.src = src
        self.manager = manager
        if isinstance(self.manager.target, Local) and isinstance(self.src.coords, Local):
            target = self.src.coords
        else:
            target = self.manager.target
        self._fieldtype = Geometry(coords=target)

    def patches(self, stepid: int, force: bool = False, coords: Optional[Coords] = None) -> FieldPatches:
        if self.manager.converter is None:
            self.manager.converter = graph.path(self.src.coords, self.manager.target)
            self.manager.source_coords = self.src.coords

        conv = self.manager.converter
        for patch, data in self.src.patches(stepid, force=force, coords=coords):
            yield patch, conv.points(self.src.coords, self.coords, data, patch.key)


class CoordinateTransformField(SourcedField):

    manager: CoordinateTransformFilter

    def __init__(self, src: Field, manager: CoordinateTransformFilter):
        self.src = src
        self.manager = manager

    def patches(self, stepid: int, force: bool = False, coords: Optional[Coords] = None) -> FieldPatches:
        conv = self.manager.converter
        for patch, data in self.src.patches(stepid, force=force, coords=coords):
            if self.is_vector:
                yield patch, conv.vectors(self.manager.source_coords, self.manager.target, data, patch.key)
            else:
                yield patch, data



# Offset
# ----------------------------------------------------------------------


class OffsetFilter(Source):

    src: Source
    path: Path

    def __init__(self, src: Source, path: Path):
        self.src = src
        self.path = path

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield from self.src.steps()

    def fields(self) -> Iterable[Field]:
        for fld in self.src.fields():
            if fld.is_geometry:
                if not fld.coords.is_affine:
                    continue
                yield OffsetGeometryField(fld, self.path)
            else:
                yield fld


class OffsetGeometryField(SourcedField):

    x: float
    y: float

    def __init__(self, src: Field, path: Path):
        self.src = src
        with open(path, 'r') as f:
            self.x, self.y = map(float, next(f).split())

    def patches(self, stepid: int, force: bool = False, coords: Optional[Coords] = None) -> FieldPatches:
        for patch, data in self.src.patches(stepid, force=force, coords=coords):
            data[:,0] += self.x
            data[:,1] += self.y
            yield patch, data
