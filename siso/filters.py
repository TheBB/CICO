from abc import ABC, abstractmethod
from contextlib import contextmanager

from . import config
from .coords import Coords
from .fields import Field, CombinedField, SimpleField, SourcedField, PatchData, FieldData, FieldPatches
from .geometry import GeometryManager, Patch

from .typing import StepData, Array2D
from typing import ContextManager, Iterable, Tuple, Optional

import numpy as np



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
