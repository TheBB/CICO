from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import chain, count, repeat
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
from attrs import define, field
from numpy import floating
from typing_extensions import Self

from .. import api, util
from ..api import Shape, Zone
from ..coord import Named
from ..impl import Basis, Field, Step
from ..topology import LrTopology, SplineTopology, UnstructuredTopology
from ..util import FieldData


class Locator(ABC):
    @abstractmethod
    def patch_root(self, name: str, step: int) -> str:
        ...

    def patch_path(self, name: str, step: int, patch: int) -> str:
        return f"{self.patch_root(name, step)}/{patch + 1}"

    @abstractmethod
    def coeff_root(self, basis_name: str, field_name: str, step: int, cellwise: bool) -> str:
        ...

    def coeff_path(self, basis_name: str, field_name: str, step: int, patch: int, cellwise: bool) -> str:
        return f"{self.coeff_root(basis_name, field_name, step, cellwise)}/{patch + 1}"


class StandardLocator(Locator):
    def patch_root(self, name: str, step: int) -> str:
        return f"{step}/{name}/basis"

    def coeff_root(self, basis_name: str, field_name: str, step: int, cellwise: bool) -> str:
        subdir = "knotspan" if cellwise else "fields"
        return f"{int(step)}/{basis_name}/{subdir}/{field_name}"


class EigenLocator(Locator):
    def patch_root(self, name: str, step: int) -> str:
        return f"0/{name}/basis"

    def coeff_root(self, basis_name: str, field_name: str, step: int, cellwise: bool) -> str:
        return f"0/{basis_name}/Eigenmode/{step+1}"


def is_legal_group_name(name: str) -> bool:
    try:
        int(name)
        return True
    except ValueError:
        return name.casefold() in ("anasol", "log")


@define(frozen=True)
class IfemBasis(Basis):
    locator: Locator = field(eq=False, repr=False)
    fields: List[IfemField] = field(factory=list, init=False, eq=False, repr=False)

    @lru_cache(maxsize=1)
    def num_patches(self, source: Ifem) -> int:
        i = 0
        for i in count():
            if self.patch_path(0, i) not in source.h5:
                break
        return i

    def updates_at(self, step: int, source: Ifem) -> bool:
        return self.locator.patch_root(self.name, step) in source.h5

    def last_update_before(self, step: int, source: Ifem) -> int:
        while not self.updates_at(step, source):
            step -= 1
        return step

    def patch_path(self, step: int, patch: int) -> str:
        return self.locator.patch_path(self.name, step, patch)

    @lru_cache(maxsize=128)
    def patch_at(self, step: int, patch: int, source: Ifem) -> Tuple[Zone, api.Topology, FieldData[floating]]:
        step = self.last_update_before(step, source)
        topology: api.Topology
        patchdata = source.h5[self.patch_path(step, patch)][:]
        initial = patchdata[:20].tobytes()
        raw_data = memoryview(patchdata).tobytes()
        if initial.startswith(b"# LAGRANGIAN"):
            corners, topology, cps = UnstructuredTopology.from_ifem(raw_data)
        elif initial.startswith(b"# LRSPLINE"):
            corners, topology, cps = next(LrTopology.from_bytes(raw_data, source.rationality))
        else:
            corners, topology, cps = next(SplineTopology.from_bytes(raw_data))
        shape = [Shape.Line, Shape.Quatrilateral, Shape.Hexahedron][topology.pardim - 1]

        zone = Zone(shape=shape, coords=corners, local_key=f"{self.name}/{step}/{patch}")
        return zone, topology, cps

    @lru_cache(maxsize=1)
    def ncomps(self, source: Ifem) -> int:
        _, _, cps = self.patch_at(0, 0, source)
        return cps.ncomps


@define(frozen=True)
class IfemField(Field):
    basis: IfemBasis = field(kw_only=True, eq=False)

    def splits(self) -> Iterator[api.SplitFieldSpec]:
        if not self.splittable or "&&" not in self.name:
            return

        component_names = [comp.strip() for comp in self.name.split("&&")]
        if " " in component_names[0]:
            prefix, component_names[0] = component_names[0].split(" ", maxsplit=1)
            component_names = [f"{prefix} {comp}" for comp in component_names]

        for i, comp in enumerate(component_names):
            yield api.SplitFieldSpec(
                source_name=self.name,
                new_name=comp,
                components=[i],
                destroy=True,
            )

    @abstractmethod
    def cps_at(self, step: int, patch: int, source: Ifem) -> FieldData[floating]:
        ...

    @abstractmethod
    def updates_at(self, step: int, source: Ifem) -> bool:
        ...


class IfemGeometryField(IfemField):
    def __init__(self, basis: IfemBasis, source: Ifem):
        super().__init__(
            name=basis.name,
            type=api.Geometry(basis.ncomps(source), coords=Named(basis.name)),
            splittable=False,
            basis=basis,
        )

    def cps_at(self, step: int, patch: int, source: Ifem) -> FieldData[floating]:
        _, _, cps = self.basis.patch_at(step, patch, source)
        return cps

    def updates_at(self, step: int, source: Ifem) -> bool:
        return self.basis.updates_at(step, source)


class IfemStandardField(IfemField):
    def __init__(
        self,
        name: str,
        cellwise: bool,
        basis: IfemBasis,
        source: Ifem,
    ):
        _, topology, _ = basis.patch_at(0, 0, source)
        coeff_path = basis.locator.coeff_path(basis.name, name, 0, 0, cellwise)
        cps = source.h5[coeff_path][:]
        divisor = topology.num_cells if cellwise else topology.num_nodes
        ncomps, remainder = divmod(len(cps), divisor)
        assert remainder == 0

        tp: api.FieldType
        if ncomps > 1:
            tp = api.Vector(ncomps=ncomps, interpretation=source.default_vector)
        else:
            tp = api.Scalar(interpretation=source.default_scalar)

        super().__init__(name, type=tp, cellwise=cellwise, basis=basis)
        basis.fields.append(self)

    def __repr__(self) -> str:
        return f"Field({self.name}, {'cellwise' if self.cellwise else 'nodal'}, {self.basis.name})"

    def updates_at(self, step: int, source: Ifem) -> bool:
        return self.coeff_root(step) in source.h5

    def last_update_before(self, step: int, source: Ifem) -> int:
        while not self.updates_at(step, source):
            step -= 1
        return step

    def coeff_root(self, step: int) -> str:
        return self.basis.locator.coeff_root(
            self.basis.name,
            self.name,
            step,
            self.cellwise,
        )

    def coeff_path(self, step: int, patch: int) -> str:
        return self.basis.locator.coeff_path(
            self.basis.name,
            self.name,
            step,
            patch,
            self.cellwise,
        )

    # @lru_cache(maxsize=1)
    # def raw_ncomps(self, source: Ifem) -> int:
    #     _, topology, _ = self.basis.patch_at(0, 0, source)
    #     my_cps = self.raw_cps_at(0, 0, source)
    #     divisor = topology.num_cells if self.cellwise else topology.num_nodes
    #     ncomps, remainder = divmod(len(my_cps), divisor)
    #     assert remainder == 0
    #     return ncomps

    @lru_cache(maxsize=8)
    def raw_cps_at(self, step: int, patch: int, source: Ifem) -> np.ndarray:
        return source.h5[self.coeff_path(step, patch)][:]

    def cps_at(self, step: int, patch: int, source: Ifem) -> FieldData[floating]:
        # ncomps = self.raw_ncomps(source)
        cps = self.raw_cps_at(step, patch, source)
        return FieldData(data=cps.reshape(-1, self.ncomps))


class Ifem(api.Source[IfemBasis, IfemField, Step, Zone]):
    filename: Path
    h5: h5py.File

    geometry: IfemBasis

    _bases: Dict[str, IfemBasis]
    _fields: Dict[str, IfemField]

    locator: ClassVar[Locator] = StandardLocator()
    default_scalar: ClassVar[api.ScalarInterpretation] = api.ScalarInterpretation.Generic
    default_vector: ClassVar[api.VectorInterpretation] = api.VectorInterpretation.Generic

    rationality: Optional[api.Rationality] = None

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with h5py.File(path, "r") as f:
                assert all(is_legal_group_name(name) for name in f)
            return True
        except (AssertionError, OSError):
            return False

    def __init__(self, filename: Path):
        self.filename = filename
        self._fields = {}

    def __enter__(self) -> Self:
        self.h5 = h5py.File(self.filename, "r").__enter__()

        self.discover_bases()
        for basis in self._bases.values():
            logging.debug(
                f"Basis {basis.name} with " f"{util.pluralize(basis.num_patches(self), 'patch', 'patches')}"
            )

        self.discover_fields()
        return self

    def __exit__(self, *args) -> None:
        self.h5.__exit__(*args)

    @property
    def properties(self) -> api.SourceProperties:
        splits, recombineations = self.propose_recombinations()
        return api.SourceProperties(
            instantaneous=False,
            split_fields=splits,
            recombine_fields=recombineations,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        self.rationality = settings.rationality

    @property
    def nsteps(self) -> int:
        return len(self.h5)

    def timestep_groups(self) -> Iterator[h5py.Group]:
        for index in range(self.nsteps):
            yield self.h5[str(index)]

    def make_basis(self, name: str) -> IfemBasis:
        return IfemBasis(name, self.locator)

    def discover_bases(self) -> None:
        basis_names = dict.fromkeys(
            name for name in chain.from_iterable(self.h5.values()) if name != "timeinfo"
        )
        bases = (self.make_basis(name) for name in basis_names)
        self._bases = {basis.name: basis for basis in bases if basis.num_patches(self) > 0}

    def discover_fields(self) -> None:
        for step_grp in self.timestep_groups():
            for basis_name, basis_grp in step_grp.items():
                if basis_name not in self._bases:
                    continue

                fields: Iterator[Tuple[str, bool]] = chain(
                    zip(basis_grp.get("fields", ()), repeat(False)),
                    zip(basis_grp.get("knotspan", ()), repeat(True)),
                )
                for field_name, cellwise in fields:
                    if field_name in self._fields:
                        continue
                    self._fields[field_name] = IfemStandardField(
                        name=field_name,
                        cellwise=cellwise,
                        basis=self._bases[basis_name],
                        source=self,
                    )

    @lru_cache(maxsize=1)
    def propose_recombinations(self) -> Tuple[List[api.SplitFieldSpec], List[api.RecombineFieldSpec]]:
        splits = list(chain.from_iterable(field.splits() for field in self._fields.values()))

        candidates: Dict[str, List[str]] = {}
        field_names = chain(self._fields, (split.new_name for split in splits))
        for field_name in field_names:
            if len(field_name) <= 2 or field_name[-2] != "_":
                continue
            prefix, suffix = field_name[:-2], field_name[-1]
            if suffix not in "xyz":
                continue
            candidates.setdefault(prefix, []).append(field_name)

        recombinations = [
            api.RecombineFieldSpec(source_names, new_name)
            for new_name, source_names in candidates.items()
            if new_name not in self._fields and len(source_names) > 1
        ]

        return splits, recombinations

    def use_geometry(self, geometry: Field) -> None:
        self.geometry = self._bases[geometry.name]

    def steps(self) -> Iterator[Step]:
        for i, group in enumerate(self.timestep_groups()):
            if "timeinfo/level" in group:
                time = group["timeinfo/level"][0]
            else:
                time = float(i)
            yield Step(index=i, value=time)

    def zones(self) -> Iterator[Zone]:
        for patch in range(self.geometry.num_patches(self)):
            zone, _, _ = self.geometry.patch_at(0, patch, self)
            yield zone

    def bases(self) -> Iterator[IfemBasis]:
        return iter(self._bases.values())

    def basis_of(self, field: IfemField) -> IfemBasis:
        return field.basis

    def geometries(self, basis: IfemBasis) -> Iterator[IfemField]:
        yield IfemGeometryField(basis, self)

    def fields(self, basis: IfemBasis) -> Iterator[IfemField]:
        return iter(basis.fields)

    def topology(self, timestep: Step, basis: Basis, zone: Zone) -> api.Topology:
        ibasis = self._bases[basis.name]
        patch = int(zone.local_key.split("/")[-1])
        _, topology, _ = ibasis.patch_at(timestep.index, patch, self)
        return topology

    def field_data(self, timestep: Step, field: Field, zone: Zone) -> FieldData[floating]:
        patch = int(zone.local_key.split("/")[-1])
        if field.is_geometry:
            basis = self._bases[field.name]
            _, _, coeffs = basis.patch_at(timestep.index, patch, self)
            return coeffs
        ifield = self._fields[field.name]
        coeffs = ifield.cps_at(timestep.index, patch, self)
        return coeffs

    def field_updates(self, timestep: Step, field: Field) -> bool:
        if field.is_geometry:
            basis = self._bases[field.name]
            return basis.updates_at(timestep.index, self)
        return self._fields[field.name].updates_at(timestep.index, self)


class IfemModes(Ifem):
    locator = EigenLocator()

    default_scalar = api.ScalarInterpretation.Eigenmode
    default_vector = api.VectorInterpretation.Eigenmode

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with h5py.File(path, "r") as f:
                assert "0" in f
                basis_name = next(iter(f["0"]))
                assert "Eigenmode" in f[f"0/{basis_name}"]
            return True
        except (AssertionError, OSError):
            return False

    @property
    def properties(self) -> api.SourceProperties:
        basis = next(iter(self._bases.values()))
        group = self.h5[f"0/{basis.name}/Eigenmode/1"]
        if "Value" in group:
            step = api.StepInterpretation.Eigenmode
        else:
            assert "Frequency" in group
            step = api.StepInterpretation.EigenFrequency
        return super().properties.update(step_interpretation=step)

    def discover_bases(self) -> None:
        group = self.h5["0"]
        basis_name = util.only(group)
        self._bases = {basis_name: self.make_basis(basis_name)}

    def discover_fields(self) -> None:
        basis = util.only(self._bases.values())
        self._fields = {
            "Mode Shape": IfemStandardField("Mode Shape", cellwise=False, basis=basis, source=self)
        }

    @property
    def nsteps(self) -> int:
        basis = util.only(self._bases.values())
        return len(self.h5[f"0/{basis.name}/Eigenmode"])

    def timestep_groups(self) -> Iterator[h5py.Group]:
        basis = util.only(self._bases.values())
        for index in range(self.nsteps):
            yield self.h5[f"0/{basis.name}/Eigenmode/{index+1}"]

    def steps(self) -> Iterator[Step]:
        for i, group in enumerate(self.timestep_groups()):
            if "Value" in group:
                time = group["Value"][0]
            else:
                time = group["Frequency"][0]
            yield Step(index=i, value=time)

    def field_updates(self, timestep: Step, field: Field) -> bool:
        if field.is_geometry:
            return timestep.index == 0
        return super().field_updates(timestep, field)
