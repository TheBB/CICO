from abc import abstractmethod
from pathlib import Path
import re
import sys

import f90nml
import numpy as np
from scipy.io import FortranFile, FortranFormattingError
import treelog as log

from typing import Optional, Iterable, Tuple, TextIO, Any, Dict
from ..typing import StepData, Array2D, Shape

from .reader import Reader
from .. import config, ConfigTarget
from ..coords import Local
from ..fields import Field, SimpleField, Geometry, FieldPatches
from ..geometry import Topology, UnstructuredTopology, StructuredTopology, Hex, Quad, Patch
from ..util import fortran_skip_record, save_excursion, cache, prod
from ..writer import Writer



# Utilities
# ----------------------------------------------------------------------


def read_many(lines, n, tp, skip=True):
    if skip:
        next(lines)
    values = []
    while len(values) < n:
        values.extend(map(tp, next(lines).split()))
    return np.array(values)


def split_sparse(values, ncomps=1):
    jump = ncomps + 1
    indexes = values[::jump].astype(int)
    comps = [values[i::jump] for i in range(1, ncomps+1)]
    return (indexes, *comps)


def make_mask(n, indices, values=1.0):
    retval = np.zeros((n,))
    retval[indices-1] = values
    return retval


def dtypes(endianness):
    endian = {'native': '=', 'big': '>', 'small': '<'}[endianness]
    return np.dtype(f'{endian}f4'), np.dtype(f'{endian}u4')


def transpose(array, nodeshape):
    if config.fix_orientation:
        return array.reshape(*nodeshape, -1).transpose(1, 0, 2, 3).reshape(prod(nodeshape), -1)
    return array.reshape(prod(nodeshape), -1)


def ensure_native(data: np.ndarray) -> np.ndarray:
    if data.dtype.byteorder in ('=', sys.byteorder):
        return data
    return data.byteswap().newbyteorder()



# Fields
# ----------------------------------------------------------------------


class SIMRAField(SimpleField):

    decompose = True

    index: int
    reader: 'SIMRAResultReader'
    scale: float

    def __init__(self, name: str, index: int, ncomps: int, reader: 'SIMRAReader', scale: float = 1.0, cells: bool = False):
        self.name = name
        self.index = index
        self.ncomps = ncomps
        self.reader = reader
        self.scale = scale
        self.cells = cells

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        data = self.reader.data(stepid)[int(self.cells)]
        yield (
            self.reader.patch(),
            data[:, self.index : self.index + self.ncomps] * self.scale,
        )


class SIMRAGeometryField(SimpleField):

    name = 'Geometry'
    cells = False
    ncomps = 3

    reader: 'SIMRAReader'

    def __init__(self, reader: 'SIMRAReader'):
        self.reader = reader
        self.fieldtype = Geometry(Local().substitute())

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        yield self.reader.patch(), self.reader.nodes()



# Abstract reader
# ----------------------------------------------------------------------


class SIMRAReader(Reader):

    f4_type: np.dtype
    u4_type: np.dtype

    def validate(self):
        super().validate()
        config.ensure_limited(
            ConfigTarget.Reader,
            'input_endianness', 'mesh_file', 'fix_orientation',
            reason="not supported by SIMRA"
        )

    def __init__(self):
        self.f4_type, self.u4_type = dtypes(config.input_endianness)

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield (0, {'time': 0.0})

    @abstractmethod
    def patch(self) -> Topology:
        pass

    @abstractmethod
    def nodes(self) -> Array2D:
        pass

    def fields(self) -> Iterable[Field]:
        yield SIMRAGeometryField(self)



# Concrete readers
# ----------------------------------------------------------------------


class SIMRAMeshReader(SIMRAReader):

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason="SIMRA files do not support multiple timesteps")
        config.require(offset_should_exist=True)


class SIMRA2DMapReader(SIMRAMeshReader):

    reader_name = "SIMRA-map"

    filename: Path
    mapfile: TextIO
    shape: Tuple[int, int]

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        # The first line should have two right-aligned positive
        # integers, exactly eight characters each
        try:
            with open(filename, 'r') as f:
                line = next(f)
            assert len(line) == 17
            assert re.match(r'[ ]*[0-9]+', line[:8])
            assert re.match(r'[ ]*[0-9]+', line[8:16])
            assert line[-1] == '\n'
            return True
        except:
            return False

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def __enter__(self):
        self.mapfile = open(self.filename, 'r').__enter__()
        with save_excursion(self.mapfile):
            self.shape = tuple(int(s) - 1 for s in next(self.mapfile).split())
        return self

    def __exit__(self, *args):
        return self.mapfile.__exit__(*args)

    @property
    def nodeshape(self):
        return tuple(s+1 for s in self.shape)

    def patch(self):
        return Patch(('geometry',), StructuredTopology(self.shape[::-1], celltype=Quad()))

    def nodes(self):
        nodes = []
        with save_excursion(self.mapfile):
            next(self.mapfile)
            for line in self.mapfile:
                nodes.extend(map(float, line.split()))
        nodes = np.array(nodes).reshape(*self.nodeshape[::-1], 3)
        nodes[...,2 ] /= 10      # Map files have a vertical resolution factor of 10
        return nodes.reshape(-1, 3)


class SIMRA2DMeshReader(SIMRAMeshReader):

    reader_name = "SIMRA-2D"

    filename: Path
    meshfile: TextIO
    shape: Tuple[int, int]

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        try:
            with open(filename, 'r') as f:
                assert next(f) == 'text\n'
            return True
        except:
            return False

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def __enter__(self):
        self.meshfile = open(self.filename, 'r').__enter__()
        next(self.meshfile)
        self.shape = tuple(int(s) - 1 for s in next(self.meshfile).split()[2:])
        return self

    def __exit__(self, *args):
        self.meshfile.__exit__(*args)

    @cache(1)
    def patch(self) -> Patch:
        return Patch(('geometry',), StructuredTopology(self.shape[::-1], celltype=Quad()))

    @cache(1)
    def nodes(self) -> Array2D:
        nnodes = prod(s+1 for s in self.shape)
        nodes = np.array([tuple(map(float, next(self.meshfile).split()[1:])) for _ in range(nnodes)])
        return nodes


class SIMRA3DMeshReader(SIMRAMeshReader):

    reader_name = "SIMRA-3D"

    filename: Path
    mesh: FortranFile
    nodeshape: Shape

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        _, u4_type = dtypes(config.input_endianness)
        try:
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                assert f._read_size() == 6 * 4
            return True
        except:
            return False

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def __enter__(self):
        self.mesh = FortranFile(self.filename, 'r', header_dtype=self.u4_type).__enter__()
        with save_excursion(self.mesh._fp):
            _, _, imax, jmax, kmax, _ = self.mesh.read_ints(self.u4_type)
        if config.fix_orientation:
            self.nodeshape = (jmax, imax, kmax)
        else:
            self.nodeshape = (imax, jmax, kmax)
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)

    @cache(1)
    def patch(self) -> Patch:
        i, j, k = self.nodeshape
        return Patch(('geometry',), StructuredTopology((j-1, i-1, k-1), celltype=Hex()))

    @cache(1)
    def nodes(self) -> Array2D:
        with save_excursion(self.mesh._fp):
            fortran_skip_record(self.mesh)
            nodes = transpose(self.mesh.read_reals(self.f4_type), self.nodeshape)
        nodes = ensure_native(nodes)
        return nodes


class SIMRABoundaryReader(SIMRAReader):

    reader_name = "SIMRA-Boun"

    data_fn: Path
    mesh_fn: Path

    io: TextIO

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        try:
            with open(filename, 'r') as f:
                assert next(f).startswith('Boundary conditions')
            assert SIMRA3DMeshReader.applicable(
                Path(config.mesh_file) if config.mesh_file
                else filename.with_name('mesh.dat')
            )
            return True
        except:
            return False

    def __init__(self, data_fn: Path, mesh_fn: Optional[Path] = None):
        self.data_fn = data_fn
        self.mesh_fn = Path(config.mesh_file) if config.mesh_file else self.data_fn.with_name('mesh.dat')

    def __enter__(self):
        self.io = open(self.data_fn, 'r').__enter__()
        self.mesh = SIMRA3DMeshReader(self.mesh_fn).__enter__()
        return self

    def __exit__(self, *args):
        self.io.__exit__(*args)
        self.mesh.__exit__(*args)

    def patch(self) -> Patch:
        return self.mesh.patch()

    def nodes(self) -> Array2D:
        return self.mesh.nodes()

    def fields(self) -> Iterable[Field]:
        yield from self.mesh.fields()

        yield SIMRAField('u', 0, 3, self)
        yield SIMRAField('p', 3, 1, self)
        yield SIMRAField('k', 4, 1, self)
        yield SIMRAField('eps', 5, 1, self)
        yield SIMRAField('pt', 6, 1, self)
        yield SIMRAField('surface-roughness', 7, 1, self)

        yield SIMRAField('u-mask', 8, 3, self)
        yield SIMRAField('p-mask', 11, 1, self)
        yield SIMRAField('wall-mask', 12, 1, self)
        yield SIMRAField('log-mask', 13, 1, self)
        yield SIMRAField('k,e-mask', 14, 1, self)
        yield SIMRAField('pt-mask', 15, 1, self)

    @cache(1)
    def data(self, stepid: int) -> Tuple[Array2D, Array2D]:
        self.io.seek(0)
        lines = iter(self.io)
        next(lines)

        *ints, z0 = next(lines).split()
        nfixu, nfixv, nfixw, nfixp, nfixe, nfixk, *rest, nlog = map(int, ints)

        parallel = bool(rest)
        if parallel:
            nwalle = rest[0]

        z0_var = read_many(lines, nlog, float, skip=False)
        ifix_z0 = np.arange(1, prod(self.mesh.nodeshape), self.mesh.nodeshape[-1])

        ifixu, fixu = split_sparse(read_many(lines, 2*nfixu, float))
        ifixv, fixv = split_sparse(read_many(lines, 2*nfixv, float))
        ifixw, fixw = split_sparse(read_many(lines, 2*nfixw, float))
        ifixp, fixp = split_sparse(read_many(lines, 2*nfixp, float))

        next(lines)

        if parallel:
            walle = read_many(lines, nwalle, int)

        t = read_many(lines, 2*nlog, int, skip=not parallel)
        iwall, ilog = t[::2], t[1::2]

        ifixk = read_many(lines, nfixk, int)
        t = read_many(lines, 2*nfixk, float, skip=False)
        fixk, fixd = t[::2], t[1::2]

        npts = prod(self.mesh.nodeshape)
        if parallel:
            read_many(lines, npts, float, skip=False)

        ifixtemp = read_many(lines, nfixe, int)
        fixtemp = read_many(lines, nfixe, float, skip=False)

        ndata = np.array([
            make_mask(npts, ifixu, fixu),
            make_mask(npts, ifixv, fixv),
            make_mask(npts, ifixw, fixw),
            make_mask(npts, ifixp, fixp),
            make_mask(npts, ifixk, fixk),
            make_mask(npts, ifixk, fixd),
            make_mask(npts, ifixtemp, fixtemp),
            make_mask(npts, ifix_z0, z0_var),

            make_mask(npts, ifixu),
            make_mask(npts, ifixv),
            make_mask(npts, ifixw),
            make_mask(npts, ifixp),
            make_mask(npts, iwall),
            make_mask(npts, ilog),
            make_mask(npts, ifixk),
            make_mask(npts, ifixtemp),
        ]).T
        return ensure_native(transpose(ndata, self.mesh.nodeshape)), None


class SIMRADataReader(SIMRAReader):

    result_fn: Path
    mesh_fn: Path
    input_fn: Path

    result: FortranFile
    mesh: SIMRA3DMeshReader
    input_data: Dict[str, Any]

    f4_type: np.dtype
    u4_type: np.dtype

    def __enter__(self):
        self.result = FortranFile(self.result_fn, 'r', header_dtype=self.u4_type).__enter__()
        self.mesh = SIMRA3DMeshReader(self.mesh_fn).__enter__()
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)
        self.result.__exit__(*args)

    def __init__(self, result_fn: Path):
        super().__init__()
        self.result_fn = Path(result_fn)
        self.mesh_fn = Path(config.mesh_file) if config.mesh_file else self.result_fn.with_name('mesh.dat')
        self.input_fn = self.result_fn.with_name('simra.in')

        if not self.mesh_fn.is_file():
            raise IOError(f"Unable to find mesh file: {self.mesh_fn}")

        if self.input_fn.is_file():
            self.input_data = f90nml.read(self.input_fn)
        else:
            self.input_data = {}
            log.warning(f"SIMRA input file not found, scales will be missing")

    def validate(self):
        self.mesh.validate()
        super().validate()

    def patch(self) -> Patch:
        return self.mesh.patch()

    def nodes(self) -> Array2D:
        return self.mesh.nodes()

    def scale(self, name: str) -> float:
        return self.input_data.get('param_data', {}).get(name, 1.0)

    def fields(self, apply_scales: bool = True) -> Iterable[Field]:
        yield from self.mesh.fields()

        uref = self.scale('uref') if apply_scales else 1.0
        lref = self.scale('lenref') if apply_scales else 1.0
        yield SIMRAField('u', 0, 3, self, scale=uref)
        yield SIMRAField('ps', 3, 1, self, scale=uref**2)
        yield SIMRAField('tk', 4, 1, self, scale=uref**2)
        yield SIMRAField('td', 5, 1, self, scale=uref**3/lref)
        yield SIMRAField('vtef', 6, 1, self, scale=uref*lref)
        yield SIMRAField('pt', 7, 1, self)
        yield SIMRAField('pts', 8, 1, self)
        yield SIMRAField('rho', 9, 1, self)
        yield SIMRAField('rhos', 10, 1, self)

        # yield SIMRAField('pressure', 0, 1, self, cells=True)

    @abstractmethod
    def data(self, stepid: int) -> Tuple[Array2D, Array2D]:
        pass


class SIMRAContinuationReader(SIMRADataReader):

    reader_name = "SIMRA-Cont"

    extra_field: Optional[bool] = None

    def __enter__(self):
        super().__enter__()

        # Check whether file contains strat
        with save_excursion(self.result._fp):
            size = self.result._read_size()
            self.result._fp.seek(size, 1)
            self.result._read_size()

            try:
                size = self.result._read_size()
                if size == prod(self.mesh.nodeshape) * self.f4_type.itemsize:
                    self.extra_field = 'strat'
                elif size == prod(c-1 for c in self.mesh.nodeshape) * self.f4_type.itemsize:
                    self.extra_field = '?'
            except FortranFormattingError:
                pass

        return self

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        _, u4_type = dtypes(config.input_endianness)
        try:
            # It's too easy to mistake other files for SIMRA results,
            # so we require a certain suffix
            assert filename.suffix == '.res' or filename.suffix == '.dat'
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize
                if filename.suffix == '.res':
                    assert (size // u4_type.itemsize - 1) % 11 == 0  # Eleven scalars per point plus a time
                elif filename.suffix == '.dat':
                    assert (size // u4_type.itemsize) % 11 == 0  # Eleven scalars per point
            assert SIMRA3DMeshReader.applicable(
                Path(config.mesh_file) if config.mesh_file
                else filename.with_name('mesh.dat')
            )
            return True
        except:
            return False

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason="SIMRA files do not support multiple timesteps")

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        # This is slightly hacky, but grabs the time value for the
        # next timestep without reading the whole dataset
        with save_excursion(self.result._fp):
            self.result._read_size()
            time = np.fromfile(self.result._fp, dtype=self.f4_type, count=1)[0]
        yield (0, {'time': time})

    def fields(self) -> Iterable[Field]:
        # Don't apply scaling on init.dat files
        apply_scales = self.result_fn.suffix == '.res'
        yield from super().fields(apply_scales=apply_scales)

        if self.result_fn.suffix == '.res':
            if self.extra_field == 'strat':
                yield SIMRAField('strat', 11, 1, self)
            elif self.extra_field == '?':
                yield SIMRAField('pressure?', 0, 1, self, cells=True)
        else:
            yield SIMRAField('pressure', 0, 1, self, cells=True)

    @cache(1)
    def data(self, stepid: int) -> Tuple[Array2D, Array2D]:
        ndata = self.result.read_reals(dtype=self.f4_type)
        if self.result_fn.suffix == '.res':
            _, ndata = ndata[0], ndata[1:]  # Strip away time

        cdata = None
        if self.result_fn.suffix == '.res':
            if self.extra_field == 'strat':
                sdata = self.result.read_reals(dtype=self.f4_type)
                ndata = np.hstack([ndata.reshape(-1, 11), sdata.reshape(-1, 1)])
            elif self.extra_field == '?':
                cdata = self.result.read_reals(dtype=self.f4_type)
                cdata = ensure_native(transpose(cdata, tuple(s-1 for s in self.mesh.nodeshape)))
        else:
            cdata = self.result.read_reals(dtype=self.f4_type)
            cdata = ensure_native(transpose(cdata, tuple(s-1 for s in self.mesh.nodeshape)))

        return (
            ensure_native(transpose(ndata, self.mesh.nodeshape)),
            cdata,
        )


class SIMRAHistoryReader(SIMRADataReader):

    reader_name = "SIMRA-Hist"

    cur_stepid: int

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        _, u4_type = dtypes(config.input_endianness)
        try:
            # It's too easy to mistake other files for SIMRA results,
            # so we require a certain suffix
            assert filename.suffix == '.res'
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                with save_excursion(f._fp):
                    size = f._read_size()
                    assert size == u4_type.itemsize
                assert f.read_ints(u4_type)[0] == u4_type.itemsize
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize
                assert (size // u4_type.itemsize - 1) % 12 == 0  # Twelve scalars per point plus a time
            assert SIMRA3DMeshReader.applicable(
                Path(config.mesh_file) if config.mesh_file
                else filename.with_name('mesh.dat')
            )
            return True
        except:
            return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_stepid = -1

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        # Skip the word size indicator
        self.result.read_ints(self.u4_type)

        while True:
            # This is slightly hacky, but grabs the time value for the
            # next timestep without reading the whole dataset
            with save_excursion(self.result._fp):
                try:
                    self.result._read_size()
                except FortranFormattingError:
                    return
                time = np.fromfile(self.result._fp, dtype=self.f4_type, count=1)[0]

            self.cur_stepid += 1
            yield (self.cur_stepid, {'time': time})

    def fields(self) -> Iterable[Field]:
        yield from super().fields()
        yield SIMRAField('pressure', 0, 1, self, cells=True)

    @cache(1)
    def data(self, stepid: int) -> Array2D:
        assert stepid == self.cur_stepid

        ndata = self.result.read_reals(dtype=self.f4_type)
        _, ndata = ndata[0], ndata[1:]  # Strip away time

        cdata = self.result.read_reals(dtype=self.f4_type)

        return (
            ensure_native(transpose(ndata, self.mesh.nodeshape)),
            ensure_native(transpose(cdata, tuple(s-1 for s in self.mesh.nodeshape))),
        )
