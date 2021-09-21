from operator import attrgetter
import treelog as log

from typing import Iterable, List, Tuple

from . import config
from .filters import (
    OffsetFilter, Source, LastStepFilter, StepSliceFilter, TesselatorFilter,
    MergeTopologiesFilter, CoordinateTransformFilter,
)
from .coords import Coords, Converter, graph
from .fields import Field, ComponentField
from .reader import Reader
from .writer import Writer


def discover_decompositions(fields: List[Field]) -> Iterable[Field]:
    for field in fields:
        yield field
        for subfield in field.decompositions():
            log.debug(f"Discovered decomposed scalar field '{subfield.name}'")
            yield subfield


def discover_fields(reader: Reader) -> Tuple[List[Field], List[Field]]:
    geometries, fields = [], []
    for field in reader.fields():
        if field.is_geometry:
            geometries.append(field)
            continue
        if config.field_filter is not None and field.name.lower() not in config.field_filter:
            continue
        fields.append(field)

    for field in fields:
        log.debug(f"Discovered field '{field.name}' with {field.ncomps} component(s)")

    fields = sorted(fields, key=attrgetter('name'))
    fields = sorted(fields, key=attrgetter('cells'))
    fields = list(discover_decompositions(fields))

    for field in geometries:
        log.debug(f"Discovered geometry '{field.name}' with coordinates {field.coords}")

    return geometries, fields


def pick_geometry(geometries: List[Field]) -> Tuple[Field, Converter]:
    if not geometries:
        raise TypeError("No geometry found, don't know what to do")

    # Find the geometry that can most easily be converted to the target
    index, converter = graph.optimal_source(config.coords, map(attrgetter('coords'), geometries))
    return geometries[index], converter


def pipeline(reader: Source, writer: Writer):
    """Main driver for moving data from reader to writer."""

    # TODO: Streamline filter application
    if config.only_final_timestep:
        reader = LastStepFilter(reader)
    elif config.timestep_slice is not None:
        reader = StepSliceFilter(reader, *map(int, config.timestep_slice.split(':')))
    reader = TesselatorFilter(reader)

    if writer.writer_name != 'VTF':
        reader = MergeTopologiesFilter(reader)

    reader = CoordinateTransformFilter(reader, config.coords)

    if config.offset_should_exist and config.offset_file is None:
        log.warning("Unable to find mesh origin info, coordinates may be unreliable")
    if config.offset_file is not None:
        log.info(f"Using {config.offset_file} for offset information")
        reader = OffsetFilter(reader, config.offset_file)

    geometries, fields = discover_fields(reader)
    if not geometries:
        raise ValueError(f"Unable to find any useful geometry fields")
    geometry = geometries[0]
    log.debug(f"Using '{geometry.name}' as geometry input")

    first = True
    for stepid, stepdata in log.iter.plain('Step', reader.steps()):
        with writer.step(stepdata) as step:
            with step.geometry(geometry) as geom:
                for patch, data in geometry.patches(stepid, force=first):
                    geom(patch, data)

            for field in fields:
                with step.field(field) as fld:
                    for patch, data in field.patches(stepid, force=first, coords=geometry.coords):
                        fld(patch, data)

            first = False
