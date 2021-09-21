from functools import wraps
from pathlib import Path
import sys
import traceback
import warnings

import click
import treelog as log

from typing import Optional

from . import config, ConfigSource
from .coords import Coords, Local, Geocentric
from .pipeline import pipeline
from .util import split_commas
from .reader import Reader
from .writer import Writer


def suppress_warnings(func):
    @wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Conversion of the second argument of issubdtype')
            warnings.filterwarnings('ignore', r'No GPU/TPU found, falling back to CPU')
            return func(*args, **kwargs)
    return inner


class RichOutputLog(log.RichOutputLog):

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, text, level):
        message = ''.join([self._cmap[level.value], text, '\033[0m\n', self._current])
        click.echo(message, file=self.stream, nl=False)


class CoordsType(click.ParamType):
    """Parameter type for coordinate systems."""

    name = "coords"

    def convert(self, value, param, ctx):
        if value is None or value is False:
            return None
        if value is None or isinstance(value, Coords):
            return value
        rval = Coords.find(value)
        return rval


FORMATS = ['vtf', 'vtk', 'vtu', 'vts', 'pvd', 'nc', 'dat']

@click.command()
@click.option('--fmt', '-f', type=click.Choice(FORMATS), required=False, help='Output format.')

# Options that are forwarded to config
@click.option('--periodic/--no-periodic', help='Hint that the data may be periodic.', default=False)
@click.option('--basis', '-b', 'only_bases', multiple=True, help='Include fields in this basis.')
@click.option('--nvis', '-n', 'nvis', default=1, help='Extra sampling points per element.')
@click.option('--last', 'only_final_timestep', is_flag=True, help='Read only the last step.')
@click.option('--times', 'timestep_slice', default=None, help='Slice the timestep list (Python syntax).')
@click.option('--time', 'timestep_index', type=int, default=None)
@click.option('--mode', '-m', 'output_mode', type=click.Choice(['binary', 'ascii', 'appended']),
              default='binary', help='Output mode.')
@click.option('--strict-id', 'strict_id', is_flag=True, help='Strict patch identification.')
@click.option('--unstructured', 'require_unstructured', is_flag=True, help='Ensure unstructured output format.')
@click.option('--fix-orientation/--no-fix-orientation', 'fix_orientation', default=True)
@click.option('--lr-are-nurbs', 'lr_are_nurbs', flag_value='always', help='Always treat LR patches as NURBS.')
@click.option('--lr-are-not-nurbs', 'lr_are_nurbs', flag_value='never', help='Never treat LR patches as NURBS.')

@click.option('--endianness', 'input_endianness', type=click.Choice(['native', 'little', 'big']), default='native')
@click.option('--in-endianness', 'input_endianness', type=click.Choice(['native', 'little', 'big']), default='native')
@click.option('--out-endianness', 'output_endianness', type=click.Choice(['native', 'little', 'big']), default='native')

@click.option('--no-fields', 'no_fields', is_flag=True)
@click.option('--filter', '-l', 'field_filter', multiple=True, help='List of fields to include.', default=None)

@click.option('--volumetric', 'volumetric', flag_value='volumetric', help='Only include volumetric fields.', default=True)
@click.option('--planar', 'volumetric', flag_value='planar', help='Only include planar (surface) fields.')
@click.option('--extrude', 'volumetric', flag_value='extrude', help='Extrude planar (surface) fields.')

@click.option('--mesh', 'mesh_file', help='Name of mesh file.')

@click.option('--geometry', '-g', 'coords', default=Local(), help='Use this basis to provide geometry.', type=CoordsType())
@click.option('--local', 'coords', flag_value=Local(), help='Local (cartesian) mapping.', type=CoordsType())
@click.option('--global', 'coords', flag_value=Geocentric(), help='Global (spherical) mapping.', type=CoordsType())
@click.option('--coords', default=Local(), type=CoordsType())
@click.option('--in-coords', 'input_coords', nargs=2, multiple=True, type=click.Tuple([str, CoordsType()]))

@click.option('--no-offset', 'no_offset', is_flag=True, help="Don't read info.txt offset file")
@click.option('--offset', 'offset_file', type=str, required=False)

# Logging and verbosity
@click.option('--debug', 'verbosity', flag_value='debug')
@click.option('--info', 'verbosity', flag_value='info', default=True)
@click.option('--user', 'verbosity', flag_value='user')
@click.option('--warning', 'verbosity', flag_value='warning')
@click.option('--error', 'verbosity', flag_value='error')
@click.option('--rich/--no-rich', default=True)

# Filenames
@click.argument('infile', type=str, required=True)
@click.argument('outfile', type=str, required=False)

@click.pass_context

@suppress_warnings
def convert(ctx, verbosity, rich, infile, fmt, outfile, **kwargs):
    # Set up logging
    if rich:
        logger = RichOutputLog(sys.stdout)
    else:
        logger = log.TeeLog(
            log.FilterLog(log.StdoutLog(), maxlevel=log.proto.Level.user),
            log.FilterLog(log.StderrLog(), minlevel=log.proto.Level.warning),
        )
    log.current = log.FilterLog(logger, minlevel=getattr(log.proto.Level, verbosity))

    # Print potential warnings
    if '--global' in sys.argv:
        log.warning(f"--global is deprecated; use --coords geocentric instead")
    if '--local' in sys.argv:
        log.warning(f"--local is deprecated; use --coords local instead")
    if '--geometry' in sys.argv or '-g' in sys.argv:
        log.warning(f"--geometry is deprecated; use --coords instead")
    if '--endianness' in sys.argv:
        log.warning(f"--endianness is deprecated; use --in-endianness instead")

    # Convert to pathlib
    infile = Path(infile)
    outfile = Path(outfile) if outfile else None

    # Determine filename of output
    if outfile and not fmt:
        fmt = outfile.suffix[1:].lower()
    elif not outfile:
        fmt = fmt or 'pvd'
        outfile = Path(infile.name).with_suffix(f'.{fmt}')

    # Handle default values of multi-valued options that should be
    # distinguished from empty, as well as comma splitting and other
    # transformations
    explicit = {click.core.ParameterSource.COMMANDLINE, click.core.ParameterSource.ENVIRONMENT}
    kwargs['input_coords'] = dict(kwargs['input_coords'])
    for k in ['field_filter', 'only_bases']:
        kwargs[k] = tuple(split_commas(kwargs[k]))

    if kwargs['no_fields']:
        kwargs['field_filter'] = []
    elif ctx.get_parameter_source('field_filter') not in explicit:
        kwargs['field_filter'] = None
    else:
        kwargs['field_filter'] = tuple(f.lower() for f in kwargs['field_filter'])

    if kwargs['no_offset']:
        kwargs['offset_file'] = False

    if kwargs['offset_file'] is None:
        info_txt_path = infile.parent / 'info.txt'
        if info_txt_path.exists():
            kwargs['offset_file'] = info_txt_path
    elif isinstance(kwargs['offset_file'], str):
        kwargs['offset_file'] = Path(kwargs['offset_file'])

    if isinstance(kwargs['timestep_index'], int):
        n = kwargs['timestep_index']
        kwargs['timestep_slice'] = f'{n}:{n+1}:1'
        config.require(multiple_timesteps=False, reason="--time is set")

    # Remove meta-options
    for k in ['timestep_index', 'no_fields', 'no_offset']:
        kwargs.pop(k)

    try:
        # The config can influence the choice of readers or writers,
        # so apply it first.  Since kwargs may include options that
        # are not explicity set by the user, we set the source to
        # Default, and later use the upgrade_source method.
        with config(source=ConfigSource.Default, **kwargs):
            for name in kwargs:
                if ctx.get_parameter_source(name) in explicit:
                    config.upgrade_source(name, ConfigSource.User)
            if not infile.exists():
                raise IOError(f"File or directory does not exist: {infile}")
            ReaderClass = Reader.find_applicable(infile)
            WriterClass = Writer.find_applicable(fmt)
            with ReaderClass(infile) as reader, WriterClass(outfile) as writer:
                reader.validate()
                writer.validate()
                pipeline(reader, writer)

    except Exception as e:
        if verbosity == 'debug':
            # In debug mode, allow exceptions to filter through in raw form
            traceback.print_exc()
        else:
            log.error(f"Error: {e}")
            log.error("More information may be available with --debug")
        sys.exit(1)


def deprecated():
    print("ifem-to-vt is deprecated, please launch with 'siso'\n\n", file=sys.stderr)
    convert()


if __name__ == '__main__':
    convert()
