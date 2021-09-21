from contextlib import contextmanager
from enum import IntEnum, Enum
from pathlib import Path

from typing import Dict, Any, Optional, Tuple

import treelog as log


class ConfigSource(IntEnum):
    """Simple enum that tracks where a config value comes from, in
    order of 'overridability'.
    """

    # Default values. Can be changed at will when needed.
    Default = 0

    # Values provided by the user. We can change these if needed but
    # must issue a warning when doing so.
    User = 1

    # Values required by the program. We cannot change these if
    # needed: an error must be thrown instead.
    Required = 2

Default = ConfigSource.Default
User = ConfigSource.User
Required = ConfigSource.Required


class ConfigTarget(Enum):
    """Simple enum that tracks the intended target of a config option."""

    # Option intended for readers
    Reader = 0

    # Option intended for writers
    Writer = 1

    # Option intended for the pipeline layer
    Pipeline = 2

Reader = ConfigTarget.Reader
Writer = ConfigTarget.Writer
Pipeline = ConfigTarget.Pipeline



class Setting:

    def __init__(self, default, *targets, name=None):
        self.default = default
        self.targets = targets
        self.name = name


class ConfigMeta(type):
    """Metaclass for the Config class.  Collects metadata about settings."""

    def __new__(cls, clsname, bases, attrs):
        sources = dict()
        names = dict()
        targets = dict()
        defaults = dict()

        for key, value in attrs.items():
            if not isinstance(value, Setting):
                continue
            defaults[key] = value.default
            sources[key] = ConfigSource.Default
            targets[key] = value.targets

            if value.name:
                names[key] = value.name
            else:
                names[key] = f'--{key}'

        attrs['_value_sources'] = sources
        attrs['_canonical_names'] = names
        attrs['_targets'] = targets
        attrs.update(defaults)

        return super().__new__(cls, clsname, bases, attrs)


class Config(metaclass=ConfigMeta):
    """Configuration object that should be shared between reader and
    writer.  It is initialized by the main function, but may have its
    attributes manipulated by either reader or writer depending on
    their needs.
    """

    # Number of subdivisions for additional resolution.
    nvis = Setting(1)

    # Which fields to include
    field_filter = Setting(None, Pipeline, name='--filter')

    # Whether to copy only the final time step.
    only_final_timestep = Setting(False, Pipeline, name='--last')

    # Whether to slice the time step iterator
    timestep_slice = Setting(None, Pipeline, name='--times')

    # Desired output coordinate system.
    coords = Setting('local', Pipeline, name='--coords')

    # Whether the data set contains multiple time steps.
    multiple_timesteps = Setting(True, Writer)

    # Enforce unstructured output, even if input is structured.  Used by writers
    # that can switch dynamically, e.g. VTK.
    require_unstructured = Setting(False, Writer, name='--unstructured')

    # Output mode. Used by the VTK and VTF writers.
    output_mode = Setting('binary', Writer, name='--mode')

    # Output endianness indicator. Used by the SIMRA writer.
    output_endianness = Setting('native', Writer, name='--out-endianness')

    # Input coordinate overrides.
    input_coords = Setting({}, Pipeline, name='--in-coords')

    # Use info.txt grid offset
    offset_file = Setting(None, Pipeline, name="--offset")

    # Expect to find info.txt grid offset
    offset_should_exist = Setting(False, Pipeline)

    # Input endianness indicator. Used by the SIMRA reader.
    input_endianness = Setting('native', Reader, name='--in-endianness')

    # Input mesh file. Used by the SIMRA reader.
    mesh_file = Setting(None, Reader, name='--mesh')

    # List of basis objects to copy to output. Used by the IFEM reader.
    only_bases = Setting((), Reader, name='--basis')

    # Strict patch identification. Used by the IFEM reader.
    strict_id = Setting(False, Reader, name='--strict-id')

    # Strictly assume LR patches are NURBS. Used by the IFEM and LR readers.
    # 'always', 'never' or None
    lr_are_nurbs = Setting(None, Reader, name='--lr-are-nurbs')

    # Volumetric/surface field behaviour. Used by the WRF reader.
    # - volumetric: only include volumetric fields
    # - planar: only include surface fields
    # - extrude: include all, extruding surface fields upwards
    volumetric = Setting('volumetric', Reader, name='--volumetric/planar/extrude')

    # Hint to the reader that the data may be periodic. Used by the WRF reader.
    periodic = Setting(False, Reader)

    # Whether to attempt to automatically resolve RHS/LHS systems
    fix_orientation = Setting(True, Reader, Writer, name='--(no-)fix-orientation')

    def cname(self, key: str) -> str:
        """Get the canonical name of a setting."""
        return self._canonical_names[key]

    def source(self, key: str) -> ConfigSource:
        """Get the source of a setting."""
        return self._value_sources[key]

    def target_compatible(self, key: str, target: ConfigTarget) -> bool:
        """Check if a setting is intended for a target."""
        return target in self._targets[key]

    def upgrade_source(self, key: str, source: ConfigSource):
        """Override the source of a setting. This will only ever increase the
        source level, and will do so quietly: no warnings will be
        issued.
        """
        current_source = self.source(key)
        if current_source > source:
            raise ValueError(f"Attempted to downgrade source of {self.cname(key)}")
        self._value_sources[key] = source

    def assign(self, key: str, value: Any, source: ConfigSource, reason: Optional[str] = None):
        """Assign a value to a setting.  This may issue a warning or
        an error in case the source levels require it.
        """
        current_source = self.source(key)
        current_value = getattr(self, key)
        if current_source == ConfigSource.Required and value != current_value:
            if reason is not None:
                raise ValueError(f"Incompatibility with setting '{self.cname(key)}': {reason}")
            else:
                raise ValueError(f"Incompatibility with setting '{self.cname(key)}'")

        elif current_source == User and value != current_value:
            log.warning(f"Setting '{self.cname(key)}' was overridden from {current_value} to {value}")
            if reason is not None:
                log.warning(f"Reason: {reason}")

        assert current_source <= source
        setattr(self, key, value)
        self._value_sources[key] = source

    def require(self, reason: Optional[str] = None, **kwargs: Any):
        """Set some settings to new values with the highest source
        level.  If they have previously been required to have
        different values, an error will be thrown.
        """
        for key, value in kwargs.items():
            self.assign(key, value, Required, reason)

    def require_in(self, reason: Optional[str] = None, **kwargs: Tuple[Any]):
        """Ensure some settings to be one of a given set of options.
        If not, choose the first one.  If they have previously been
        required to have different values, an error will be thrown."""
        for key, values in kwargs.items():
            if getattr(self, key) not in values:
                self.assign(key, values[0], Required, reason)

    def ensure_limited(self, target: ConfigTarget, *args: str, reason: Optional[str] = None,
                       lower: ConfigSource = User, upper: ConfigSource = User):
        """Ensure that the set of explicitly sourced settings for a given
        target is limited to only those given.  If not, an error will
        be thrown.  By default, checks the user-provided settings.
        """
        args = set(args)
        for key, source in self._value_sources.items():
            if self.target_compatible(key, target) and lower <= source <= upper and key not in args:
                if reason is not None:
                    raise ValueError(f"'{self.cname(key)}' should not have been set ({reason})")
                else:
                    raise ValueError(f"'{self.cname(key)}' should not have been set")

    @contextmanager
    def __call__(self, source: ConfigSource = User, **kwargs: Any):
        """Context manager for running code with different settings."""
        prev = dict(self.__dict__)
        prev_sources = dict(self._value_sources)
        for key, value in kwargs.items():
            self.assign(key, value, source)
        try:
            yield
        finally:
            self.__dict__ = prev
            self._value_sources = prev_sources


config = Config()
