from splipy.io import G2

from .. import config
from ..geometry import SplinePatch
from .hdf5 import GeometryManager


class SimpleBasis:

    def __init__(self, patches):
        self.patches = patches
        self.name = 'Geometry'

    @property
    def npatches(self):
        return len(self.patches)

    def update_at(self, stepid):
        return stepid == 0

    def patch_at(self, stepid, patchid):
        return self.patches[patchid]


class Reader:

    def __init__(self, filename):
        self.filename = filename
        config.require(last=True)

    def __enter__(self):
        with G2(self.filename) as g2:
            patches = list(map(SplinePatch, g2.read()))
        self.basis = SimpleBasis(patches)
        return self

    def __exit__(self, type_, value, backtrace):
        pass

    def write(self, w):
        w.add_step(time=0.0)
        geometry = GeometryManager(self.basis)
        geometry.update(w, 0)
        w.finalize_step()
