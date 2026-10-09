"""Chunked snapshot projection into comoving HEALPix RING mass shells.

Build once with: python -m src.lightcone --build
This projects a fixed snapshot; it does not interpolate evolution in time.
"""
import argparse
import ctypes
from itertools import product
import operator
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np

_DIRECTORY = Path(__file__).resolve().parent
_LIBRARY = _DIRECTORY / "_lightcone.so"


def build(*, openmp=True, output=None):
    """Compile the C kernel with a C11 compiler (Linux shared library)."""
    target = Path(output).resolve() if output is not None else _LIBRARY
    compiler = shlex.split(os.environ.get("CC", "cc"))
    command = compiler + ["-O3", "-std=c11", "-Wall", "-Wextra",
                          "-Werror", "-fPIC", "-shared"]
    if openmp:
        command.append("-fopenmp")
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        temporary = Path(directory) / target.name
        subprocess.run(command + [str(_DIRECTORY / "lightcone.c"),
                                  "-o", str(temporary), "-lm"], check=True)
        os.replace(temporary, target)
    return target


def _load():
    if not _LIBRARY.exists():
        raise RuntimeError("Build the kernel first: python -m src.lightcone --build")
    library = ctypes.CDLL(str(_LIBRARY))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int64
    library.lightcone_accumulate.argtypes = [
        pointer, ctypes.c_int, integer, pointer, ctypes.c_double,
        pointer, pointer, integer, pointer, integer, integer,
        pointer, ctypes.c_int,
    ]
    library.lightcone_accumulate.restype = None
    library.lightcone_has_openmp.argtypes = []
    library.lightcone_has_openmp.restype = ctypes.c_int
    return library


def _edges(values):
    edges = np.array(values, dtype=np.float64, order="C", copy=True)
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("distance_edges must be finite, increasing and nonnegative")
    return edges


def _observer(values):
    observer = np.array(values, dtype=np.float64, order="C", copy=True)
    if observer.shape != (3,) or not np.all(np.isfinite(observer)):
        raise ValueError("observer must be a finite vector of length three")
    return observer


def periodic_offsets(boxsize, observer, distance_edges):
    """Translations of [0,L)^3 tiles intersecting the requested radial range."""
    if not np.isfinite(boxsize) or boxsize <= 0:
        raise ValueError("boxsize must be finite and positive")
    observer = _observer(observer)
    edges = _edges(distance_edges)
    rmin, rmax = edges[0], edges[-1]
    low = np.floor((observer-rmax)/boxsize).astype(np.int64)
    high = np.floor((observer+rmax)/boxsize).astype(np.int64)
    offsets = []
    for tile in product(*(range(int(a), int(b)+1) for a,b in zip(low,high))):
        shift = np.array(tile, dtype=np.float64)*boxsize
        near = shift-observer
        far = near+boxsize
        nearest = np.maximum(np.maximum(near, -far), 0.)
        farthest = np.maximum(np.abs(near), np.abs(far))
        if np.dot(nearest,nearest) < rmax*rmax and np.dot(farthest,farthest) >= rmin*rmin:
            offsets.append(shift)
    return np.asarray(offsets, dtype=np.float64).reshape(-1,3)


class ShellProjector:
    """Accumulate particle counts or masses in [distance_edges[j], edges[j+1]).

    All geometry uses a single comoving length unit, e.g. Mpc/h.
    Positions are (N,3) arrays; offsets are box translations.
    Output is (number_of_shells, 12*nside**2), float64, RING ordering.
    An existing writable C-contiguous float64 output, including a memmap,
    can be supplied via out; its current contents are preserved.
    """

    def __init__(self, distance_edges, nside, *, observer=(0.,0.,0.),
                 offsets=None, threads=1, out=None):
        self.distance_edges = _edges(distance_edges)
        self.nside = operator.index(nside)
        if self.nside < 1 or self.nside > 2**20 or self.nside & (self.nside-1):
            raise ValueError("nside must be a power of two between 1 and 2**20")
        self.observer = _observer(observer)
        self.offsets = np.array([[0.,0.,0.]] if offsets is None else offsets,
                                dtype=np.float64, order="C", copy=True)
        if (self.offsets.ndim != 2 or self.offsets.shape[1] != 3
                or not np.all(np.isfinite(self.offsets))):
            raise ValueError("offsets must have shape (number_of_copies, 3)")
        self.threads = operator.index(threads)
        if not 1 <= self.threads <= 2**31 - 1:
            raise ValueError("threads must be between 1 and 2**31 - 1")
        shape = (len(self.distance_edges)-1, 12*self.nside**2)
        if out is None:
            self.maps = np.zeros(shape, dtype=np.float64)
        else:
            if (not isinstance(out, np.ndarray) or out.shape != shape
                    or out.dtype != np.float64 or not out.flags.c_contiguous
                    or not out.flags.writeable or not out.flags.aligned):
                raise ValueError("out must be writable contiguous float64 with shape " + str(shape))
            self.maps = out
        self._library = _load()
        self.openmp = bool(self._library.lightcone_has_openmp())
        if self.threads > 1 and not self.openmp:
            raise ValueError("Multiple threads require a kernel built with OpenMP")
        # Geometry must not change during accumulation.
        for values in (self.distance_edges, self.observer, self.offsets):
            values.flags.writeable = False

    @classmethod
    def periodic(cls, distance_edges, nside, *, boxsize, observer=None,
                 threads=1, out=None):
        """Full-sky tiling of a snapshot whose positions lie in [0,boxsize)^3.

        Default observer is at the centre of the primary box. Each physical
        periodic image is accumulated once. Replicas repeat structure.
        """
        observer = (np.full(3, boxsize/2.) if observer is None else observer)
        offsets = periodic_offsets(boxsize, observer, distance_edges)
        return cls(distance_edges, nside, observer=observer, offsets=offsets,
                   threads=threads, out=out)

    def add(self, positions, *, masses=1.0):
        """Accumulate one chunk, returning maps. Scalar masses=1 gives counts.

        Contiguous native float32/float64 positions are passed without copying.
        Strided or other dtypes are converted for this chunk only. Per-particle
        masses are converted to contiguous float64. Inputs must be finite;
        particles exactly at the observer have no direction and are skipped.
        Calls sharing the same output must not run concurrently.
        """
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N,3)")
        dtype = positions.dtype if positions.dtype in (np.dtype("float32"), np.dtype("float64")) else np.float64
        positions = np.require(positions, dtype=dtype, requirements=["C", "A"])
        if np.may_share_memory(positions, self.maps):
            raise ValueError("positions must not overlap the output map")
        weights = np.asarray(masses)
        if weights.ndim == 0:
            constant_mass = float(weights)
            if not np.isfinite(constant_mass):
                raise ValueError("mass must be finite")
            mass_pointer = None
        else:
            if weights.shape != (len(positions),):
                raise ValueError("masses must be scalar or have shape (N,)")
            weights = np.require(weights, dtype=np.float64, requirements=["C", "A"])
            if np.may_share_memory(weights, self.maps):
                raise ValueError("masses must not overlap the output map")
            mass_pointer = weights.ctypes.data
            constant_mass = 0.
        self._library.lightcone_accumulate(
            positions.ctypes.data, positions.itemsize, len(positions),
            mass_pointer, constant_mass, self.observer.ctypes.data,
            self.offsets.ctypes.data, len(self.offsets),
            self.distance_edges.ctypes.data, len(self.distance_edges)-1,
            self.nside, self.maps.ctypes.data, self.threads)
        return self.maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="compile the C kernel")
    parser.add_argument("--serial", action="store_true", help="build without OpenMP")
    args = parser.parse_args()
    if args.build:
        print(build(openmp=not args.serial))
    else:
        parser.print_help()
