# Projecting a periodic snapshot into HEALPix shells

The C kernel in `src/lightcone.c` accumulates particle masses directly into
HEALPix **RING** maps. The NumPy/ctypes wrapper in `src/lightcone.py` streams
chunks, accepts float32 or float64 coordinates, and supports OpenMP.
It requires Python, NumPy and a C11 compiler. healpy is required for the tests,
but not by the projection kernel.

This is a frozen-snapshot lightcone approximation: the same snapshot is used
throughout the selected radial interval. Time evolution and particle crossings
between snapshots are not interpolated.

## Build

Run from the project root with the Python environment used for the project:

```bash
python -m src.lightcone --build
```

This creates `src/_lightcone.so`. If your compiler lacks OpenMP:

```bash
python -m src.lightcone --build --serial
```

Use `threads=1` with a serial build. Set the `CC` environment variable to choose
a compiler. Rebuild after changing the C source; start a new Python process
after rebuilding a library that is already loaded.

## Full-sky periodic projection

Positions must be comoving Cartesian coordinates in **[0, boxsize)** on each
axis. Distances, positions, box size and observer position must use the same
unit, such as Mpc/h. Supply the selected shell boundaries in comoving distance;
there is no per-particle redshift conversion.

```python
import numpy as np
from src.lightcone import ShellProjector

# Example only: replace these with your actual selected distance boundaries.
distance_edges = np.array([1000., 1200., 1400., 1600.])  # Mpc/h
boxsize = 1000.                                       # Mpc/h

projector = ShellProjector.periodic(
    distance_edges,
    nside=1024,
    boxsize=boxsize,
    observer=(boxsize / 2, boxsize / 2, boxsize / 2),
    threads=8,
)

# Example input layout: an (N, 3) float32/float64 .npy file.
# For HDF5 or simulation binaries, replace this reader with chunked reads.
positions = np.load("positions.npy", mmap_mode="r")
chunk_size = 1_000_000
for start in range(0, len(positions), chunk_size):
    projector.add(positions[start:start + chunk_size])

# Shape: (3, 12 * 1024**2). Scalar masses=1 gives particle counts.
np.save("particle_shells.npy", projector.maps)
```

To use your existing redshift selection, compute `selected_zb =
get_replaced_zs(zb, zctr, boxsize, Om)`, then `distance_edges =
np.array([z2d(z, Om) for z in selected_zb])`. The existing `z2d` helper uses
`np.trapezoid`, which requires NumPy 2 or newer. At least two selected
boundaries are needed to define a shell. Ensure this distance cosmology
matches the one used to generate the GLASS shells.

For equal-mass particles use `projector.add(chunk, masses=particle_mass)`.
For unequal masses supply a one-dimensional array for the same chunk:
`projector.add(chunk, masses=mass_chunk)`.

A shell contains radii `[distance_edges[j], distance_edges[j+1])`. Each periodic
image contributes once; particles exactly at the observer have no direction
and are skipped. Coordinates and particle masses must be finite.
The automatic tile selection rejects copies whose boxes do not intersect the
radial interval. No copies of the full particle catalogue are constructed.
Repeated images contain repeated structure, as expected for periodic tiling.

The observer defaults to the primary box centre. Changing it changes the sky
realization. The snapshot redshift determines which shells you choose; it does
not imply a unique placement of the observer.

## Disk-backed output

One float64 shell at nside=1024 occupies 96 MiB. To avoid keeping many shell
maps in RAM, supply a writable memory-mapped output initialized to zero:

```python
shape = (len(distance_edges) - 1, 12 * 1024**2)
maps = np.lib.format.open_memmap(
    "mass_shells.npy", mode="w+", dtype=np.float64, shape=shape
)
maps[:] = 0
projector = ShellProjector.periodic(
    distance_edges, 1024, boxsize=boxsize, threads=8, out=maps
)
# Stream chunks using projector.add(...) as above.
maps.flush()
```

An existing output is accumulated into, rather than cleared. Contiguous,
aligned native float32/float64 position chunks are passed without copying.
Other layouts are converted one chunk at a time. The C loop uses atomic map
updates rather than allocating a full map per thread. Thread count should be
benchmarked: histogram contention and memory bandwidth can limit scaling.
Do not call `add` concurrently on projectors sharing the same output.

## Converting masses to density contrast

For full-sky, volume-weighted top-hat shells, the comoving volume of one
HEALPix pixel in shell j is

```python
npix = projector.maps.shape[1]
pixel_volumes = (4 * np.pi / (3 * npix)) * np.diff(distance_edges**3)
mean_density = total_mass_in_original_box / boxsize**3
delta = projector.maps / (mean_density * pixel_volumes[:, None]) - 1
```

For particle-count maps, use the original box particle count divided by
`boxsize**3` as the mean number density. Use the original box's mean rather than
forcing each shell's measured mean to zero.

These maps are volume-weighted shell projections. Match the radial weighting
and normalization of your GLASS fields before replacing them; a normalized
window in redshift or distance is not generally the same as volume weighting.
The kernel does not implement linear/cubic GLASS windows or redshift-space
distortions.

## Explicit translations

For a single box or a custom selection of images:

```python
projector = ShellProjector(
    distance_edges, 1024,
    observer=(0., 0., 0.),
    offsets=[[0., 0., 1500.]],  # positions + translation - observer
    threads=8,
)
```

No coordinate wrapping occurs inside the kernel.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests compare against independent healpy/NumPy projections, including polar
and equatorial directions, nside up to 1024, shell-edge assignment, masses,
both coordinate precisions, chunk accumulation, memmaps, periodic tile
culling, and parallel accumulation into a single pixel.
