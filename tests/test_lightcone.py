import tempfile
import unittest
from pathlib import Path
import healpy as hp
import numpy as np
from src.lightcone import ShellProjector, build

def reference(positions, edges, nside, observer, offsets, masses=1.):
    result = np.zeros((len(edges)-1, hp.nside2npix(nside)))
    for offset in offsets:
        xyz = positions.astype(np.float64) + offset - observer
        r = np.linalg.norm(xyz, axis=1)
        shell = np.searchsorted(edges, r, side="right") - 1
        valid = (r > 0) & (shell >= 0) & (shell < len(edges)-1)
        pix = hp.vec2pix(nside, *xyz[valid].T)
        index = shell[valid] * result.shape[1] + pix
        weights = np.broadcast_to(masses, (len(positions),))[valid]
        result += np.bincount(index, weights=weights,
                             minlength=result.size).reshape(result.shape)
    return result

class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()

    def test_pixels_shells_and_weights(self):
        rng = np.random.default_rng(42)
        for dtype in (np.float32, np.float64):
            pos = rng.uniform(-4, 4, (10000, 3)).astype(dtype)
            mass = rng.uniform(.1, 3, len(pos))
            edges = np.array([0., 1., 2.5, 4., 7.])
            obs = np.array([.2, -.1, .3])
            shifts = np.array([[0., 0., 0.], [1., -2., .5]])
            for nside in (1, 4, 32, 1024):
                p = ShellProjector(edges, nside, observer=obs, offsets=shifts)
                p.add(pos, masses=mass)
                np.testing.assert_allclose(p.maps,
                    reference(pos, edges, nside, obs, shifts, mass),
                    rtol=1e-12, atol=1e-12)

    def test_axes_poles_and_radial_edges(self):
        pos = np.array([[0.,0.,0.], [1.,0.,0.], [0.,2.,0.],
                        [0.,0.,1.], [0.,0.,-1.], [3.,0.,0.],
                        [1e-9,0.,1.], [-1.,0.,0.]])
        p = ShellProjector([0.,1.,2.,3.], 8)
        p.add(pos)
        np.testing.assert_array_equal(p.maps, reference(
            pos, np.array([0.,1.,2.,3.]), 8, np.zeros(3), np.zeros((1,3))))
        np.testing.assert_array_equal(p.maps.sum(axis=1), [0.,5.,1.])

    def test_chunks_strides_and_memmap(self):
        pos = np.random.default_rng(7).uniform(-3,3,(300,6))[:,::2]
        with tempfile.TemporaryDirectory() as directory:
            out = np.memmap(Path(directory)/"maps.dat", mode="w+",
                            dtype=np.float64, shape=(2,192))
            p = ShellProjector([0.,2.,6.], 4, out=out)
            for chunk in np.array_split(pos,7):
                p.add(chunk, masses=2.)
            np.testing.assert_array_equal(out, reference(
                pos, np.array([0.,2.,6.]), 4, np.zeros(3), np.zeros((1,3)), 2.))

    def test_periodic_tile_culling(self):
        pos = np.random.default_rng(3).uniform(0,2,(300,3)).astype(np.float32)
        edges = np.array([.3,1.5,3.5])
        obs = np.array([.1,1.9,.7])
        p = ShellProjector.periodic(edges, 8, boxsize=2., observer=obs)
        p.add(pos)
        shifts = np.array([(x,y,z) for x in range(-6,7,2)
                           for y in range(-6,7,2) for z in range(-6,7,2)],float)
        np.testing.assert_array_equal(p.maps, reference(pos,edges,8,obs,shifts))

    def test_thread_count_cannot_overflow_c_integer(self):
        # ctypes must not wrap a positive Python integer into a negative count.
        with self.assertRaises(ValueError):
            ShellProjector([0.,2.], 8, threads=2**31)

    def test_parallel_atomic_accumulation(self):
        pos = np.tile([1.,.2,.3], (100000,1))
        p = ShellProjector([0.,2.], 8, threads=4)
        p.add(pos)
        self.assertEqual(p.maps.sum(), 100000.)
        self.assertEqual(np.count_nonzero(p.maps), 1)

if __name__ == "__main__":
    unittest.main()
