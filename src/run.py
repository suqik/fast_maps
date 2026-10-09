import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# uses the CAMB cosmology which produced the cls
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb

# creating a numpy random number generator for sampling
rng = np.random.default_rng(seed=42)

@dataclass
class Config:
    cosmo_dict: dict
    nside: int
    lmax: int
    zmin: float
    zmax: float
    dx: float
    window_type: str

def z2d(z, Om):
    Ol = 1 - Om
    sol = 2.99792458e5 # km/s
    z_for_inte = np.linspace(0, z, 1000)
    Ez = np.sqrt(Om*(1+z_for_inte)**3 + Ol)
    dist = sol / 100.0 * np.trapezoid(1./Ez, z_for_inte)

    return dist

def get_cosmo_instance(cosmo_dict):
    h = cosmo_dict['hubble']
    Om = cosmo_dict['Omega_M']
    Ob = cosmo_dict['Omega_b']
    Oc = Om - Ob

    # set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_background(pars)

    # get the cosmology from CAMB
    cosmo = Cosmology(results)

    return pars, cosmo

def gen_lognormal_fields(config:Config):
    # basic parameters of the simulation
    nside = config.nside
    lmax = config.lmax

    cosmo_dict = config.cosmo_dict

    zmin = config.zmin
    zmax = config.zmax
    dx   = config.dx

    window_type = config.window_type

    pars, cosmo = get_cosmo_instance(cosmo_dict)

    zb = glass.distance_grid(cosmo, zmin, zmax, dx)

    match window_type:
        case 'tophat':
            shells = glass.tophat_windows(zb)
            ncorr = 1
        case 'linear':
            shells = glass.linear_windows(zb)
            ncorr = 3
        case 'cubic':
            shells = glass.cubic_windows(zb)
            ncorr = 3

    print("Computing angular matter power spectra ... ", end='', flush=True)

    cls = glass.ext.camb.matter_cls(pars, lmax, shells)

    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax)

    print('Done.', flush=True)

    # set up lognormal fields for simulation
    fields = glass.lognormal_fields(shells)

    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)

    # this generator will yield the matter fields in each shell
    matter = glass.generate(fields, gls, nside, ncorr=ncorr, rng=rng)

    return matter, zb

def get_replaced_zs(zb, zctr, boxsize, Om):

    distctr = z2d(zctr, Om)
    dmin_box = max(0.0, distctr - boxsize / 2.0)
    dmax_box = distctr + boxsize / 2.0

    # Limits outside the available shells snap to the endpoint boundary.
    # Only invert distances inside this range, where a root is bracketed.
    distb = np.array([z2d(z, Om) for z in zb])

    imin = np.argmin(np.abs(distb - dmin_box))
    if dmin_box > distb[imin]:
        imin += 1
    imax = np.argmin(np.abs(distb - dmax_box))
    if dmax_box < distb[imax]:
        imax -= 1

    return zb[imin:imax + 1]


if __name__ == "__main__":
    example_config = Config(
        cosmo_dict = {
            'hubble': 0.6727,
            'Omega_M': 0.2,
            'Omega_b': 0.049,
            'n_s': 0.9667,
            'sigma8': 0.831
        },
        nside = 1024,
        lmax=1000,
        zmin = 0.0,
        zmax = 1.0,
        dx = 200.0,
        window_type='tophat'
    )