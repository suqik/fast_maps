/*
 * Streaming snapshot-to-HEALPix projection. Positions, edges, observer and
 * offsets must use the same comoving length unit. Maps use RING ordering.
 * HEALPix geometry: Gorski et al., 2005, doi:10.1086/427976.
 * No particle catalogues, per-particle pixel arrays or private full maps.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Map a nonzero direction to a zero-based RING pixel. */
static int64_t ring_pixel(int64_t n, double x, double y, double z, double r)
{
    const double pi = 3.141592653589793238462643383279502884;
    const int64_t npix = 12*n*n, cap = 2*n*(n-1);
    double mu = z/r, a = fabs(mu);
    double longitude = atan2(y, x);
    if (longitude < 0.0) longitude += 2.0*pi;
    double quadrant = longitude * (2.0/pi);
    if (quadrant >= 4.0) quadrant = 0.0;

    if (a <= 2.0/3.0) {
        int64_t ascending = (int64_t)floor(n*(0.5 + quadrant - 0.75*mu));
        int64_t descending = (int64_t)floor(n*(0.5 + quadrant + 0.75*mu));
        int64_t ring = n + 1 + ascending - descending;
        int64_t parity = 1 - (ring & 1);
        int64_t column = (ascending + descending - n + parity + 1)/2;
        column %= 4*n;
        if (column < 0) column += 4*n;
        return cap + (ring-1)*4*n + column;
    }

    double fraction = quadrant - floor(quadrant);
    /* Avoid cancellation in 1-|mu| close to either pole. */
    double scale = n * sqrt(3.0) * (hypot(x,y)/r) / sqrt(1.0+a);
    int64_t ascending = (int64_t)floor(fraction*scale);
    int64_t descending = (int64_t)floor((1.0-fraction)*scale);
    int64_t ring = ascending + descending + 1;
    int64_t column = (int64_t)floor(quadrant*ring);
    if (column >= 4*ring) column -= 4*ring;
    return mu >= 0.0 ? 2*ring*(ring-1) + column
                     : npix - 2*ring*(ring+1) + column;
}

/* Upper-bound lookup implements [edge[j], edge[j+1]). */
static int64_t find_shell(const double *edges, int64_t nshell, double radius)
{
    int64_t lo = 0, hi = nshell + 1;
    while (lo < hi) {
        int64_t mid = lo + (hi-lo)/2;
        if (edges[mid] <= radius) lo = mid+1;
        else hi = mid;
    }
    return lo-1;
}

int lightcone_has_openmp(void)
{
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

/*
 * Accumulate one chunk. itemsize is 4 or 8; positions are (N,3), C-contiguous.
 * masses=NULL means use constant_mass; otherwise masses has N doubles.
 * Caller validates array sizes. Do not update the same maps concurrently
 * from separate calls. Within one call, OpenMP updates are atomic.
 */
void lightcone_accumulate(
    const void *positions, int itemsize, int64_t count,
    const double *masses, double constant_mass,
    const double *observer, const double *offsets, int64_t ncopies,
    const double *edges, int64_t nshell, int64_t nside,
    double *maps, int threads)
{
    const int64_t npix = 12*nside*nside;
    const double rmin2 = edges[0]*edges[0];
    const double rmax2 = edges[nshell]*edges[nshell];
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t i = 0; i < count; ++i) {
        double px, py, pz;
        if (itemsize == 4) {
            const float *p = (const float *)positions + 3*i;
            px=p[0]; py=p[1]; pz=p[2];
        } else {
            const double *p = (const double *)positions + 3*i;
            px=p[0]; py=p[1]; pz=p[2];
        }
        px -= observer[0]; py -= observer[1]; pz -= observer[2];
        double mass = masses ? masses[i] : constant_mass;
        for (int64_t c = 0; c < ncopies; ++c) {
            double x=px+offsets[3*c], y=py+offsets[3*c+1], z=pz+offsets[3*c+2];
            double r2 = x*x + y*y + z*z;
            if (!(r2 > 0.0) || r2 < rmin2 || r2 >= rmax2) continue;
            double r = sqrt(r2);
            int64_t shell = find_shell(edges, nshell, r);
            if (shell < 0 || shell >= nshell) continue;
            int64_t pix = ring_pixel(nside, x, y, z, r);
            int64_t index = shell*npix + pix;
#ifdef _OPENMP
#pragma omp atomic update
#endif
            maps[index] += mass;
        }
    }
}
