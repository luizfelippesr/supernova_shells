"""
Contain functions that allow constructing simple fields on a grid
"""
import astropy.units as apu
import numpy as np
from shell.util import derive, rotate_field

pi = np.pi*apu.rad
sqrt2 = np.sqrt(2)

def uniform(grid, B, beta=0, gamma=0):
    """
    A uniform magnetic field

    Parameters
    ----------
    B : astropy.units.Quantity
        Amplitude of the magnetic field (originally along x)
    beta, gamma
        Angles for the rotation matrices (around y and z)

    Returns
    -------
    Bx, By, Bz
        List containing the three components
    """

    # Helical parallel to x
    Bvec = np.zeros((*grid.shape,3)) << B.unit
    Bvec[:,:,:,0] = B

    Bvec = rotate_field(Bvec, 0, beta, gamma)

    return [Bvec[...,i] for i in range(3)]


def simple_helical(grid, B, period=70*apu.pc, alpha=0, beta=0, gamma=0):
    """
    Computes a simple helical field

    Parameters
    ----------
    B : astropy.units.Quantity
        Amplitude of the magnetic field
    period
        Period of the helical field
    alpha, beta, gamma
        Angles for the rotation matrices

    Returns
    -------
    Bx, By, Bz
        List containing the three components
    """

    # Helical parallel to x
    Bx = np.ones(grid.shape) /sqrt2  * B
    arg = pi*grid.x/period
    By = np.cos(arg) /sqrt2 * B
    Bz = np.sin(arg) /sqrt2 * B

    return rotate_field([Bx, By, Bz], alpha, beta, gamma)



def simple_random(grid, Brms):
    """
    Generates a basic random field

    Parameters
    ----------
    grid : imagine.fields.grid.Grid
        An IMAGINE grid object
    Brms : float
        RMS value of the computed field

    Returns
    -------
    Bx, By, Bz
    """
    mu = 0; sigma = 1
    # Defines a random vector potential
    A_rnd = {} # Dictionary of vector components
    for i, c in enumerate(('x','y','z')):
        A_rnd[c] = np.random.normal(mu, sigma, grid.resolution.prod())
        A_rnd[c] = A_rnd[c].reshape(grid.resolution)

    # Prepares the derivatives to compute the curl
    dBi_dj ={}
    for i, c in enumerate(['x','y','z']):
        for j, d in enumerate(['x','y','z']):
            dj = (grid.box[j][-1]-grid.box[j][0])/float(grid.resolution[j])
            dBi_dj[c,d] = derive(A_rnd[c], dj, axis=j)

    # Computes the curl of A_rnd
    Brnd = {}
    Brnd['x'] = dBi_dj['z','y'] -  dBi_dj['y','z']
    Brnd['y'] = dBi_dj['x','z'] -  dBi_dj['z','x']
    Brnd['z'] = dBi_dj['y','x'] -  dBi_dj['x','y']

    # Finds normalization factor  to obtain correct Brms
    norm = np.sqrt(np.mean([Brnd[k]**2 for k in Brnd]))
    f = Brms/norm/np.sqrt(3)

    return Brnd['x']*f, Brnd['y']*f,  Brnd['z']*f


def helical_alt(grid, B, period=70*apu.pc):
    """
    Computes a simple helical field

    Parameters
    ----------
    B : float
        Amplitude of the initial magnetic field

    period
        Period of the helical field

    Returns
    -------
    Bx, By, Bz
    """

    # Prepares vector basis
    B = np.stack(B)
    amplitude = np.linalg.norm(B)
    i, j, k = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    u = B / amplitude
    w = np.cross(u, j)
    w /= np.linalg.norm(w)
    v = np.cross(w, u)
    v /= np.linalg.norm(v)

    # Position vector
    r = np.stack([grid.x, grid.y, grid.z], axis=-1)
    arg = pi * (r @ u) / period

    # Helical parallel to x
    Bu = np.ones(grid.x.shape) /sqrt2  * amplitude
    Bv = np.cos(arg) /sqrt2 * amplitude
    Bw = np.sin(arg) /sqrt2 * amplitude

    Bx = Bu*(u@i) + Bv*(v@i) + Bw*(w@i)
    By = Bu*(u@j) + Bv*(v@j) + Bw*(w@j)
    Bz = Bu*(u@k) + Bv*(v@k) + Bw*(w@k)

    return Bx, By, Bz


def helical(grid, B, period=70*apu.pc):
    """
    Computes a simple helical field

    Note
    ----
    Bogus version of the helical field


    Parameters
    ----------
    B : list
        List containing the x, y and z magnitudes
    period
        Period of the helical field

    Returns
    -------
    Bx, By, Bz
    """

    # Helical parallel to x
    Bx = np.ones(grid.x.shape) /sqrt2  * B[0]
    arg = pi*grid.x/period
    By = np.cos(arg) /sqrt2 * B[0]
    Bz = np.sin(arg) /sqrt2 * B[0]

    # Helical parallel to y
    arg = pi*grid.y/period
    Bx += np.ones(grid.y.shape) /sqrt2  * B[1]
    By += np.cos(arg) /sqrt2 * B[1]
    Bz += np.sin(arg) /sqrt2 * B[1]

    # Helical parallel to z
    arg = pi*grid.z/period
    Bx += np.ones(grid.z.shape) /sqrt2  * B[2]
    By += np.cos(arg) /sqrt2 * B[2]
    Bz += np.sin(arg) /sqrt2 * B[2]

    return Bx, By, Bz
