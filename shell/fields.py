"""
Contain functions that allow constructing simple fields on a grid
"""
import astropy.units as u
import numpy as np
from galmag.util import derive
pi = np.pi*u.rad
sqrt2 = np.sqrt(2)

def uniform(grid, B):
    """Uniform field"""
    return [Bi*np.ones(grid.shape) for Bi in B]

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


def helical(grid, B, period=70*u.pc):
    """
    Computes a simple helical field
    
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
