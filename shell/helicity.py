"""
This module contains tools that allow estimating current helicity
from faraday rotation measure, polarization angle and polarized 
intensity.
"""
import numpy as np
import astropy.units as u
import astropy.constants as c
import numba as nb
from galmag.util import derive

# Convenience
pi = np.pi*u.rad
# B/RM conversion factor
RM_constant = 1/(0.812*u.rad/u.m**2)*u.microgauss*(u.cm**-3)*u.pc

def average_helicity(RM, I, PA, wavelengths, ne, ncr, L, x, y, 
                     boundary_radius=5, output_current=False):
    """
    Computes average helicity current from observables
    
    Parameters
    ----------
    RM : astropy.units.Quantity
        2-array containing the observed rotation measure 
    I : list
        2-array containing observed total synchrotron intensity
    PA : list
        2-array the observed polarization angle
    ne : astropy.units.Quantity
        Number density of thermal electrons (scalar)
    ncr : astropy.units.Quantity
        Number density of cosmic ray electrons (scalar)
    L : astropy.units.Quantity
        Estimated size of the remnant (scalar)
    boundary_radius : int
        Number of mesh points used for the path integral in the
        computation of helicity
    output_current : bool
        If True, returns the current density
    
    Returns
    -------
    Hz : astropy.units.Quantity
        2-array with the average current helicity
    Jz : astropy.units.Quantity
        Available only if output_current is True. 
        Contains 2-array with the average current helicity
    """

    # Gets grid shape
    (N,N) = RM.shape
    
    # Estimates Bz
    Bz = 2 * RM / ne  / (2*L) * RM_constant
    
    # De-rotates the polarization angles
    Psi = [ PA[i] - wavelengths[i]**2*RM for i in (0,1)]
    PA_avg = (Psi[0] + Psi[1])/2.
    # Averages the intensity
    I_avg = (I[0] + I[1])/2.
    
    Bxy = np.sqrt(I_avg / ncr / (2*L))    
    Bx = Bxy * np.cos(PA_avg - pi/2.)
    By = Bxy * np.sin(PA_avg - pi/2.)

    # Computes Jz (with numba-accelerated function)
    Jz = _compute_Jz(x.to_value(u.pc), y.to_value(u.pc),
                     Bx.to_value(u.microgauss), 
                     By.to_value(u.microgauss),
                     boundary_radius=boundary_radius)
    # Restores and adjusts units
    Jz = Jz * u.microgauss*u.pc * (c.c/4*np.pi) /u.pc/u.pc
    Jz = Jz.to(u.microgauss/u.s)
    
    # z-component of Helicity
    Hz = Jz * Bz
    
    if not output_current:
        return Hz
    else:
        return Hz, Jz
    
    
@nb.njit(parallel=True)
def _compute_Jz(x, y, By, Bx, boundary_radius=1):    
    
    #boundary radius excluding central pixel
    #e.g. s=1 corresponds to 3x3 pixel regions
    #e.g. s=2 corresponds to 5x5 pixel regions
    s = boundary_radius
    
    # Initializes with an array of NaNs
    # (to avoid artificially filling with zeroes)
    Jz = np.empty_like(Bx) *np.nan
    
    Nx, Ny = x.size, y.size
    
    # Idea: (side*B) over the path divided by the area
    for j in nb.prange(s,Ny-s):
        for i in nb.prange(s,Nx-s):        
            
            delta_x = x[i+s] - x[i-s]
            delta_y = y[j+s] - y[j-s]
            
            # Upper path   ->
            sum_B_para = np.sum( Bx[i-s:i+s+1, j-s] )
            Jz[i,j] = sum_B_para*delta_x
            # Right path  \/    
            sum_B_para = np.sum( By[i+s, j-s:j+s+1] )
            Jz[i,j] += sum_B_para*delta_y
            # Bottom path  <-
            sum_B_para = np.sum(Bx[i-s:i+s+1, j+s])
            Jz[i,j] -= sum_B_para*delta_x
            # Left path  /\
            sum_B_para = np.sum(By[i-s, j-s:j+s+1])
            Jz[i,j] -= sum_B_para*delta_y
            # To finally get Jz we need to divide by the surface within the path
            Jz[i,j] /= delta_x*delta_y
            Jz[i,j] /= 3.
            
#         Jz[i,j] = Jz[i,j] - sum(Bx[i+s,j-s:j+s]) * ( x[i+s,j+s]-x[i+s,j-s] ) # \/
#         Jz[i,j] = Jz[i,j] - sum(By[i-s:i+s,j-s]) * ( y[i+s,j-s]-y[i-s,j-s] ) # 
#         Jz[i,j] = Jz[i,j] - sum(By[i-s:i+s,j+s]) * ( y[i-s,j+s]-y[i+s,j+s] )
#         Jz[i,j] = Jz[i,j] / abs(x[i-s,j-s]-x[i-s,j+s]) / abs(y[i+s,j-s]-y[i-s,j-s]) / 3
    return Jz
            
    
def compute_theoretical_Hz(grid, B):
    """
    Computes the expected average helicity
    
    Parameters
    ----------
    grid : imagine.fields.grid.Grid
        A *cartesian* IMAGINE grid object
    B : list
        A list containing the astropy.units.Quantity objects 
        with the x, y, and z-components of the magnetic field
    
    Returns
    -------
    Jz_mean
        The z-component of the current, averaged over the z direction
    Hz_mean
        The z-component of the current helicity, averaged over the z direction
    """
    Bx, By, Bz = B
    # Computes the z component of the curl of B
    dx = (grid.box[0][-1]-grid.box[0][0])/float(grid.resolution[0])
    dBy_dx = derive(By.value, dx.value, axis=0)*By.unit/dx.unit

    dy = (grid.box[1][-1]-grid.box[1][0])/float(grid.resolution[1])
    dBx_dy = derive(Bx.value, dy.value, axis=1)*Bx.unit/dy.unit

    curl_Bz = dBy_dx - dBx_dy

    Jz = curl_Bz * c.c/(4*np.pi)
    Hz = Jz * Bz
    
    return Jz.mean(axis=2).to(u.microgauss/u.s), Hz.mean(axis=2).to(u.microgauss**2/u.s)
