from scipy.integrate import trapz, cumtrapz
import numpy as np
import astropy.units as u

pi = np.pi*u.rad # convenience

def compute_stokes_parameters(grid, wavelength, Bx, By, Bz, ne, ncr):
    """
    Computes Stokes I, Q, U integrated along z axis
    
    Parameters
    ----------
    grid : imagine.fields.grid.Grid
        A *cartesian* IMAGINE grid object
    wavelength : float
        The wavelength of the observatin
    Bx, By, Bz : numpy.ndarray
        Magnetic field components
    ne : numpy.ndarray
        Thermal electron density
    ncr : numpy.ndarray
        Cosmic ray electron density
        
    Returns
    -------
    I : np.ndarray
        Total synchroton intensity (arbitrary units)
    Q : np.ndarray
        Stokes Q parameter intensity (arbitrary units)
    U : np.ndarray
        Stokes U parameter intensity (arbitrary units)
    """
    Bperp2 = Bx**2+By**2

    # Total synchrotron intensity
    I = trapz(Bperp2*ncr, grid.z, axis=2)

    # Intrinsic polarization angles
    psi0 = np.arctan(By/Bx) + pi/2
    # Keeps angles in the [-pi pi] interval
    psi0[psi0>pi] = psi0[psi0>pi]-2*pi
    psi0[psi0<-pi] = psi0[psi0<-pi]+2*pi

    # Cummulative Faraday rotation (i.e. rotation up to a given depth)
    if wavelength != 0:
        integrand = Bz.to_value(u.microgauss)*ne.to_value(u.cm**-3)
        RM = (0.812*u.rad/u.m**2) * cumtrapz(integrand, 
                                             grid.z.to_value(u.pc),
                                             axis=2, initial=0)
    else:
        # Avoids unnecessary calculation
        RM = 0.0 * u.rad/u.m**2

    # Rotated polarization angle grid
    psi = psi0 + wavelength**2*RM

    # # Stokes Q and U
    U = trapz( ncr * Bperp2 * np.sin(2*psi) , grid.z, axis=2);
    Q = trapz( ncr * Bperp2 * np.cos(2*psi) , grid.z, axis=2);
    
    return I, U, Q