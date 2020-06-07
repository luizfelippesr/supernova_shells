from scipy.integrate import trapz, cumtrapz #, simps
import numpy as np
import astropy.units as u

pi = np.pi*u.rad # convenience
integrate = trapz 

def compute_stokes_parameters(grid, wavelength, Bx, By, Bz,
                              ne, ncr, gamma=1.0):
    """
    Computes Stokes I, Q, U integrated along z axis
    
    Parameters
    ----------
    grid : imagine.fields.grid.Grid
        A *cartesian* IMAGINE grid object
    wavelength : float
        The wavelength of the observatin
    Bx, By, Bz : astropy.units.Quantity
        Magnetic field components
    ne : astropy.units.Quantity
        Thermal electron density
    ncr : astropy.units.Quantity
        Cosmic ray electron density
        
    Returns
    -------
    I : astropy.units.Quantity
        Total synchroton intensity (arbitrary units)
    Q : astropy.units.Quantity
        Stokes Q parameter intensity (arbitrary units)
    U : astropy.units.Quantity
        Stokes U parameter intensity (arbitrary units)
    """
    Bperp2 = Bx**2+By**2

    # Total synchrotron intensity
    I = integrate(Bperp2*ncr, grid.z, axis=2)

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

    # Intrinsic polarization degree
    p0 = (gamma+1)/(gamma+7/3)
    
    # # Stokes Q and U
    U = p0*integrate( ncr * Bperp2 * np.sin(2*psi) , grid.z, axis=2);
    Q = p0*integrate( ncr * Bperp2 * np.cos(2*psi) , grid.z, axis=2);
    
    return I, U, Q

def compute_Psi(U, Q):
    """
    Computes the observed polarization angle
    
    Parameters
    ----------
    U, Q : astropy.units.Quantity
        Stokes U and Q
    
    Returns
    -------
    psi 
        Polarization angle
    """
    pi = np.pi*u.rad
    psi = np.arctan2(U,Q)
    # Keeps angles in the [-pi pi] interval
    psi[psi>pi] = psi[psi>pi]-2*pi
    psi[psi<-pi] = psi[psi<-pi]+2*pi
    return psi

        
def compute_RM(Psi1, Psi2, lambda1, lambda2):
    """
    Computes Faraday rotation measure from two frequencies and angles
    """
    return (Psi2-Psi1) / (lambda2**2 - lambda1**2)
