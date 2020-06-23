import numpy as np
from numba import njit

@njit(parallel=True)
def derive(V, dx, axis=0, order=2):
    """
    Computes the numerical derivative of a function specified over a
    3 dimensional uniform grid. Uses second order finite differences.

    Obs: extremities will use forward or backwards finite differences.

    Parameters
    ----------
    V : array_like
        NxNxN array
    dx : float
        grid spacing
    axis : int
        specifies over which axis the derivative should be performed. Default: 0.
    order : int
        order of the finite difference method. Default: 2

    Returns
    -------
    same as V
        The derivative, dV/dx

    Note
    ----
    This was ariginally part of the GalMag package
    """
    dVdx = np.empty_like(V)

    if axis==0:
        if order==2:
            dVdx[1:-1,:,:] = (V[2:,:,:] - V[:-2,:,:])/2.0/dx
            dVdx[0,:,:]  = (-3.0*V[0,:,:]  +4.0*V[1,:,:]  - V[2,:,:])/dx/2.0
            dVdx[-1,:,:] = ( 3.0*V[-1,:,:] -4.0*V[-2,:,:] + V[-3,:,:])/dx/2.0
        elif order==4:
            dVdx[2:-2,:,:] = ( V[:-4,:,:]/12.0 - V[4:,:,:]/12.0
                             - V[1:-3,:,:]*(2./3.) + V[3:-1,:,:]*(2./3.) )/dx

            a0 = -25./12.; a1=4.0; a2=-3.0; a3=4./3.; a4=-1./4.
            dVdx[0:2,:,:] = ( V[0:2,:,:]*a0 + V[1:3,:,:]*a1
                            + V[2:4,:,:]*a2 + V[3:5,:,:]*a3
                            + V[3:5,:,:]*a4 )/dx

            dVdx[-2:,:,:] = - ( V[-2:,:,:]*a0 + V[-3:-1,:,:]*a1
                              + V[-4:-2,:,:]*a2 + V[-5:-3,:,:]*a3
                              + V[-6:-4,:,:]*a4 )/dx
        else:
            raise ValueError('Only order 2 and 4 are currently implemented.')
    elif axis==1:
        if order==2:
            dVdx[:,1:-1,:] = (V[:,2:,:] - V[:,:-2,:])/2.0/dx
            dVdx[:,0,:]  = (-3.0*V[:,0,:]  +4.0*V[:,1,:]  - V[:,2,:])/dx/2.0
            dVdx[:,-1,:] = ( 3.0*V[:,-1,:] -4.0*V[:,-2,:] + V[:,-3,:])/dx/2.0
        elif order==4:
            dVdx[:,2:-2,:] = ( V[:,:-4,:]/12.0 - V[:,4:,:]/12.0
                             - V[:,1:-3,:]*(2./3.) + V[:,3:-1,:]*(2./3.) )/dx

            a0 = -25./12.; a1=4.0; a2=-3.0; a3=4./3.; a4=-1./4.
            dVdx[:,0:2,:] = ( V[:,0:2,:]*a0 + V[:,1:3,:]*a1
                            + V[:,2:4,:]*a2 + V[:,3:5,:]*a3
                            + V[:,3:5,:]*a4 )/dx

            dVdx[:,-2:,:] = - ( V[:,-2:,:]*a0 + V[:,-3:-1,:]*a1
                              + V[:,-4:-2,:]*a2 + V[:,-5:-3,:]*a3
                              + V[:,-6:-4,:]*a4 )/dx
        else:
            raise ValueError('Only order 2 and 4 are currently implemented.')
    elif axis==2:
        if order==2:
            dVdx[:,:,1:-1] = (V[:,:,2:] - V[:,:,:-2])/2.0/dx
            dVdx[:,:,0]  = (-3.0*V[:,:,0]  +4.0*V[:,:,1]  - V[:,:,2])/dx/2.0
            dVdx[:,:,-1] = ( 3.0*V[:,:,-1] -4.0*V[:,:,-2] + V[:,:,-3])/dx/2.0
        elif order==4:
            dVdx[:,:,2:-2] = ( V[:,:,:-4]/12.0 - V[:,:,4:]/12.0
                             - V[:,:,1:-3]*(2./3.) + V[:,:,3:-1]*(2./3.) )/dx

            a0 = -25./12.; a1=4.0; a2=-3.0; a3=4./3.; a4=-1./4.
            dVdx[:,:,0:2] = ( V[:,:,0:2]*a0 + V[:,:,1:3]*a1
                            + V[:,:,2:4]*a2 + V[:,:,3:5]*a3
                            + V[:,:,3:5]*a4 )/dx

            dVdx[:,:,-2:] = - ( V[:,:,-2:]*a0 + V[:,:,-3:-1]*a1
                              + V[:,:,-4:-2]*a2 + V[:,:,-5:-3]*a3
                              + V[:,:,-6:-4]*a4 )/dx
        else:
            raise ValueError('Only order 2 and 4 are currently implemented.')

    return dVdx
