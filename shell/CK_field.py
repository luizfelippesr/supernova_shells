from imagine.fields import BaseGrid
from scipy.special import jv
import astropy.units as apu
from shell.util import rotate_field
import numpy as np

def CK_magnetic_field(grid, B=1*apu.microgauss, m=0,
                      period=50*apu.pc, period_z=np.inf*apu.pc,
                      x_shift=0*apu.pc, y_shift=0*apu.pc, z_shift=0*apu.pc,
                      alpha=0, beta=0, gamma=0):
    """
    Computes a Chandrasekhar-Kendall field, i.e. a force-field corresponding
    to eigenfnction of the curl derivatives.

    Parameters
    ----------
    B : astropy.units.Quantity
        Amplitude of the magnetic field
    period : astropy.units.Quantity
        Period of the helical field (in x, y)
    period_z : astropy.units.Quantity
        Period of the helical field (in z)
    alpha, beta, gamma
        Angles for the rotation matrices
    x_shift, y_shift, z_shift
        Translation of the CK solution

    Returns
    -------
    [Bx, By, Bz] : list
        List containing the three components of the field
    """
    [Bx, By, Bz] = _CK_magnetic_field_core(ShifftedGrid(grid,
                                                        x_shift,
                                                        y_shift,
                                                        z_shift),
                                           B=B, m=m, period=period,
                                           period_z=period_z)

    return rotate_field([Bx, By, Bz], alpha, beta, gamma)


def _CK_magnetic_field_core(grid, B=1*apu.microgauss,
                            m=0, period=50*apu.pc,
                            period_z=np.inf*apu.pc):

    # Notation adjustment
    k = 1/period_z
    g = 1/period
    l = np.sqrt(g**2 + k**2)

    # Shifted grid
    r = grid.r_cylindrical
    phi = grid.phi
    z = grid.z

    # Intermediate quantities (which appear repeated)
    gr = g*r
    kz = k*z
    J = jv(m, gr)
    dJdr = (jv(m-1, gr) - jv(m+1, gr))/2
    arg = m*phi + kz*apu.rad
    cos_arg = np.cos(arg)

    # Field in cylindrical coordinates
    Br =   -B/g*( m*l/gr*J + k*dJdr )*np.sin(arg)
    Bphi = -B/g*( m*k/gr*J + l*dJdr )*cos_arg
    Bz =    B*J*cos_arg

    # Conversion to cartesian
    Bx = Br*grid.cos_phi - Bphi*grid.sin_phi
    By = Br*grid.sin_phi + Bphi*grid.cos_phi

    return [Bx, By, Bz]


class ShifftedGrid(BaseGrid):
    """
    Creates a new grid object where the cartesian coordinates
    are shifted based on a previous IMAGINE grid.

    Parameters
    ----------
    grid : imagine.fields.grid.Grid
        Previous IMAGINE grid
    x, y, z : astropy.units.Quantity
        The amount of shift in each of the coordinates
    """
    def __init__(self, grid, x, y, z):
        assert grid.grid_type == 'cartesian', 'Only cartesian grid is currently supported'

        # Base class initialization
        super().__init__(grid.box, grid.resolution)

        # Subclass specific attributes
        self.grid_type = grid.grid_type

        # Prepares shifted coordinates
        self._coordinates = {c: getattr(grid, c) - v
                             for c, v in zip(['x', 'y', 'z'],
                                             [x, y, z])}
        # Corrects box parameters
        if self.grid_type == 'cartesian':
            self.box[0] = [self.x[0,0,0], self.x[-1,-1,-1]]
            self.box[1] = [self.y[0,0,0], self.y[-1,-1,-1]]
            self.box[2] = [self.z[0,0,0], self.z[-1,-1,-1]]
        else:
            # TODO implement other cases
            raise ValueError('Only cartesian grid is currently supported')

    def generate_coordinates(self):
        return self._coordinates
