import numba as nb
import numpy as np

class FieldTransformer:
    """
    Given an initial magnetic field and an inital gas density,
    this helper class can compute the resulting magnetic field
    and gas distribution produced by a supernova (SN) explosion in
    that medium.  The helper is initialized has to be initialized
    with a coordinate grid and a model for the SN shell.

    Parameters
    ----------
    grid : imagine.fields.grid.Grid
        A *cartesian* IMAGINE grid object
    shell_model : shell.ShellModel
        An instance of ShellModel, which characterizes the dynamics
        of the SN shell and contains the required coordinate
        transformations
    """
    def __init__(self, grid, shell_model):
        # final distance r
        r = grid.r_spherical

        # initial distance r0
        r0 = shell_model.initial_radius(r)

        # dr/dr0
        dr_dr0 = shell_model.dr_dr0(r, r0)

        # initial Lagrangian mesh
        x0 = r0*grid.x/r;
        y0 = r0*grid.y/r;
        z0 = r0*grid.z/r;

        # Computes and stores the components of the Jacobian matrix
        self.dx_dx0 = dr_dr0 * (x0/r0)**2 + r/r0*(1-(x0/r0)**2)
        self.dy_dy0 = dr_dr0 * (y0/r0)**2 + r/r0*(1-(y0/r0)**2)
        self.dz_dz0 = dr_dr0 * (z0/r0)**2 + r/r0*(1-(z0/r0)**2)

        self.dx_dy0 = (dr_dr0 - r/r0) * x0*y0/r0**2
        self.dy_dx0 = self.dx_dy0

        self.dx_dz0 = (dr_dr0 - r/r0) * x0*z0/r0**2
        self.dz_dx0 = self.dx_dz0

        self.dy_dz0 = (dr_dr0 - r/r0) * y0*z0/r0**2;
        self.dz_dy0 = self.dy_dz0

        self._inv_J = None

    @property
    def inv_J(self):
        """
        The inverse of the determinant of the Jacobian matrix, :math:`1/|J|`,
        with

        ..math::

            J_{ij} = \frac{\partial x_i}{\partial x_{0j}}

        """
        if self._inv_J is None:
            self._inv_J = _get_inv_J(self.dx_dx0, self.dx_dy0, self.dx_dz0,
                                     self.dy_dx0, self.dy_dy0, self.dy_dz0,
                                     self.dz_dx0, self.dz_dy0, self.dz_dz0)
        return self._inv_J

    def transform_density_field(self, n0):
        r"""
        Transforms density field using:

        ..math::

            \rho = \frac{1}{|J|} \rho_0

        See the property :py:data:`inv_J` for details.

        Parameters
        ----------
        n0 : numpy.ndarray
            Initial ambient gas density distribution

        Returns
        -------
        n : numpy.ndarray
            Final density distribution in the remnant
        """
        # Checks shape to avoid mistakes
        assert n0.shape == self.dx_dy0.shape

        # Computes the new field
        return self.inv_J * n0

    def transform_magnetic_field(self, Bx0, By0, Bz0):
        r"""
        Transforms magnetic field using:

        ..math::

            B_i = \frac{1}{|J|} J_{ij} B_{0j}


        See also property :py:data:`inv_J`.

        Parameters
        ----------
        Bx0, By0, Bz0 : numpy.ndarray
            Initial ambient magnetic field components

        Returns
        -------
        Bx, By, Bz : numpy.ndarray
            Final magnetic field in the remnant
        """
        # Checks shapes to avoid mistakes
        assert Bx0.shape == By0.shape == Bz0.shape == self.dx_dy0.shape

        # Computes the new field
        Bx = self.inv_J * (Bx0*self.dx_dx0 + By0*self.dx_dy0 + Bz0*self.dx_dz0)
        By = self.inv_J * (Bx0*self.dy_dx0 + By0*self.dy_dy0 + Bz0*self.dy_dz0)
        Bz = self.inv_J * (Bx0*self.dz_dx0 + By0*self.dz_dy0 + Bz0*self.dz_dz0)

        return Bx, By, Bz

    def __call__(self, n0, B0_components_list):
        """
        Transforms an initial ambient density field and ambient magnetic field
        to the solution after the SN explosion (i.e. in a SN remnant).

        Parameters
        ----------
        n0 : numpy.ndarray
            Initial ambient gas density distribution
        B0_components_list : list
            List containing the Bx, By and Bz components of initial ambient
            magnetic field

        Returns
        -------
        n : numpy.ndarray
            Final density distribution in the remnant
        final_components_list : tuple
            Magnetic field components (Bx, By, Bz) in the SN remnant
        """
        n = self.transform_density_field(n0)
        final_components_list = self.transform_magnetic_field(*B0_components_list)

        return n, final_components_list


# Technical note: numbas parallel mode is not working for some reason
# Nevertheless, just using njit makes it ~54 times (!) faster
@nb.njit
def _get_inv_J(*args):
    """
    Takes 9 arrays corresponding to the derivatives dxi_dx0j
    over the whole grid and compute the inverse of the
    Jacobian, inv_J
    """
    # Makes sure there are 9 arrays
    assert len(args) == 9
    # Uses first array as example
    dx_dx0 = args[0]
    # Makes sure all have the same shape
    for arg in args:
        assert arg.shape == dx_dx0.shape
    # Creates the output array
    inv_J = np.empty_like(dx_dx0)

    # Loops through the grid
    for i in range(dx_dx0.size):
        # Constructs a 3x3 Jacobian at a given gridpoint
        tmp_array = np.empty((3,3))
        for j, arg in enumerate(args):
            tmp_array.ravel()[j] = arg.ravel()[i]
        # Computes inverse of det(J) at that point
        inv_J.ravel()[i] = 1./np.linalg.det(tmp_array)

    return inv_J
