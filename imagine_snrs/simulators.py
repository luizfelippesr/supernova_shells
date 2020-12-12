import astropy.units as u
import numpy as np

from imagine.simulators import Simulator
from shell.observable import compute_stokes_parameters
from scipy.interpolate import RegularGridInterpolator

import astropy.units as u

class SimpleSynchrotron(Simulator):
    """
    Example simulator to illustrate
    """
    # Class attributes
    SIMULATED_QUANTITIES = ['sync']
    REQUIRED_FIELD_TYPES = ['magnetic_field',
                            'cosmic_ray_electron_density',
                            'thermal_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']

    def __init__(self, measurements, distance, gamma=1.0, interp_method='nearest'):
        super().__init__(measurements)
        self.gamma = gamma
        self.Stokes = {}
        self.interp_method = interp_method
        self.distance = distance

    def _units(self, key):
        if key[0] == 'sync':
            if key[3] in ('I', 'Q', 'U', 'PI'):
                return u.K
            elif key[3] == 'PA':
                return u.rad
            else:
                raise ValueError
        elif key[0] == 'fd':
            return u.rad/u.m/u.m
        elif key[0] == 'dm':
            return u.pc/u.cm**3
        else:
            raise ValueError


    def _sync_constant(self, gamma):
        from math import sqrt, pi
        from astropy.constants import c, e, m_e
        e = e.esu

        #return ( (sqrt(3) * e**3) / (8*pi*m_e*c**2)
                #*(4*pi*m_e*c/(3*e))**((1-gamma)/2) ) * c**((1-gamma)/2)
        A = sqrt(3) * e**3 / (8*pi*m_e*c**2)
        B = (4*pi*m_e*c**2 / (3*e))**((1-gamma)/2)
        return A*B

    def simulate(self, key, coords_dict, realization_id, output_units):

        _, freq, _, flag = key

        if flag not in self.Stokes:
            # Accesses fields and grid
            grid = self.grid
            ne = self.fields['thermal_electron_density']
            ncr = self.fields['cosmic_ray_electron_density'][:,:,:,0]
            B = self.fields['magnetic_field']
            Bx, By, Bz = [B[:,:,:,i] for i in range(3)]

            wavelength = (freq*u.GHz).to(u.cm, equivalencies=u.spectral())

            I, U, Q = compute_stokes_parameters(grid, wavelength,
                                                Bx, By, Bz,
                                                ne, ncr, gamma=self.gamma)
            self.Stokes['I'] = I
            self.Stokes['Q'] = Q
            self.Stokes['U'] = U

        sync_data = self.Stokes.pop(flag)
        sync_data_units = sync_data.unit

        # Now, we need to interpolate to the original image resolution
        # First, store the available coordinates
        x = self.grid.x[:,0,0].to_value(u.kpc)
        y = self.grid.y[0,:,0].to_value(u.kpc)

        # Setup the interpolator
        interpolator = RegularGridInterpolator(points=(x, y),
                                               values=sync_data.value,
                                               bounds_error=False,
                                               fill_value=0,
                                               method=self.interp_method)

        # Now we convert the original Galactic coordinates into x and y
        # under the assumption that the centres of the images coincide
        coords = self.output_coords[key]
        nx, ny = coords['shape']

        lon_range = np.abs(coords['lon_max']-coords['lon_min'])
        lat_range = np.abs(coords['lat_max']-coords['lat_min'])

        x_range = lon_range.to_value(u.rad) * self.distance
        y_range = lat_range.to_value(u.rad) * self.distance

        x_target = np.linspace(-x_range/2, x_range/2, nx)
        y_target = np.linspace(-y_range/2, y_range/2, ny)
        x_target, y_target = np.meshgrid(x_target, y_target)

        interp_points = np.array([x_target, y_target]).T

        result = interpolator(interp_points) * sync_data_units

        # Adjusts the units
        sync_constant = self._sync_constant(self.gamma)

        # The following is a hack to deal with a missing equivalency in
        # astropy units
        B_unit_adj = (1./u.gauss) * (u.Fr/u.cm**2)  # This should be 1
        sync_constant *= B_unit_adj**((self.gamma+1)/2)

        result = result * sync_constant

        # The result, so far, corresponds to a surface density of luminosity,
        # i.e. the energy per area in the remnant, but we want flux density,
        # the energy per unit area at the detector, thus
        result *= (x_range/nx)*(y_range/ny)/(self.distance)**2

        # Finally, we want the surface brightness (the flux density per
        # detector solid angle), i.e. we nee to account for the (pencil) beam
        beam_size = (lon_range/nx)*(lat_range/ny)
        result /= beam_size

        # Converts into brightness temperature
        result = result.to(u.K,
                           equivalencies=u.brightness_temperature(freq<<u.GHz))

        return result.ravel()


