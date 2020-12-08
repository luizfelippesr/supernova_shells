import astropy.units as u
import numpy as np

from imagine.simulators import Simulator
from shell.observable import compute_stokes_parameters
from scipy.interpolate import RegularGridInterpolator


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
        # TODO UNITS HAVE TO BE ADJUSTED!
        # Cosmic ray particle's energy needs to be accounted for
        sync_data *= u.K/u.pc/u.microgauss/u.microgauss*u.cm**3

        # Now, we need to interpolate to the original image resolution
        # First, store the available coordinates
        x = self.grid.x[:,0,0].to_value(u.kpc)
        y = self.grid.y[0,:,0].to_value(u.kpc)
        # Setup the interpolator
        interpolator = RegularGridInterpolator(points=(x, y),
                                               values=sync_data.to_value(output_units),
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

        result = interpolator(interp_points) << output_units

        return result.ravel()


