import astropy.units as u
import numpy as np

from imagine.simulators import Simulator
import shell.observable as obs
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as scipy_rotation
import astropy.units as u


class SimpleSynchrotron(Simulator):
    """
    Simulates the radio images associated with the synchrotron emission signal
    of an object at a large distance.

    It assumes that the distances are sufficient large (relative to the object
    size) to approximate the projections as parallel.  The grid where the
    Fields are constructed is placed at distance `distance`, with the centre
    of the smallest z face coinciding with the centre of the simulated image,
    and x and y corresponding to longitudes and latitudes, respectively.

    Parameters
    ----------
    measurements : imagine.observables.Measurements
        Observational data
    distance : astropy.units.Quantity
        The distance to the object in apropriate units (e.g. kpc)
    gamma : float
        The assumed spectral index for the cosmic ray electron distribution
    beam_kernel_sd : float
        If different from `None`, the resulting signal is convolved with
        a gaussian kernel with standard deviation `beam_kernel_sd` (in pixels).
        Otherwise, a pencil beam is assumed.
    interp_method : str
        The interpolation method used by `scipy.interpolate.RegularGridInterpolator`
        to interpolate the resulting images to the same dimensions as the
        images in the `measurements`.
    """
    # Class attributes
    SIMULATED_QUANTITIES = ['sync', 'fd']
    REQUIRED_FIELD_TYPES = ['magnetic_field',
                            'cosmic_ray_electron_density',
                            'thermal_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']

    def __init__(self, measurements, distance, gamma=1.0, wavelength_factor=1.01,
                 beam_kernel_sd=None, backlit_RM=True, interp_method='nearest'):
        super().__init__(measurements)
        self.gamma = gamma
        self.Stokes = {}
        self.interp_method = interp_method
        self.distance = distance
        self.beam_kernel_sd = beam_kernel_sd
        self.backlit_RM = backlit_RM
        self.wavelength_factor = wavelength_factor

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

        obs_name, freq, _, flag = key

        if obs_name == 'fd':
            flag = 'fd'
            if self.backlit_RM:
                self.Stokes['fd'] = obs.compute_fd(
                    self.grid, self.fields['magnetic_field'][:,:,:,2],
                    self.fields['thermal_electron_density'],
                    beam_kernel_sd=self.beam_kernel_sd)

        if flag not in self.Stokes:
            # Accesses fields and grid
            grid = self.grid
            ne = self.fields['thermal_electron_density']
            ncr = self.fields['cosmic_ray_electron_density'][:,:,:,0]
            B = self.fields['magnetic_field']
            Bx, By, Bz = [B[:,:,:,i] for i in range(3)]

            wavelength = (freq*u.GHz).to(u.cm, equivalencies=u.spectral())

            I, U, Q = obs.compute_stokes_parameters(grid, wavelength,
                                                    Bx, By, Bz,
                                                    ne, ncr, gamma=self.gamma,
                                                    beam_kernel_sd=self.beam_kernel_sd)
            self.Stokes['I'] = I
            self.Stokes['Q'] = Q
            self.Stokes['U'] = U

            if ('fd' in [k[0] for k in self.observables]) and (not self.backlit_RM):
                wavelength2 = wavelength * self.wavelength_factor
                _, U2, Q2 = obs.compute_stokes_parameters(grid, wavelength2,
                                                          Bx, By, Bz,
                                                          ne, ncr, gamma=self.gamma,
                                                          beam_kernel_sd=self.beam_kernel_sd)
                Psi1 = obs.compute_Psi(U, Q)
                Psi2 = obs.compute_Psi(U2, Q2)
                RM = obs.compute_RM(Psi1, Psi2, wavelength, wavelength2)
                self.Stokes['fd'] = RM

        out_data = self.Stokes.pop(flag)
        out_data_units = out_data.unit

        # Now, we need to interpolate to the original image resolution
        # First, store the available coordinates
        x = self.grid.x[:,0,0].to_value(u.kpc)
        y = self.grid.y[0,:,0].to_value(u.kpc)

        # Setup the interpolator
        interpolator = RegularGridInterpolator(points=(x, y),
                                               values=out_data.value,
                                               bounds_error=False,
                                               fill_value=0,
                                               method=self.interp_method)

        # Now we convert the original Galactic coordinates into x and y
        # under the assumption that the centres of the images coincide
        coords = self.output_coords[key]
        nx, ny = coords['shape']

        lon_range = np.abs(coords['lon_max'] - coords['lon_min'])
        lat_range = np.abs(coords['lat_max'] - coords['lat_min'])

        x_range = lon_range.to_value(u.rad) * self.distance
        y_range = lat_range.to_value(u.rad) * self.distance

        x_target = np.linspace(-x_range/2, x_range/2, nx)
        y_target = np.linspace(-y_range/2, y_range/2, ny)
        x_target, y_target = np.meshgrid(x_target, y_target)

        interp_points = np.array([x_target, y_target]).T

        result = interpolator(interp_points) * out_data_units

        # Adjusts the units
        if obs_name == 'sync':
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


class LoS_integrator(Simulator):
    """
    Simulates the radio images associated with the synchrotron emission signal
    of an object at a large distance.

    It assumes that the distances are sufficient large (relative to the object
    size) to approximate the projections as parallel.  The grid where the
    Fields are constructed is placed at distance `distance`, with the centre

    Parameters
    ----------
    measurements : imagine.observables.Measurements
        Observational data
    distance : astropy.units.Quantity
        The distance to the object in apropriate units (e.g. kpc)
    gamma : float
        The assumed spectral index for the cosmic ray electron distribution
    beam_kernel_sd : float
        If different from `None`, the resulting signal is convolved with
        a gaussian kernel with standard deviation `beam_kernel_sd` (in pixels).
        Otherwise, a pencil beam is assumed.
    interp_method : str
        The interpolation method used by `scipy.interpolate.RegularGridInterpolator`
        to interpolate the resulting images to the same dimensions as the
        images in the `measurements`.
    """
    # Class attributes
    SIMULATED_QUANTITIES = ['sync', 'fd']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'dummy',
                            'cosmic_ray_electron_density',
                            'thermal_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']

    def __init__(self, measurements, gamma=1.,
                 beam_kernel_sd=None, interp_method='nearest'):
        super().__init__(measurements)
        self.gamma = gamma
        self.Stokes = {}
        self.interp_method = interp_method
        self.beam_kernel_sd = beam_kernel_sd

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

        A = sqrt(3) * e**3 / (8*pi*m_e*c**2)
        B = (4*pi*m_e*c**2 / (3*e))**((1-gamma)/2)
        return A*B

    def _find_ranges(self, coords):
        """
        Finds the coordinate ranges necessary for constructing the simulated image
        """
        # Extracts ranges from coordinates
        lon_range = np.abs(coords['lon_max']-coords['lon_min'])
        lat_range = np.abs(coords['lat_max']-coords['lat_min'])

        x_range = lon_range.to_value(u.rad) * self.distance
        y_range = lat_range.to_value(u.rad) * self.distance

        # Uses a diagonal through the box to construct the z range
        # (this aims at using all the available model data)
        z_range = 0
        for i in range(3):
            z_range += (self.grid.box[i][1] - self.grid.box[i][0])**2
        z_range = np.sqrt(z_range)

        return x_range, y_range, z_range

    def _construct_rotation_aux_grid(self, x_range, y_range, z_range, nx, ny, nz):
        """
        Constructs auxiliary quantities associated with rotation

        These comprise: an array with rotated coordinates, whose first two
        array axes correspond to the horizontal and vertical directions
        in the plane of the sky, and the third axis corresponds to the
        depth in the line of sight (LoS), an array containing the LoS depth
        at each point, three unit vectors with the projection directions
        (parallel and perpendicular to the LoS).

        The rotation (Euler) angles are read *directly* from the dummy Field
        parameters.

        Parameters
        ----------
        x_range, y_range : astropy.units.Quantity
            *Physical* distances corresponding to the horizontal and
            vertical sizes of the output in the plane of the sky.
        z_range : astropy.units.Quantity
            *Physical* distance corresponding to the depth along the LoS.
        nx, ny, nz : int
            Number of points used in the grid.

        Returns
        -------
        aux_grid : astropy.units.Quantity
            Array with rotated coordinates, whose first two axes correspond to the
            horizontal and vertical directions in the plane of the sky, and
            the third axis corresponds to the depth in the LoS.
        depths_grid :
            the LoS depth at each point.
        u_LoS_par : numpy.ndarray
            Unit vector parallel to the LoS
        u_LoS_perp_x, u_LoS_perp_y : numpy.ndarray
            Unit vectors perpendicular to the LoS and to each other
        """
        x_aux = np.linspace(-x_range/2, x_range/2, nx)
        y_aux = np.linspace(-y_range/2, y_range/2, ny)
        z_aux = np.linspace(-z_range/2, z_range/2, nz)

        aux_grid = np.empty((nx, ny, nz, 3)) << x_range.unit

        aux_grid[:,:, :, 0], aux_grid[:,:, :, 1], aux_grid[:,:, :, 2] = np.meshgrid(x_aux, y_aux,
                                                                                    z_aux, indexing='ij')
        # Saves the LoS depth for the integration later
        depths_grid = aux_grid[:,:, :, 2]

        # Rotates the coordinates grid
        params = self.fields['dummy']
        rot = scipy_rotation.from_euler('zyx', (params['gamma'].to_value(u.rad),
                                                params['beta'].to_value(u.rad),
                                                params['alpha'].to_value(u.rad)))
        # Matrix product of each value in the grid
        aux_grid = np.einsum('ji,klmi->klmj', rot.as_matrix(), aux_grid)

        # Constructs projection versors
        u_LoS_par = rot.apply(np.array([0,0,1]))
        u_LoS_perp_x = rot.apply(np.array([1,0,0]))
        u_LoS_perp_y = rot.apply(np.array([0,1,0]))

        return aux_grid, depths_grid, u_LoS_par, u_LoS_perp_x, u_LoS_perp_y

    def _interpolate_data(self, aux_grid, data, fill_value=0.0):
        """
        Interpolates data on a new grid

        Paramters
        ---------
        aux_grid : astropy.units.Quantity
            Auxiliary (L,M,N,3)-array with the coordinate grid to which one wants
            to interpolate the data.
        data : astropy.units.Quantity
            Array containing the dataset to be interpolated. It must be defined
            on the grid available at `self.grid`.

        Returns
        -------
        interpolated_data
            The interpolated `data` onto the `aux_grid` coordinates

        """
        unit = aux_grid.unit
        interpolator = RegularGridInterpolator(points=(self.grid.x[:,0,0].to_value(unit),
                                                       self.grid.y[0,:,0].to_value(unit),
                                                       self.grid.z[0,0,:].to_value(unit)),
                                               values=data.value,
                                               bounds_error=False,
                                               fill_value=fill_value)
        interpolated_data = interpolator(aux_grid.to_value(unit)) << data.unit

        return interpolated_data

    def _get_projected_interpolated_B(self, aux_grid, B, v_par, v_perp_x, v_perp_y):
        B_interp = np.empty(aux_grid.shape) << B.unit
        for i in range(3):
            B_interp[:,:,:,i] = self._interpolate_data(aux_grid, B[:,:,:,i])

        # Projects onto LoS
        Bpara = np.einsum('ijkl,l->ijk', B_interp, v_par, optimize=True)
        # Project on plane of the sky
        Bperp_x = np.einsum('ijkl,l->ijk', B_interp, v_perp_x, optimize=True)
        Bperp_y = np.einsum('ijkl,l->ijk', B_interp, v_perp_y, optimize=True)

        return Bperp_x, Bperp_y, Bpara


    def simulate(self, key, coords_dict, realization_id, output_units):
        obs_name, freq, _, flag = key

        if obs_name == 'fd':
            flag = 'fd'
            # This makes the 'fd' (which usually do not have a flag)
            # behave similarly to 'sync'


        if flag not in self.Stokes:
            # Stores the distance for later
            self.distance = self.fields['dummy']['distance']

            x_range, y_range, z_range = self._find_ranges(coords_dict)

            aux_grid, depth, v_par, v_perp_x, v_perp_y = self._construct_rotation_aux_grid(
                x_range, y_range, z_range, *self.grid.resolution)

            # Prepares interpolated quantities
            ne = self._interpolate_data(aux_grid, self.fields['thermal_electron_density'])
            ncr = self._interpolate_data(aux_grid,
                                         self.fields['cosmic_ray_electron_density'][:,:,:,0])

            Bperp_x, Bperp_y, B_para = self._get_projected_interpolated_B(aux_grid,
                                                                          self.fields['magnetic_field'],
                                                                          v_par, v_perp_x, v_perp_y)

            # Computes Faraday depth, if this was requested
            if ('fd' in [k[0] for k in self.observables]):

                self.Stokes['fd'] = obs.compute_fd(depth, B_para, ne,
                                                   beam_kernel_sd=self.beam_kernel_sd)

            # Computes synchrotron (I,Q,U), if this was requested
            if ('sync' in [k[0] for k in self.observables]):
                wavelength = (freq*u.GHz).to(u.cm, equivalencies=u.spectral())

                I, U, Q = obs.compute_stokes_parameters(depth, wavelength,
                                                        Bperp_x, Bperp_y, B_para,
                                                        ne, ncr, gamma=self.gamma,
                                                        beam_kernel_sd=self.beam_kernel_sd)
                self.Stokes['I'] = I
                self.Stokes['Q'] = Q
                self.Stokes['U'] = U

        out_data = self.Stokes.pop(flag)
        out_data_units = out_data.unit

        # Now, we need to interpolate to the original image resolution
        # First, store the available coordinates
        # (NB there was another round of interpolation before, which
        # placed the data precisely within the coordinates of the image,
        # but perhaps in the wrong resolution)
        x_range, y_range, z_range = self._find_ranges(coords_dict)
        x = np.linspace(-x_range/2, x_range/2, self.grid.resolution[0])
        y = np.linspace(-y_range/2, y_range/2, self.grid.resolution[1])

        # Setup the interpolator
        interpolator = RegularGridInterpolator(points=(x.to_value(u.kpc), y.to_value(u.kpc)),
                                               values=out_data.value,
                                               bounds_error=False,
                                               fill_value=0,
                                               method=self.interp_method)

        # Now we convert the original Galactic coordinates into x and y
        # under the assumption that the centres of the images coincide
        coords = self.output_coords[key]
        nx, ny = coords['shape']

        lon_range = np.abs(coords['lon_max'] - coords['lon_min'])
        lat_range = np.abs(coords['lat_max'] - coords['lat_min'])

        x_range = lon_range.to_value(u.rad) * self.distance
        y_range = lat_range.to_value(u.rad) * self.distance

        x_target = np.linspace(-x_range/2, x_range/2, nx)
        y_target = np.linspace(-y_range/2, y_range/2, ny)
        x_target, y_target = np.meshgrid(x_target, y_target)

        interp_points = np.array([x_target, y_target]).T

        result = interpolator(interp_points) * out_data_units

        # Adjusts the units
        if obs_name == 'sync':
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
