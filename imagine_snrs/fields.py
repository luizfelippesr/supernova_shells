import os, sys
import numpy as np
import astropy.units as apu
sys.path.append('../')

import imagine as img
from shell import ShellModel, FieldTransformer, fields

class SNRThermalElectrons(img.fields.ThermalElectronDensityField):
    """Example: thermal electron density of an (double) exponential disc"""

    NAME = 'SNR_magnetic_field'
    PARAMETER_NAMES = ['initial_electron_density',
                       'shell_V0', 'shell_a', 'shell_b',
                       'elapsed_time', 'shell_radius']

    def compute_field(self, seed):

        # Computes the shell model
        shell_model = ShellModel(V0=self.parameters['shell_V0'],
                                 a=self.parameters['shell_a'],
                                 b=self.parameters['shell_b'],
                                 R=self.parameters['shell_radius'],
                                 elapsed_time=self.parameters['elapsed_time'])

        # Prepares field transformer
        self.field_transformer = FieldTransformer(self.grid, shell_model)

        # Prepares and transforms the initial field
        ne_0 = np.ones(self.data_shape) * self.parameters['initial_electron_density']
        ne_shell = self.field_transformer.transform_density_field(ne_0)

        return ne_shell


class SNRHelicalMagneticField(img.fields.MagneticField):
    """
    Magnetic field of a supernova remnant
    """
    NAME = 'SNR_helical_magnetic_field'
    PARAMETER_NAMES = ['Bx', 'By', 'Bz', 'period']
    DEPENDENCIES_LIST = [SNRThermalElectrons]

    def compute_field(self, seed):
        # Computes initial field
        B_input = [self.parameters[Bi] for Bi in ('Bx', 'By', 'Bz')]
        Blist = fields.helical_new(self.grid, B_input, self.parameters['period'])

        # Transforms the initial field
        ne_obj = self.dependencies[SNRThermalElectrons]
        Bx, By, Bz = ne_obj.field_transformer.transform_magnetic_field(*Blist)

        B = np.empty(self.data_shape) << Bx.unit
        for i, Bc in enumerate([Bx, By, Bz]):
            B[:,:,:,i] = Bc

        return B


class SNRUniformMagneticField(img.fields.MagneticField):
    """
    Magnetic field of a supernova remnant
    """
    NAME = 'SNR_unif_magnetic_field'
    PARAMETER_NAMES = ['Bx', 'By', 'Bz']
    DEPENDENCIES_LIST = [SNRThermalElectrons]

    def compute_field(self, seed):
        # Computes initial field

        Blist = [ self.parameters[Bi]*np.ones(self.grid.shape)
                  for Bi in ('Bx', 'By', 'Bz') ]

        # Transforms the initial field
        ne_obj = self.dependencies[SNRThermalElectrons]
        Bx, By, Bz = ne_obj.field_transformer.transform_magnetic_field(*Blist)

        B = np.empty(self.data_shape) << Bx.unit
        for i, Bc in enumerate([Bx, By, Bz]):
            B[:,:,:,i] = Bc

        return B


class CosmicRayElectronSingleEnergyDensityField(img.fields.Field):
    """
    Base class for the inclusion of models for spatial distribution of
    cosmic ray  electrons.

    Note
    ----
    This is provisional

    Parameters
    ----------
    grid : imagine.fields.grid.BaseGrid
        Instance of :py:class:`imagine.fields.grid.BaseGrid` containing a 3D
        grid where the field is evaluated
    parameters : dict
        Dictionary of full parameter set {name: value}
    ensemble_size : int
        Number of realisations in field ensemble
    ensemble_seeds
        Random seed(s) for generating random field realisations
    """
    # Class attributes
    TYPE = 'cosmic_ray_electron_density'
    UNITS = apu.cm**(-3)

    @property
    def parameter_names(self):
        """Parameters of the field"""
        return self.PARAMETER_NAMES + ['cr_energy']

    #def __init__(self, *args, cr_energy=1*apu.GeV, **kwargs)
        #super().__init__(*args, **kwargs)
        #self.Ecr = cr_energy

    @property
    def data_description(self):
        return(['grid_x', 'grid_y', 'grid_z', 'energy_bins'])

    @property
    def data_shape(self):
        return (*self.grid.shape, 1)


class ConstantCosmicRayElectrons(CosmicRayElectronSingleEnergyDensityField):
    """
    Constant cosmic ray electron density field

    The field parameters are:
    'ncre', the number density of thermal electrons
    """

    # Class attributes
    NAME = 'constant_CRe'
    PARAMETER_NAMES = ['ncre']


    def compute_field(self, seed):
        return np.ones(self.data_shape)*self.parameters['ncre']


class TrackingCosmicRayElectrons(CosmicRayElectronSingleEnergyDensityField):
    """
    'Tracking' cosmic ray electron density field

    This assumes that the cosmic ray energy density is proportional to the
    thermal electron density

    The field parameters are:
    'ncre_nte', the ratio between the number density of thermal electrons
    and the number density of cosmic ray electrons
    """

    # Class attributes
    NAME = 'tracking_CRe'
    PARAMETER_NAMES = ['ncre_nte']
    DEPENDENCIES_LIST = ['thermal_electron_density']

    def compute_field(self, seed):
        nte = self.dependencies['thermal_electron_density']

        ncr = np.empty(self.data_shape) << self.units
        ncr[:,:,:,0] = nte * self.parameters['ncre_nte']

        return ncr



class EquipartitionCosmicRayElectrons(CosmicRayElectronSingleEnergyDensityField):
    """
    Equipartition cosmic ray electron density field

    This assumes that the energy density in cosmic ray electrons is
    proportional to the magnetic energy density.

    The field parameters are:
    'Ecr_Em', the ratio between the energy density in cosmic ray electrons
    and the energy density of the magnetic field
    """

    # Class attributes
    NAME = 'constant_CR'
    PARAMETER_NAMES = ['Ecr_Em']
    DEPENDENCIES_LIST = ['magnetic_field']

    def compute_field(self, seed):
        # Computes the magnetic energy density
        B = self.dependencies['magnetic_field']
        Em = (B[:,:,:,0]**2 + B[:,:,:,1]**2 + B[:,:,:,2]**2)/(8*np.pi)
        # Adjusts Em's units (astropy does not do this alone..)
        Em = Em * apu.microgauss**(-2) * apu.erg * apu.cm**(-3)

        ncr = np.empty(self.data_shape) << self.units
        ncr[:,:,:,0] = Em / self.parameters['cr_energy'] * self.parameters['Ecr_Em']

        return ncr

