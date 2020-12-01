import os, sys
import numpy as np
sys.path.append('../')

import imagine as img
from shell import ShellModel, FieldTransformer, fields

class SupernovaShellThermalElectrons(img.fields.ThermalElectronDensityField):
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


class SupernovaShellHelicalMagneticField(img.fields.MagneticField):
    """
    Magnetic field of a supernova remnant
    """
    NAME = 'SNR_helical_magnetic_field'
    PARAMETER_NAMES = ['Bx', 'By', 'Bz', 'period']
    DEPENDENCIES_LIST = [SupernovaShellThermalElectrons]

    def compute_field(self, seed):
        # Computes initial field
        B_input = [self.parameters[Bi] for Bi in ('Bx', 'By', 'Bz')]
        Blist = fields.helical_new(self.grid, B_input, self.parameters['period'])

        # Transforms the initial field
        ne_obj = self.dependencies[SupernovaShellThermalElectrons]
        Bx, By, Bz = ne_obj.field_transformer.transform_magnetic_field(*Blist)

        B = np.empty(self.data_shape) << Bx.unit
        for i, Bc in enumerate([Bx, By, Bz]):
            B[:,:,:,i] = Bc

        return B


class SupernovaShellUniformMagneticField(img.fields.MagneticField):
    """
    Magnetic field of a supernova remnant
    """
    NAME = 'SNR_unif_magnetic_field'
    PARAMETER_NAMES = ['Bx', 'By', 'Bz']
    DEPENDENCIES_LIST = [SupernovaShellThermalElectrons]

    def compute_field(self, seed):
        # Computes initial field

        Blist = [ self.parameters[Bi]*np.ones(self.grid.shape)
                  for Bi in ('Bx', 'By', 'Bz') ]

        # Transforms the initial field
        ne_obj = self.dependencies[SupernovaShellThermalElectrons]
        Bx, By, Bz = ne_obj.field_transformer.transform_magnetic_field(*Blist)

        B = np.empty(self.data_shape) << Bx.unit
        for i, Bc in enumerate([Bx, By, Bz]):
            B[:,:,:,i] = Bc

        return B
