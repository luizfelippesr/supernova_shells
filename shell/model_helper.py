import os, sys
sys.path.append('../')
import numpy as np
import astropy.units as u
import shell.fields as fields

import shell.observable as obs
from shell import ShellModel, FieldTransformer
from collections import defaultdict
import astropy.constants as c
muG = u.microgauss

from importlib import reload
import .helicity as hel

class Model:
    """
    Helper class to make it easier to construct the model library

    Parameters
    ----------
    grid
        IMAGINE grid object
    shell_parameters : dict
        Dictionary containing parameters of the SN shell which
        one wants to modify
    shell_a, shell_b : float
        Shell a and b parameters (overrides `shell_parameters`)
    cr_type : str
        Type of CR model. May be 'thermal' i.e. CRE proportional to
        the TE, or 'uniform', i.e. CRe constant.
    ncr_ne : float
        The ratio of CRE to TE.
    B_type : str
        Type of B model. May be 'uniform' or 'helical'
    B : list
        A list containing the magnitude of the tree modes from which
        the B field will be constructed
    Brnd_B : float
        Ratio of the initial random field intensity and initial
        LS field intensity.
    period
        If a 'helical' B model is chosen, this is the period
        used.
    freqs : list
        List of frequencies used in the observables calculations
    gamma : int
        Gamma value used in the synchrotron calculation.
    """
    def __init__(self, grid, shell_parameters={}, ne=1*u.cm**-3,
                 Brnd_B=1, B=[0*muG, 0*muG, 0*muG], ncr_ne=1,
                 shell_a=None, shell_b=None, #for convenience
                 cr_type='thermal', B_type='uniform', period=70*u.pc,
                 freqs=[1.4*u.GHz, 1.42*u.GHz],
                 # 2.7*u.GHz], #, 4.8*u.GHz, 10*u.GHz],
                 gamma=1):

            # Creates the shell model
            shell_dict = dict(V0=0.0153*u.pc/u.yr,
                              a=1.3, b=10, R=50*u.pc,
                              elapsed_time=1300*u.yr)
            shell_dict.update(shell_parameters)
            if shell_a is not None:
                shell_dict['a'] = shell_a
            if shell_b is not None:
                shell_dict['b'] = shell_b
            shell_model = ShellModel(**shell_dict)

            # Generates the field transformer
            field_transformer = FieldTransformer(grid, shell_model)

            # Initial density model
            n0 = ne*np.ones(grid.shape)

            # Initial large scale field model
            if B_type == 'uniform':
                Bls = [Bi*np.ones(grid.shape) for Bi in B]
            elif B_type == 'helical':
                Bls = fields.helical(grid, B, period)

            # Initial random field model
            Brms = np.sqrt(np.mean(Bls[0]**2+Bls[1]**2+Bls[2]**2)) * Brnd_B
            Brnd = fields.simple_random(grid, Brms)

            # Total initial field
            B0 = [Bls_i + Brnd_i for Bls_i, Brnd_i in zip(Bls, Brnd)]

            # Computes final B and thermal electron density
            self.ne, self.B = field_transformer(n0, B0)
            self.ne0, self.B0 = n0, B0

            # Convenience unpacking
            self.Bx, self.By, self.Bz = self.B
            # Cosmic ray model
            if cr_type == 'uniform':
                # Uniformly distributed cosmic rays
                self.ncr = ne*np.ones(grid.shape)*ncr_ne
            elif cr_type == 'thermal':
                # Cosmic ray density proportional to thermal electron density
                self.ncr = self.ne*ncr_ne
            else:
                raise ValueError('Available options: "thermal" or "uniform"')

            # Stores frequencies (converted to wavelenghts), grid, etc
            self.wavelengths = [(c.c/(f)).to(u.cm) for f in freqs]
            self.grid = grid
            self.gamma = gamma


            self.patch_RM = True
            self.RM_threshold = 900*u.rad/u.m/u.m

            self._stokes = None
            self._Psi = None
            self._RM = None
            self._PI = None
            self._Hz = None
            self._Jz = None
            self._Hz_real = None
            self._Jz_real = None

    @property
    def Q(self):
        if self._stokes is None:
            self._compute_stokes()
        return self._stokes['Q']
    @property
    def U(self):
        if self._stokes is None:
            self._compute_stokes()
        return self._stokes['U']
    @property
    def I(self):
        if self._stokes is None:
            self._compute_stokes()
        return self._stokes['I']

    @property
    def Psi(self):
        if self._Psi is None:
            self._Psi = []
            for U, Q in zip(self.U, self.Q):
                # Computes Psi for all requested wavelengths
                self._Psi.append(obs.compute_Psi(U,Q))
        return self._Psi

    @property
    def PI(self):
        if self._PI is None:
            self._PI = []
            for U, Q in zip(self.U, self.Q):
                # Computes Psi for all requested wavelengths
                self._PI.append(np.sqrt(U**2+Q**2))
        return self._PI

    @property
    def RM(self):
        if self._RM is None:
            self._RM = []
            for Psi1, lambda1, Psi2, lambda2 in zip(self.Psi[:-1],
                                                    self.wavelengths[:-1],
                                                    self.Psi[1:],
                                                    self.wavelengths[1:]):
                RM = obs.compute_RM(Psi1, Psi2, lambda1, lambda2)

                if self.patch_RM:
                    # Patches bogus RM values
                    RM = np.where(np.abs(RM) < self.RM_threshold, RM, np.nan)

                self._RM.append(RM.to(u.rad/u.m/u.m))
        return self._RM

    def _compute_stokes(self):
        self._stokes = defaultdict(list)
        for wavelength in self.wavelengths:
            I, Q, U = obs.compute_stokes_parameters(self.grid, wavelength,
                                                    self.Bx, self.By, self.Bz,
                                                    self.ne, self.ncr,
                                                    self.gamma)
            self._stokes['I'].append(I)
            self._stokes['Q'].append(Q)
            self._stokes['U'].append(U)

    def estimate_observed_helicity(self):
        # Radius of the remnant/box
        Lz = (self.grid.box[2][1]-self.grid.box[2][0])/2

        ncr_mean = np.mean(self.ncr, axis=2)
        ne_mean = np.mean(self.ne, axis=2)

        self._Hz, self._Jz = hel.average_helicity(RM=self.RM[0], I=self.I, PA=self.Psi,
                                                  wavelengths=self.wavelengths,
                                                  ne=1*u.cm**-3, ncr=1*u.cm**-3, L=Lz,
                                                  x=self.grid.x[:,0,0], y=self.grid.y[0,:,0],
                                                  boundary_radius=1, output_current=True)

    @property
    def Hz(self):
        if self._Hz is None:
            self.estimate_observed_helicity()
        return self._Hz

    @property
    def Jz(self):
        if self._Jz is None:
            self.estimate_observed_helicity()
        return self._Jz

    def compute_helicity(self):
        self._Hz_real, self._Jz_real = hel.compute_theoretical_Hz(self.grid, self.B)

    @property
    def Hz_real(self):
        if self._Hz_real is None:
            self.compute_helicity()
        return self._Hz_real

    @property
    def Jz_real(self):
        if self._Jz_real is None:
            self.compute_helicity()
        return self._Jz_real
