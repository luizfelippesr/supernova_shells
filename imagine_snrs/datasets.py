import numpy as np
from astropy.io import fits

import astropy.units as u
import matplotlib.pyplot as plt

import imagine as img
import imagine_datasets as img_data


__all__ = ['SNR_DA530_I_1420MHz', 'SNR_DA530_Q_1420MHz', 'SNR_DA530_U_1420MHz', 
           'SNR_DA530_I_4850MHz', 'SNR_DA530_Q_4850MHz', 'SNR_DA530_U_4850MHz',
           'SNR_DA530_I_10450MHz', 'SNR_DA530_Q_10450MHz', 'SNR_DA530_U_10450MHz',
           'SNR_DA530_FD']

class _SNR_DA530_base(img.observables.ImageDataset):
    def __init__(self, crop_lon=None, crop_lat=None):

        filename = '../data/{}_DA530{}.fits'.format(self._OTYPE, self._FREQ)
        hdu = fits.open(filename)[0]
        data = hdu.data[0,0].T
        header = hdu.header
        frequency = header['OBSFREQ']*u.Hz

        val_min = {}
        val_max = {}
        val_arr = {}
        delta = {}

        for i in (1, 2):
            i = str(i)

            q = header['CTYPE' + i]
            n_pix = header['NAXIS' + i]
            ref_pos = header['CRPIX' + i]
            ref_val = header['CRVAL' + i]
            delta[q] = header['CDELT' + i]

            val_min[q] = ref_val - delta[q]*(ref_pos-1)
            val_max[q] = ref_val + delta[q]*(n_pix - ref_pos)
            val_arr[q] = np.arange(val_min[q], val_max[q] + delta[q]/2, delta[q]) - ref_val

        if self._OTYPE == 'RM':
            otype = 'fd'
            unit = u.rad/u.m/u.m
            error = 20
            tag = None
            frequency=None
        else:
            otype = 'sync'
            unit = u.K
            error = 1.7e-3
            tag = self._OTYPE

        lon_min=val_min['GLON-CAR']
        lon_max=val_max['GLON-CAR']
        lat_min=val_min['GLAT-CAR']
        lat_max=val_max['GLAT-CAR']
        
        if crop_lon is not None:
            lon_range = np.linspace(lon_min, lon_max, data.shape[0])
            lon_min, lon_max = lon_range[crop_lon], lon_range[-crop_lon]
            data = data[:, crop_lon:-crop_lon]
            
        if crop_lat is not None:
            lat_range = np.linspace(lat_min, lat_max, data.shape[1])
            lat_min, lat_max = lat_range[crop_lat], lat_range[-crop_lat]
            data = data[crop_lat:-crop_lat, :]
        
        super().__init__(data << unit, otype,
                         lon_min=lon_min, lon_max=lon_max,
                         lat_min=lat_min, lat_max=lat_max,
                         object_id='SNR G093.3+06.9',
                         error=error << unit,
                         frequency=frequency, tag=tag)


    
class SNR_DA530_FD(_SNR_DA530_base):
    _OTYPE = 'RM'
    _FREQ = ''


class SNR_DA530_I_1420MHz(_SNR_DA530_base):
    _OTYPE = 'I'
    _FREQ = '_1420'

    
class SNR_DA530_U_1420MHz(_SNR_DA530_base):
    _OTYPE = 'U'
    _FREQ = '_1420'


class SNR_DA530_Q_1420MHz(_SNR_DA530_base):
    _OTYPE = 'Q'
    _FREQ = '_1420'


class SNR_DA530_I_4850MHz(_SNR_DA530_base):
    _OTYPE = 'I'
    _FREQ = '_4850'

    
class SNR_DA530_U_4850MHz(_SNR_DA530_base):
    _OTYPE = 'U'
    _FREQ = '_4850'


class SNR_DA530_Q_4850MHz(_SNR_DA530_base):
    _OTYPE = 'Q'
    _FREQ = '_4850'

    
class SNR_DA530_I_10450MHz(_SNR_DA530_base):
    _OTYPE = 'I'
    _FREQ = '_10450'

    
class SNR_DA530_U_10450MHz(_SNR_DA530_base):
    _OTYPE = 'U'
    _FREQ = '_10450'


class SNR_DA530_Q_10450MHz(_SNR_DA530_base):
    _OTYPE = 'Q'
    _FREQ = '_10450'
