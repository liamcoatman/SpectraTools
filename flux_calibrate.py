# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:05:10 2015

@author: lc585

Inputs wavelength, flux, and optionally error of spectrum, wavelength and
transmission for filter, and AB magnitude in that filter.

Returns flux-normalised wavelength, flux and (optionally) error

**Warning - this hasn't been tested**

"""

from __future__ import division

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u

def flux_calibrate(wavlen=None,
                  flux=None,
                  flux_sigma=None,
                  ftrwav=None,
                  ftrtrans=None,
                  mag=None):


    # Calculate AB zero point
    sum1 = np.sum( ftrtrans[:-1] * (0.10893/(ftrwav[:-1]**2)) * ftrwav[:-1] * np.diff(ftrwav))
    sum2 = np.sum( ftrtrans[:-1] * ftrwav[:-1] * np.diff(ftrwav) )
    zromag = -2.5 * np.log10(sum1 / sum2)

    # Now calculate magnitudes
    spc = interp1d(wavlen,
                   flux,
                   bounds_error=False,
                   fill_value=0.0)

    sum1 = np.sum( ftrtrans[:-1] * spc(ftrwav[:-1]) * ftrwav[:-1] * np.diff(ftrwav))
    sum2 = np.sum( ftrtrans[:-1] * ftrwav[:-1] * np.diff(ftrwav) )
    ftrmag = (-2.5 * np.log10(sum1 / sum2)) - zromag

    delta = mag - ftrmag

    fnew = flux * 10.0**( -delta / 2.5 )
    if flux_sigma is not None:
        enew = flux_sigma * 10.0**( -delta / 2.5 )
        return fnew*(u.erg / u.cm / u.cm / u.s / u.AA), enew*(u.erg / u.cm / u.cm / u.s / u.AA)
    else:
        return fnew*(u.erg / u.cm / u.cm / u.s / u.AA)

