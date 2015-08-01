# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:05:10 2015

@author: lc585

Takes input spectrum and transmission function.
Returns AB magnitude of spectrum
Can be used to flux-calibrate spectrum:

flux = flux * 10.0**( -(mag - mag_spectrum) / 2.5 )
err = err * 10.0**( -(mag - mag_spectrum) / 2.5 )

**Warning - this hasn't been tested**

"""

from __future__ import division

import numpy as np
from scipy.interpolate import interp1d

def get_magnitude(wavlen=None,
                  flux=None,
                  ftrwav=None,
                  ftrtrans=None):


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

    return ftrmag

