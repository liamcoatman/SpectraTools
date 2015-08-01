# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 12:52:25 2015

@author: lc585

Make wavelength scale using fits header
Returns w, dw
Bin widths dw (unlike np.diff) returns array of same length
Input is fits header, which hopefully contains the right keywords!

"""

from __future__ import division
import numpy as np

def make_wa_scale(wstart, dw, npts, constantdv=False, verbose=False):
    """
    Generates a wavelength scale from the wstart, dw, and npts
    values.

    http://nhmc.github.io/Barak/_modules/barak/spec.html
    """

    if constantdv:
        if verbose:  print 'make_wa_scale(): Using log-linear scale'
        wa = 10**(wstart + np.arange(npts, dtype=float) * dw)
        dw = wa * (10**dw - 1.0 )
    else:
        if verbose:  print 'make_wa_scale(): Using linear scale'
        wa = wstart + np.arange(npts, dtype=float) * dw

    return wa, dw

def get_wavelength(hd):

    """
    Given a fits header, get the wavelength solution.

    http://nhmc.github.io/Barak/_modules/barak/spec.html

    """
    dv = None

    dw = hd['CD1_1']      # dw
    CRVAL = hd['CRVAL1']  # wavelength at first pixel
    CRPIX = hd['CRPIX1']  # first pixel

    # wavelength of pixel 1
    wstart = CRVAL + (1 - CRPIX) * dw

    # check if it's log-linear scale (heuristic)
    if CRVAL < 10:
        # If wavelength of first pixel is less than 10 we are assuming its a log wavelength.
        from scipy.constants import c
        c_kms = c / 1.0e3
        dv = c_kms * (1. - 10. ** -dw)
        # print 'constant dv = %.3f km/s (assume CRVAL1 in log(Angstroms))' % dv

    npts = hd['NAXIS1']

    return make_wa_scale(wstart, dw, npts, constantdv=dv)

