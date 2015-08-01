# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:03:55 2015

@author: lc585

Bins up the spectrum by averaging the values of every n
pixels. Not very accurate, but much faster than rebin().
Fine if I'm binning by, say a factor of two or three. Only
need rebin if I'm dividing up into some arbitrary bin width.

"""
from __future__ import division

import numpy as np

def rebin_spectra(wa, fl, er=None, n=1, weighted=False):

    remain = -(len(wa) % n) or None

    wa = wa[:remain].reshape(-1, n)
    fl = fl[:remain].reshape(-1, n)
    if er is not None:
        er = er[:remain].reshape(-1, n)

    weights = 1./er**2

    n = float(n)

    if not weighted:
        wa = np.nansum(wa, axis=1) / n
        if er is not None:
            er = np.nansum(er, axis=1) / n / np.sqrt(n)
        fl = np.nansum(fl, axis=1) / n

    else:
        wa = np.average(wa,axis=1)
        if er is not None:
            er = np.average(er,axis=1) / np.sqrt(n)
        fl = np.average(fl, axis=1, weights=weights)

    if er is not None:
        return wa, fl, er

    else:
        return wa, fl