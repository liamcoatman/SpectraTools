# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:11:10 2015

@author: lc585

Calculates statistics (mean, standard deviation (i.e. RMS), mean
error, etc) of the flux between two wavelength points.

Returns::

mean flux, RMS of flux, mean error, SNR:
    SNR = (mean flux / RMS)

"""

import numpy as np
from spectra.get_spectra import get_liris_spec, get_boss_dr12_spec, get_sdss_dr7_spec
import matplotlib.pyplot as plt

def stats(wave, flux, err, wave_min, wave_max, show=False):


    i = np.argmin( np.abs( wave - wave_min))
    j = np.argmin( np.abs( wave - wave_max))

    flux = flux * 1e18
    err = err * 1e18

    fl = flux[i:j]
    er = err[i:j]
    w = wave[i:j]

    good = (er > 0) & ~np.isnan(fl)
    if len(good.nonzero()[0]) == 0:
        print('No good data in this range!')
        return np.nan, np.nan, np.nan, np.nan

    fl = fl[good]
    er = er[good]
    w = w[good]

    fig, ax = plt.subplots()
    ax.errorbar(w, fl, yerr=er)
    ax2 = ax.twinx()
    ax2.plot(w, er, color='red')

    mfl = fl.mean()
    std = fl.std()
    mer = er.mean()

    snr = fl / er
    snr = np.median(snr)

    ax.set_title('snr = {0:.2f}'.format(snr))

    if show:
        print 'mean %g, std %g, er %g, snr %g' % (mfl, std, mer, snr)


    return mfl, std, mer, snr

if __name__ == '__main__':

    import sys
    import os
    sys.path.append("/home/lc585/Dropbox/IoA/BlackHoleMasses")
    from wht_properties import get_wht_quasars
    quasars = get_wht_quasars().all_quasars()
    q = quasars[10]

    print q.sdss_name

    fname = os.path.join('/data/lc585/WHT_20150331/html/',q.name,'tcdimcomb.ms.fits')

    wav, dw, flux, err = get_liris_spec(fname)

    mfl, std, mer, snr = stats(wav/(1.0+q.z_ICA), flux, err, 6400, 6800, show=True)

#    plt.show()

#    for name, z in zip(names, zs):
#
#        fname = os.path.join('/data/lc585/WHT_20150331/html/',name,'dimcombLR+bkgd_v138.ms.fits')
#        wavelength, dw, flux, err = get_liris_spec(fname)
#
#        mfl, std, mer, snr = stats(wavelength/(1.0+z), flux, err, 6400.0, 6800.0, show=True)


#    for sn, bn, z in zip(sdss_name, boss_name, zs):
#
#        try:
#            if not bn:
#                wav, dw, flux, err = get_sdss_dr7_spec(sn)
#
#            else:
#                wav, dw, flux, err = get_boss_dr12_spec(bn)
#
#            mfl, std, mer, snr = stats(wav/(1.0+z), flux, err, 1500.0, 1600.0, show=True)


        # Ha window  6400.0, 6800.0
        # Hb 4700, 5100
        # CIV 1500, 1600
