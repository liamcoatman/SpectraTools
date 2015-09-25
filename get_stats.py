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

    fl = flux[i:j]
    er = err[i:j]

    good = (er > 0) & ~np.isnan(fl)
    if len(good.nonzero()[0]) == 0:
        print('No good data in this range!')
        return np.nan, np.nan, np.nan, np.nan

    fl = fl[good]
    er = er[good]

#    fig, ax = plt.subplots()
#    ax.errorbar(wave[good], fl, yerr=er)
#    plt.show()

    mfl = fl.mean()
    std = fl.std()
    mer = er.mean()

    snr = fl / er
    snr = np.median(snr)

    if show:
        print 'mean %g, std %g, er %g, snr %g' % (mfl, std, mer, snr)


    return mfl, std, mer, snr

if __name__ == '__main__':

    import sys
    sys.path.append("/home/lc585/Dropbox/IoA/BlackHoleMasses")
    from wht_properties import get_wht_quasars
    quasars = get_wht_quasars()
    zs =  np.array([q.z_HW10 for q in quasars.all_quasars()])
    sdss_name = np.array([i.sdss_name for i in quasars.all_quasars()])
    boss_name = np.array([i.boss_name for i in quasars.all_quasars()])

    for i in sdss_name:
        print i
    i = np.where(sdss_name=='SDSSJ132948.73+324124.4')[0][0]
    zs = np.delete(zs,i)
    sdss_name = np.delete(sdss_name,i)
    boss_name = np.delete(boss_name,i)

    names = ['SDSSJ0738+2710',
            'SDSSJ0743+2457',
            'SDSSJ0806+2455',
            'SDSSJ0829+2423',
            'SDSSJ0854+0317',
            'SDSSJ0858+0152',
            'SDSSJ1104+0957',
            'SDSSJ1236+1129',
            'SDSSJ1246+0426',
            'SDSSJ1306+1510',
            'SDSSJ1317+0806',
            'SDSSJ1329+3241',
            'SDSSJ1336+1443',
            'SDSSJ1339+1515',
            'SDSSJ1400+1205',
            'BQJ1525+2928',
            'SDSSJ1530+0623',
            'BQJ1538+0233',
            'SDSSJ1618+2341',
            'BQJ1627+3135',
            'SDSSJ1634+3014']


    import os
    wav, dw, flux, err = get_boss_dr12_spec('SDSSJ132948.73+324124.4')
    mfl, std, mer, snr = stats(wav/(1.0+z), flux, err, 1500.0, 1600.0, show=True)



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
