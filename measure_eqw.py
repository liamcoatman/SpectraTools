# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:53:16 2015

@author: lc585

Blue-shift to quasar rest frame
Fit power-law to continuum and normalise flux.
Measure the equivalent width of emission line.
Give wav and regions in Angstroms

"""

import astropy.units as u
import numpy as np
from lmfit.models import PowerLawModel
from lmfit import minimize

def resid(p,x,model,data=None):

        mod = model.eval(params=p, x=x)

        if data is not None:
            return mod - data
        else:
            return mod

def measure_eqw(wav,
                flux,
                z,
                continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
                eqw_region=[6400,6800]*u.AA):

    # Test if wav has units. If it does convert to Angstroms. If it doesn't
    # then give it units. This might be a bit hacky.

    try:
        wav.unit
        wav = wav.to(u.AA)
    except AttributeError:
        wav = wav*u.AA

    wav = wav / (1.0 + z)

    # eqw region
    eqw_inds = (wav > eqw_region[0]) & (wav < eqw_region[1])

    # index is true in the region where we fit the continuum
    blue_inds = (wav > continuum_region[0][0]) & (wav < continuum_region[0][1])
    red_inds = (wav > continuum_region[1][0]) & (wav < continuum_region[1][1])

    xdat = np.array( [continuum_region[0].mean().value, continuum_region[1].mean().value] )
    ydat = np.array( [np.median(flux[blue_inds]), np.median(flux[red_inds])] )

    # fit power-law to continuum region
    mod = PowerLawModel()
    pars = mod.make_params()
    pars['exponent'].value = 1.0
    pars['amplitude'].value = 1.0

    out = minimize(resid,
                   pars,
                   args=(xdat, mod, ydat),
                   method='leastsq')

    # Normalised flux
    f = flux / mod.eval(params=pars, x=wav.value)

    eqw = (f[eqw_inds][:-1] - 1.0) * np.diff(wav[eqw_inds])

#    fig, ax = plt.subplots()
#
#    ax.scatter(wav[blue_inds] , flux[blue_inds], c='red')
#    ax.scatter(wav[red_inds] , flux[red_inds], c='red')
#    ax.scatter(wav[eqw_inds], flux[eqw_inds])
#    ax.scatter(xdat, ydat, c='yellow', s=70)
#    xs = np.arange( xdat.min(), xdat.max(), 1)
#    ax.plot( xs, mod.eval(params=pars, x=xs) , c='red')
#    plt.show()
#    plt.close()

    return eqw.sum()



### Testing

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from get_spectra import get_liris_spec
    import os

    names = ['SDSSJ1339+1515',
             'SDSSJ0829+2423',
             'SDSSJ1634+3014',
             'SDSSJ1317+0806',
             'SDSSJ1104+0957',
             'SDSSJ0743+2457',
             'SDSSJ1618+2341',
             'BQJ1525+2928',
             'SDSSJ0806+2455',
             'SDSSJ1246+0426',
             'SDSSJ1329+3241',
             'SDSSJ0854+0317',
             'SDSSJ0738+2710',
             'BQJ1627+3135',
             'SDSSJ1236+1129',
             'SDSSJ1336+1443',
             'SDSSJ1400+1205',
             'SDSSJ0858+0152',
             'SDSSJ1530+0623',
             'SDSSJ1306+1510',
             'BQJ1538+0233']

    zs = [2.318977,
          2.412485,
          2.499230,
          2.375307,
          2.421565,
          2.165947,
          2.280475,
          2.361230,
          2.155533,
          2.441454,
          2.169098,
          2.247021,
          2.445636,
          2.324441,
          2.155473,
          2.146961,
          2.172160,
          2.169982,
          2.218803,
          2.401044,
          2.242759]

    for name, z in zip(names,zs):

        fname = os.path.join('/data/lc585/WHT_20150331/html/',name,'dimcombLR+bkgd_v138.ms.fits')
        wavelength, dw, flux, err = get_liris_spec(fname)
        eqw = measure_eqw(wavelength*u.AA,
                      flux,
                      z)
        print name, eqw