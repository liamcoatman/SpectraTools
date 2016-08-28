# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:49:50 2015

@author: lc585

Make composite spectra
Specific only for LIRIS

"""

from scipy.interpolate import interp1d
from astropy.table import Table
from get_spectra import get_liris_spec
from flux_calibrate import flux_calibrate
import numpy as np
import os
import matplotlib.pyplot as plt
import astropy.units as u
from fit_line import fit_line

def MakeComposite():

    fname = '/home/lc585/Dropbox/IoA/BlackHoleMasses/lineinfo_v4.dat'
    t = Table.read(fname,
                   format='ascii',
                   guess=False,
                   delimiter=',')


    spec, err2 = [], []
    for n, z, m in zip(t['Name'], t['z_ICA'], t['Median_Ha']):

        fname = os.path.join('/data/lc585/WHT_20150331/html',n,'dimcombLR+bkgd_v138.ms.fits')
        wavelength, dw, flux, err = get_liris_spec(fname)

        with open('/home/lc585/Dropbox/IoA/QSOSED/Model/Filter_Response/H.response','r') as f:
            ftrwav, ftrtrans = np.loadtxt(f,unpack=True)

        flux, err = flux_calibrate(wavlen=wavelength, flux=flux, flux_sigma=err, ftrwav=ftrwav, ftrtrans=ftrtrans, mag=18.0)

        # Transform to quasar rest-frame
        wavelength = wavelength / (1.0 + z)
        from astropy.constants import c
        wavelength = wavelength * (1.0 - m*(u.km/u.s) / c.to(u.km/u.s))

        f1 = interp1d( wavelength, flux, bounds_error=False, fill_value=np.nan )
        f2 = interp1d( wavelength, err**2, bounds_error=False, fill_value=np.nan )


        spec.append( f1( np.linspace(4000.0,7600.0,1168) ) )
        err2.append( f2( np.linspace(4000.0,7600.0,1168) ) )

    spec = np.array(spec)
    err2 = np.array(err2)

    meanspec, ns = np.zeros(1168), np.zeros(1168)

    for i in range(1168):
        for j in range(len(t)):
            if ~( np.isnan(spec[j,i]) | np.isnan(err2[j,i]) ):
                ns[i] += 1.0
                meanspec[i] += spec[j,i] / err2[j,i]

    return np.linspace(4000.0,7600.0,1168), meanspec / ns / 1e18



w, f = MakeComposite()
fig, ax = plt.subplots()
ax.plot(w, f)
# err = np.repeat(0.05,len(f))

# fit_line(w,
#          f,
#          err,
#          z=0.0,
#          w0=6564.89*u.AA,
#          continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
#          fitting_region=[6400,6800]*u.AA,
#          plot_region=[6000,7000]*u.AA,
#          nGaussians=0,
#          nLorentzians=2,
#          maskout=None,
#          verbose=True,
#          plot=True)

#Peak: -74.326125
#FWHM: 2338.39064098
#Median: -107.0
#EQW: 548.449953303 Angstrom

#fit_line(w,
#         f,
#         err,
#         z=0.0,
#         w0=4862.68*u.AA,
#         continuum_region=[[4550.,4700.]*u.AA,[5100.,5300.]*u.AA],
#         fitting_region=[4700.,4900.0]*u.AA,
#         plot_region=[4435,5535]*u.AA,
#         nGaussians=0,
#         nLorentzians=1,
#         maskout=None,
#         verbose=True,
#         plot=True)

#Peak: 206.9309375
#FWHM: 2355.49451771
#Median: 208.0
#EQW: 42.3820996054 Angstrom


plt.show()