# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:47:44 2015

@author: lc585

Return monochromatic luminosity from SED

Inputs:

wavlen - wavlelength where luminosity should be calculated
mag - observed AB magnitude (float)
ftrwav, ftrtrans - wavelength and response of filter (numpy arrays)
z - redshift

Outputs:

Monochromatic luminosity

Right now uses standard SED model, but can easily modified to, for example,
change extinction/reddening.

"""
from __future__ import division

import numpy as np
import yaml
import sys
# Change this to use astropy.
import cosmolopy.distance as cd
from lmfit import minimize, Parameters
from scipy.interpolate import interp1d
from functools import partial
import astropy.units as u
import math
from astropy.cosmology import WMAP9 as cosmoWMAP
from functools import partial
from multiprocessing import Pool

# Temporary - eventually add qsofit to python path
sys.path.insert(0, '/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit')
from qsrmod import qsrmod
from load import load as qsrload

def mono_lum(mag=18.0,
             magsys='AB',
             mono_wav=5100.0,
             z=1.0,
             ftrwav=np.ones(100),
             ftrtrans=np.ones(100)):

    plslp1 = 0.46
    plslp2 = 0.03
    plbrk = 2822.0
    bbt = 1216.0
    bbflxnrm = 0.24
    elscal = 0.71
    scahal = 0.86
    galfra = 0.31
    ebv = 0.0
    imod = 18.0

    with open('/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit/input.yml', 'r') as f:
        parfile = yaml.load(f)

    fittingobj = qsrload(parfile)

    lin = fittingobj.get_lin()
    galspc = fittingobj.get_galspc()
    ext = fittingobj.get_ext()
    galcnt = fittingobj.get_galcnt()
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax()
    wavlen_rest = fittingobj.get_wavlen()
    ztran = fittingobj.get_ztran()
    lyatmp = fittingobj.get_lyatmp()
    lybtmp = fittingobj.get_lybtmp()
    lyctmp = fittingobj.get_lyctmp()
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}
    cosmo = cd.set_omega_k_0(cosmo)
    flxcorr = np.array( [1.0] * len(wavlen_rest) )

    params = Parameters()
    params.add('plslp1', value = plslp1)
    params.add('plslp2', value = plslp2)
    params.add('plbrk', value = plbrk)
    params.add('bbt', value = bbt)
    params.add('bbflxnrm', value = bbflxnrm)
    params.add('elscal', value = elscal)
    params.add('scahal', value = scahal)
    params.add('galfra', value = galfra)
    params.add('ebv', value = ebv)
    params.add('imod', value = imod)

    wavlen, flux = qsrmod(params,
                          parfile,
                          wavlen_rest,
                          z,
                          lin,
                          galspc,
                          ext,
                          galcnt,
                          ignmin,
                          ignmax,
                          ztran,
                          lyatmp,
                          lybtmp,
                          lyctmp,
                          whmin,
                          whmax,
                          cosmo,
                          flxcorr)

    if magsys == 'AB':
        
        # Calculate AB zero point
        sum1 = np.sum( ftrtrans[:-1] * (0.10893/(ftrwav[:-1]**2)) * ftrwav[:-1] * np.diff(ftrwav))
        sum2 = np.sum( ftrtrans[:-1] * ftrwav[:-1] * np.diff(ftrwav) )
        zromag = -2.5 * np.log10(sum1 / sum2)

    if magsys == 'VEGA':

        # Calculate vega zero point 
        fvega = '/data/vault/phewett/vista_work/vega_2007.lis' 
        vspec = np.loadtxt(fvega) 
        vf = interp1d(vspec[:,0], vspec[:,1])
        sum1 = np.sum(ftrtrans[:-1] * vf(ftrwav[:-1]) * ftrwav[:-1] * np.diff(ftrwav))
        sum2 = np.sum( ftrtrans[:-1] * ftrwav[:-1] * np.diff(ftrwav) ) 
        zromag = -2.5 * np.log10(sum1 / sum2) 

    def resid(p,
              mag,
              flux,
              wavlen,
              zromag,
              ftrwav,
              ftrtrans):

        newflux = p['norm'].value * flux
        spc = interp1d(wavlen, newflux, bounds_error=False, fill_value=0.0)

        sum1 = np.sum( ftrtrans[:-1] * spc(ftrwav[:-1]) * ftrwav[:-1] * np.diff(ftrwav))
        sum2 = np.sum( ftrtrans[:-1] * ftrwav[:-1] * np.diff(ftrwav) )
        ftrmag = (-2.5 * np.log10(sum1 / sum2)) - zromag

        return [mag - ftrmag]

    resid_p = partial(resid,
                      mag=mag,
                      flux=flux,
                      wavlen=wavlen,
                      zromag=zromag,
                      ftrwav=ftrwav,
                      ftrtrans=ftrtrans)

    p = Parameters()
    p.add('norm', value = 1e-17)

    result = minimize(resid_p, p, method='leastsq')

    indmin = np.argmin( np.abs( (wavlen / (1.0 + z)) - (mono_wav-5.0)))
    indmax = np.argmin( np.abs( (wavlen / (1.0 + z)) - (mono_wav+5.0)))

    # Flux density in erg/cm2/s/A
    f5100 =   p['norm'].value * np.median(flux[indmin:indmax])

    f5100 = f5100 * (u.erg / u.cm / u.cm / u.s / u.AA)

    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)

    # Monochromatic luminosity at 5100A
    l5100 = f5100 * (1 + z) * 4 * math.pi * lumdist**2

    l5100 = l5100 * 5100.0 * (u.AA)

    # print l5100

    return l5100

def mono_lum_err(mag=18.0,
                 emag=0.1,
                 mono_wav=5100.0,
                 z=1.0,
                 ftrwav=np.ones(100),
                 ftrtrans=np.ones(100),
                 n_samples=10):

    m = np.random.normal(mag, emag, n_samples)
    mono_lum_p = partial(mono_lum, mono_wav=mono_wav, z=z, ftrwav=ftrwav, ftrtrans=ftrtrans)
    pool = Pool(processes=16)
    lmon = pool.map(mono_lum_p, m)
    lmon = np.array([i.value for i in lmon])

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.hist(lmon, bins=20)
    # plt.show()

    return np.mean(lmon), np.std(lmon)

if __name__ == '__main__':

    sys.path.insert(0, '/home/lc585/Dropbox/IoA/BlackHoleMasses')
    from wht_properties import get_wht_quasars
    qs = get_wht_quasars().all_quasars()
    for q in qs:
        print q.sdss_name, q.calc_mono_lum_5100_err()
