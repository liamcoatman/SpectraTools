# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:13:56 2015

@author: lc585

Fit emission line with model.

"""
from __future__ import division

import numpy as np
from rebin_spectra import rebin_spectra
import astropy.units as u
from lmfit.models import GaussianModel, LorentzianModel, PowerLawModel, ConstantModel
from lmfit import minimize, Parameters, fit_report
import numpy.ma as ma
import seaborn as sns

def resid(p,x,model,data=None,sigma=None):

        mod = model.eval( params=p, x=x )

        if data is not None:
            resids = mod - data
            if sigma is not None:
                weighted = np.sqrt(resids ** 2 / sigma ** 2)
                return weighted
            else:
                return resids
        else:
            return mod

def wave2doppler(w, w0):

    """
    function uses the Doppler equivalency between wavelength and velocity
    """
    w0_equiv = u.doppler_optical(w0)
    w_equiv = w.to(u.km/u.s, equivalencies=w0_equiv)

    return w_equiv

def plot_fit(wav=None,
             flux=None,
             err=None,
             pars=None,
             mod=None,
             out=None,
             plot_savefig=None,
             plot_title='',
             maskout=None,
             z=0.0,
             w0=6564.89*u.AA,
             velocity_shift=0.0*(u.km / u.s),
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=None):

    # plotting region
    plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])

    # Transform to doppler shift
    xdat = wave2doppler(wav, w0)

    # Add velocity shift
    xdat = xdat - velocity_shift

    xdat = xdat[plot_region_inds]
    ydat = flux[plot_region_inds]
    yerr = err[plot_region_inds]

    fig = plt.figure(figsize=(6,8))

    fig.suptitle(plot_title)
    fit = fig.add_subplot(3,1,1)
    fit.set_xticklabels( () )

    fit.errorbar(xdat.value, ydat, yerr=yerr, linestyle='', alpha=0.4)

    # Mark continuum fitting region
    blue_cont = wave2doppler(continuum_region[0], w0) - velocity_shift
    red_cont = wave2doppler(continuum_region[1], w0) - velocity_shift
    fit.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    fit.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

    # Mark fitting region
    fr = wave2doppler(fitting_region, w0) - velocity_shift

    # Mask out regions
    mask = (xdat.value < fr.value[0]) | (xdat.value > fr.value[1])
    for item in maskout:
        mask = mask | ((xdat.value > item.value[0]) & (xdat.value < item.value[1]))

    xdat1 = ma.array(xdat.value)
    xdat1[mask] = ma.masked

    for item in ma.extras.flatnotmasked_contiguous(xdat1):
        fit.axvspan(xdat1[item].min(), xdat1[item].max(), color=sns.color_palette('deep')[4], alpha=0.4)

    line, = fit.plot(np.sort(xdat.value), resid(pars, np.sort(xdat.value), mod), color='black')

    fit.set_xlim(xdat.min().value,xdat.max().value)


    residuals = fig.add_subplot(3,1,2)

    residuals.errorbar(xdat.value, ydat - resid(pars, xdat.value, mod) , yerr=yerr, linestyle='', alpha=0.4)

    residuals.set_xlim(fit.get_xlim())

    fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    residuals.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=12)
    residuals.set_ylabel("Residual")


    plt.figtext(0.05,0.25,r"Converged with $\chi^2$ = " + str(out.chisqr) + ", DOF = " + str(out.nfree))

    figtxt = ''
    for i in pars.valuesdict():
        figtxt += i + ' = {0} \n'.format( float('{0:.4g}'.format( pars[i].value)))

    plt.figtext(0.1,0.2,figtxt,fontsize=12,va='top')

    fig.tight_layout()

    if plot_savefig is not None:
        fig = fig.savefig(plot_savefig)

    plt.show()

    plt.close()

    return None

def fit_line(wav,
             flux,
             err,
             z=0.0,
             w0=6564.89*u.AA,
             velocity_shift=0.0*(u.km / u.s),
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=[6000,7000]*u.AA,
             nGaussians=0,
             nLorentzians=1,
             maskout=None,
             verbose=True,
             plot=True,
             plot_savefig='something.png',
             plot_title=''):

    """
    Velocity shift added to doppler shift to change zero point (can do if HW10
    redshift does not agree with Halpha centroid)

    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    """

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA

    # index is true in the region where we fit the continuum
    continuum = ((wav > continuum_region[0][0]) & \
                 (wav < continuum_region[0][1])) | \
                 ((wav > continuum_region[1][0]) & \
                 (wav < continuum_region[1][1]))

    # index of the region we want to fit
    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])

    # fit power-law to continuum region
    xdat = wav[continuum].value
    ydat = flux[continuum]
    yerr = err[continuum]

    bkgdmod = PowerLawModel()
    bkgdpars = bkgdmod.make_params()
    bkgdpars['exponent'].value = 1.0
    bkgdpars['amplitude'].value = 1.0

    out = minimize(resid,
                   bkgdpars,
                   args=(xdat, bkgdmod, ydat, yerr),
                   method='nelder')

    # subtract continuum, define region for fitting
    xdat = wav[fitting]
    ydat = flux[fitting] - resid(bkgdpars, wav[fitting].value, bkgdmod)
    yerr = err[fitting]

    # Transform to doppler shift
    xdat = wave2doppler(xdat, w0)

    # Add velocity shift
    xdat = xdat - velocity_shift

    bkgd = ConstantModel()
    mod = bkgd
    pars = bkgd.make_params()

    # A bit unnessesary, but I need a way to do += in the loop below
    pars['c'].value = 0.0
    pars['c'].vary = False

    for i in range(nGaussians):
        gmod = GaussianModel(prefix='g{}_'.format(i))
        mod += gmod
        pars += gmod.guess(ydat, x=xdat.value)

    for i in range (nLorentzians):
        lmod = LorentzianModel(prefix='l{}_'.format(i))
        mod += lmod
        pars += lmod.guess(ydat, x=xdat.value)

    for i in range(nGaussians):
        pars['g{}_center'.format(i)].value = 0.0
        pars['g{}_center'.format(i)].min = -5000.0
        pars['g{}_center'.format(i)].max = 5000.0
        pars['g{}_amplitude'.format(i)].min = 0.0

    for i in range(nLorentzians):
        pars['l{}_center'.format(i)].value = 0.0
        pars['l{}_center'.format(i)].min = -5000.0
        pars['l{}_center'.format(i)].max = 5000.0
        pars['l{}_amplitude'.format(i)].min = 0.0


    if maskout is not None:

        for item in maskout:
            xdat = ma.masked_inside(xdat, item[0], item[1])

        xdat = xdat[~ma.getmask(xdat)]
        ydat = ydat[~ma.getmask(xdat)]
        yerr = yerr[~ma.getmask(xdat)]

    out = minimize(resid,
                   pars,
                   args=(np.asarray(xdat), mod, ydat, yerr),
                   method ='nelder')

    if verbose:
        print fit_report(pars)

    # Convert Scipy cov matrix to standard covariance matrix.
    # cov = out.covar*dof / out.chisqr

    if plot:
        plot_fit(wav=wav,
                 flux = flux - resid(bkgdpars, wav.value, bkgdmod),
                 err=err,
                 pars=pars,
                 mod=mod,
                 out=out,
                 plot_savefig = plot_savefig,
                 maskout = maskout,
                 z=z,
                 w0=w0,
                 velocity_shift=velocity_shift,
                 continuum_region=continuum_region,
                 fitting_region=fitting_region,
                 plot_region=plot_region)


#    return pars, mod, x_data, dw, y_data_cr, y_sigma
    return None

###Testing###

if __name__ == '__main__':

    from get_spectra import get_liris_spec
    import matplotlib.pyplot as plt

    fname = '/data/lc585/WHT_20150331/html/SDSSJ1339+1515/dimcombLR+bkgd_v138.ms.fits'
    wavelength, dw, flux, err = get_liris_spec(fname)
    wav, flux, err = rebin_spectra(wavelength,
                                   flux,
                                   er=err,
                                   n=1,
                                   weighted=False)
    fit_line(wav,
             flux,
             err,
             z=2.318977,
             w0=6564.89*u.AA,
             maskout=[[0,1000]]*(u.km / u.s),
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             plot_title='SDSSJ133916.88+151507.6',
             plot_region=[6000,7000]*u.AA)






