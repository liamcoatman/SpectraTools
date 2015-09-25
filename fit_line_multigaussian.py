# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:04:15 2015

@author: lc585

Fit emission line with many Gaussain components using linear regression
Adapted from Andrew Connollrys AstroMl lecture course.

"""

import numpy as np
import astropy.units as u
from lmfit.models import GaussianModel, LorentzianModel, PowerLawModel
from lmfit import minimize
import numpy.ma as ma
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from spectra.fit_line import wave2doppler, resid
from scipy.interpolate import interp1d

def plot_fit(wav=None,
             flux=None,
             err=None,
             wav_mod=None,
             flux_mod=None,
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
    vdat = wave2doppler(wav, w0)

    # Add velocity shift
    vdat = vdat - velocity_shift


    xdat = wav[plot_region_inds]
    vdat = vdat[plot_region_inds]
    ydat = flux[plot_region_inds]
    yerr = err[plot_region_inds]

    fig = plt.figure(figsize=(6,8))

    fit = fig.add_subplot(2,1,1)
    fit.set_xticklabels( () )
    residuals = fig.add_subplot(2,1,2)

    fit.scatter(vdat.value, ydat, alpha=0.4)

    # Mark continuum fitting region
    blue_cont = wave2doppler(continuum_region[0], w0) - velocity_shift
    red_cont = wave2doppler(continuum_region[1], w0) - velocity_shift
    fit.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    fit.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    residuals.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    residuals.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

    # Mark fitting region
    fr = wave2doppler(fitting_region, w0) - velocity_shift

    # Mask out regions
    xdat_masking = np.arange(xdat.min().value, xdat.max().value, 0.05)*(u.AA)
    vdat_masking = wave2doppler(xdat_masking, w0) - velocity_shift

    mask = (vdat_masking.value < fr.value[0]) | (vdat_masking.value > fr.value[1])


    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            for item in maskout:
                mask = mask | ((vdat_masking.value > item.value[0]) & (vdat_masking.value < item.value[1]))

        elif maskout.unit == (u.AA):

            for item in maskout:
                xmin = item.value[0] / (1.0 + z)
                xmax = item.value[1] / (1.0 + z)
                mask = mask | ((xdat_masking.value > xmin) & (xdat_masking.value < xmax))


    vdat1_masking = ma.array(vdat_masking.value)
    vdat1_masking[mask] = ma.masked

    for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
        fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), color=sns.color_palette('deep')[4], alpha=0.4)
        residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), color=sns.color_palette('deep')[4], alpha=0.4)

    line, = fit.plot(wav_mod, flux_mod, color='black')

    plotting_limits = wave2doppler(plot_region, w0) - velocity_shift
    fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)

    f = interp1d(wav_mod, flux_mod, kind='cubic', bounds_error=False, fill_value=0.0)
    residuals.errorbar(vdat.value, ydat - f(vdat) , yerr=yerr, linestyle='', alpha=0.4)

    residuals.set_xlim(fit.get_xlim())


    fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    residuals.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=12)
    residuals.set_ylabel("Residual")


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
             nGaussians=6,
             maskout=None,
             sigma_clip=False,
             nGaussians_sigma_clip=5,
             reject_sigma=0.6,
             plot=True,
             plot_savefig='something.png',
             plot_title='',
             save_dir=None):

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
                   method='leastsq')

    # subtract continuum, define region for fitting
    xdat = wav[fitting]
    ydat = flux[fitting] - resid(bkgdpars, wav[fitting].value, bkgdmod)
    yerr = err[fitting]

    # Transform to doppler shift
    vdat = wave2doppler(xdat, w0)

    # Add velocity shift
    vdat = vdat - velocity_shift


    """
    Remember that this is to velocity shifted array

    Accepts units km/s or angstrom (observed frame)
    """

    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            mask = np.array([True] * len(vdat))
            for item in maskout:
                print 'Not fitting between {0} and {1}'.format(item[0], item[1])
                mask[(vdat > item[0]) & (vdat < item[1])] = False


        elif maskout.unit == (u.AA):

            mask = np.array([True] * len(vdat))
            for item in maskout:
                vlims = wave2doppler(item / (1.0 + z), w0) - velocity_shift
                print 'Not fitting between {0} ({1}) and {2} ({3})'.format(item[0], vlims[0], item[1], vlims[1])
                mask[(xdat > (item[0] / (1.0 + z))) & (xdat < (item[1] / (1.0 + z)))] = False

        else:
            print "Units must be km/s or angstrom"

        vdat = vdat[mask]
        ydat = ydat[mask]
        yerr = yerr[mask]


    vs = np.linspace(vdat.value.min(), vdat.value.max(), 1000)

    def gaussian_basis(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    if sigma_clip:

        # Define our Gaussians
        basis_mu = np.linspace(vdat.value.min(), vdat.value.max(), nGaussians_sigma_clip+2)[1:-1]
        basis_sigma = 1. * (basis_mu[1] - basis_mu[0])

        M = np.zeros(shape=[nGaussians_sigma_clip, vdat.value.shape[0]])

        for i in range(nGaussians_sigma_clip):
            M[i] = gaussian_basis(vdat.value, basis_mu[i], basis_sigma)

        M = np.matrix(M).T
        C = np.matrix(np.diagflat(yerr**2))
        Y = np.matrix(ydat).T

        coeff = (M.T * C.I * M).I * (M.T * C.I * Y)

        # draw the fit to the data
        mu_fit = np.zeros(len(vs))
        for i in range(nGaussians_sigma_clip):
            mu_fit += coeff[i,0]*gaussian_basis(vs, basis_mu[i], basis_sigma)

        f = interp1d(vs, mu_fit, kind='cubic')

        bad = np.abs(ydat - f(vdat)) / yerr > reject_sigma

        ydat_cr = ydat[~bad]
        vdat_cr = vdat[~bad]
        yerr_cr = yerr[~bad]

        fig, ax = plt.subplots()
        ax.scatter(vdat, ydat, color='grey', alpha=0.4)
        ax.scatter(vdat_cr, ydat_cr, color='red')
        ax.plot(vs, mu_fit, color='black')
        ax.set_title('Sigma Clipping')
        plt.show()

    else:

        ydat_cr = ydat
        vdat_cr = vdat
        yerr_cr = yerr

    basis_mu = np.linspace(vdat_cr.value.min(), vdat_cr.value.max(), nGaussians+2)[1:-1]
    basis_sigma = 1. * (basis_mu[1] - basis_mu[0])

    M = np.zeros(shape=[nGaussians, vdat_cr.value.shape[0]])

    for i in range(nGaussians):
        M[i] = gaussian_basis(vdat_cr.value, basis_mu[i], basis_sigma)

    M = np.matrix(M).T
    C = np.matrix(np.diagflat(yerr_cr**2))
    Y = np.matrix(ydat_cr).T
    coeff = (M.T * C.I * M).I * (M.T * C.I * Y)

    # draw the fit to the data
    mu_fit = np.zeros(len(vs))
    for i in range(nGaussians):
        mu_fit += coeff[i,0]*gaussian_basis(vs, basis_mu[i], basis_sigma)


    # Calculate full width at half maximum


    imax = np.argmax(mu_fit)

    halfmax = mu_fit[imax] / 2.0
    zpeak = vs[imax]

    i = 0
    while mu_fit[i] < halfmax:
        i += 1

    imid1 = i

    i = len(vs) - 1
    while mu_fit[i] < halfmax:
        i -= 1

    imid2 = i

    print 'Peak: {}'.format(zpeak)
    print 'FWHM: {}'.format(vs[imid2] - vs[imid1])

    # Median

    cdf = np.cumsum(mu_fit/np.sum(mu_fit))
    md = vs[np.argmin( np.abs( cdf - 0.5))]
    print 'Median: {}'.format(md)


    if plot:
        plot_fit(wav=wav,
                 flux = flux - resid(bkgdpars, wav.value, bkgdmod),
                 err=err,
                 wav_mod=vs,
                 flux_mod=mu_fit,
                 plot_savefig = plot_savefig,
                 maskout = maskout,
                 z=z,
                 w0=w0,
                 velocity_shift=velocity_shift,
                 continuum_region=continuum_region,
                 fitting_region=fitting_region,
                 plot_region=plot_region,
                 plot_title=plot_title)

    # Save results
    vs = np.linspace(-20000, 20000, 1000)
    mu_fit = np.zeros(len(vs))
    for i in range(nGaussians):
        mu_fit += coeff[i,0]*gaussian_basis(vs, basis_mu[i], basis_sigma)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dat_file = os.path.join(save_dir, 'vdat_mod.txt')
    parfile = open(dat_file, 'wb')
    pickle.dump(vs, parfile, -1)
    parfile.close()

    dat_file = os.path.join(save_dir, 'ydat_mod.txt')
    parfile = open(dat_file, 'wb')
    pickle.dump(mu_fit, parfile, -1)
    parfile.close()


    return None


if __name__ == '__main__':

    from get_spectra import get_boss_dr12_spec
    import matplotlib.pyplot as plt

#    wav, dw, flux, err = get_boss_dr12_spec('SDSSJ133646.87+144334.2')
#
#    fit_line(wav,
#             flux,
#             err,
#             z=2.146961,
#             w0=np.mean([1548.202,1550.774])*u.AA,
#             velocity_shift=0.0*(u.km / u.s),
#             continuum_region=[[1445.,1465.]*u.AA,[1700.,1705.]*u.AA],
#             fitting_region=[1460.,1575.]*u.AA,
#             plot_region=[1440,1720]*u.AA,
#             nGaussians=6,
#             sigma_clip=True,
#             nGaussians_sigma_clip=4,
#             reject_sigma = 0.6,
#             maskout= [[4479.1, 4492.7], [4500.8, 4516.6], [4728.1, 4790.7], [4835.7, 4870.9], [5573.14, 5583.42]]*u.AA,
#             plot=True,
#             save_dir='/data/lc585/WHT_20150331/NewLineFits/SDSSJ133646.87+144334.2/multigaussian/CIV')


#    Peak: -2546.51671202
#    FWHM: 8763.24854151
#    Median: -3833.27219407

#    wav, dw, flux, err = get_boss_dr12_spec('SDSSJ085856.00+015219.4')
#
#    fit_line(wav,
#             flux,
#             err,
#             z=2.169982,
#             w0=np.mean([1548.202,1550.774])*u.AA,
#             velocity_shift=0.0*(u.km / u.s),
#             continuum_region=[[1445.,1465.]*u.AA,[1700.,1705.]*u.AA],
#             fitting_region=[1460.,1600.]*u.AA,
#             plot_region=[1440,1720]*u.AA,
#             nGaussians=10,
#             sigma_clip=False,
#             nGaussians_sigma_clip=4,
#             reject_sigma = 0.6,
#             maskout=[[4492.5, 4498.9], [4554.0, 4572.0], [4732.5, 4746.9], [4760.9, 4784.2], [4851.7, 4868.7], [4915.8, 4925.0], [5573.14, 5583.42]]*u.AA,
#             plot=True,
#             save_dir='/data/lc585/WHT_20150331/NewLineFits/SDSSJ085856.00+015219.4/multigaussian/CIV')

#    Peak: -2620.98126447
#    FWHM: 8478.18171296
#    Median: -4079.01251447


    wav, dw, flux, err = get_boss_dr12_spec('SDSSJ123611.21+112921.6')

    fit_line(wav,
             flux,
             err,
             z=2.155473,
             w0=np.mean([1548.202,1550.774])*u.AA,
             velocity_shift=0.0*(u.km / u.s),
             continuum_region=[[1445.,1465.]*u.AA,[1700.,1705.]*u.AA],
             fitting_region=[1460.,1580.]*u.AA,
             plot_region=[1440,1720]*u.AA,
             nGaussians=8,
             sigma_clip=False,
             nGaussians_sigma_clip=4,
             reject_sigma = 0.6,
             maskout=[[5573.14, 5583.42]]*u.AA,
             plot=True,
             save_dir='/data/lc585/WHT_20150331/NewLineFits/SDSSJ123611.21+112921.6/multigaussian/CIV')

#    Peak: -1596.15413851
#    FWHM: 7577.11250264
#    Median: -2777.90562975
