# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:04:15 2015

@author: lc585

Fit emission line with many sixth order Gauss-Hermite polynomial
References: van der Marel & Franx (1993); Cappelari (2000)

"""
import matplotlib
matplotlib.use("qt4agg")

import numpy as np
import astropy.units as u
from lmfit import minimize, Parameters, fit_report
from lmfit.models import PowerLawModel
from lmfit.model import Model
import numpy.ma as ma
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from spectra.fit_line import wave2doppler, resid, doppler2wave
from scipy.interpolate import interp1d
import math
from scipy import integrate, optimize
from astropy.io import fits

def resid(p, x, model, data=None, sigma=None):

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

def gh_resid(p, x, order, data=None, sigma=None):

        mod = gausshermite(x, p, order)

        if data is not None:
            resids = mod - data
            if sigma is not None:
                weighted = np.sqrt(resids ** 2 / sigma ** 2)
                print weighted.sum()
                return weighted
            else:
                return resids
        else:
            return mod

def gausshermite(x, p, order):

    h0 = (p['amp0'].value/(np.sqrt(2*math.pi)*p['sig0'].value)) * np.exp(-(x-p['cen0'].value)**2 /(2*p['sig0'].value**2))

    if order > 0:
        h1 = np.sqrt(2.0) * x * (p['amp1'].value/(np.sqrt(2*math.pi)*p['sig1'].value)) * np.exp(-(x-p['cen1'].value)**2 /(2*p['sig1'].value**2))

    if order > 1:
        h2 = (2.0*x*x - 1.0) / np.sqrt(2.0) * (p['amp2'].value/(np.sqrt(2*math.pi)*p['sig2'].value)) * np.exp(-(x-p['cen2'].value)**2 /(2*p['sig2'].value**2))

    if order > 2:
        h3 = x * (2.0*x*x - 3.0) / np.sqrt(3.0) * (p['amp3'].value/(np.sqrt(2*math.pi)*p['sig3'].value)) * np.exp(-(x-p['cen3'].value)**2 /(2*p['sig3'].value**2))

    if order > 3:
        h4 = (x*x*(4.0*x*x-12.0)+3.0) / (2.0*np.sqrt(6.0)) * (p['amp4'].value/(np.sqrt(2*math.pi)*p['sig4'].value)) * np.exp(-(x-p['cen4'].value)**2 /(2*p['sig4'].value**2))

    if order > 4:
        h5 = (x*(x*x*(4.0*x*x-20.0) + 15.0)) / (2.0*np.sqrt(15.0)) * (p['amp5'].value/(np.sqrt(2*math.pi)*p['sig5'].value)) * np.exp(-(x-p['cen5'].value)**2 /(2*p['sig5'].value**2))

    if order > 5:
        h6 = (x*x*(x*x*(8.0*x*x-60.0) + 90.0) - 15.0) / (12.0*np.sqrt(5.0)) * (p['amp6'].value/(np.sqrt(2*math.pi)*p['sig6'].value)) * np.exp(-(x-p['cen6'].value)**2 /(2*p['sig6'].value**2))

    if order == 0:
        return h0
    if order == 1:
        return h0 + h1
    if order == 2:
        return h0 + h1 + h2
    if order == 3:
        return h0 + h1 + h2 + h3
    if order == 4:
        return h0 + h1 + h2 + h3 + h4
    if order == 5:
        return h0 + h1 + h2 + h3 + h4 + h5
    if order == 6:
        return h0 + h1 + h2 + h3 + h4 + h5 + h6

def fit_line(wav,
             flux,
             err,
             z=0.0,
             w0=6564.89*u.AA,
             velocity_shift=0.0*(u.km / u.s),
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=[6000,7000]*u.AA,
             red_shelf = [1580,1690]*u.AA,
             maskout=None,
             order=6,
             plot=True,
             plot_title='',
             verbose=True):

    """
    Velocity shift added to doppler shift to change zero point (can do if HW10
    redshift does not agree with Halpha centroid)

    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    """

    with open(os.path.join('/data/lc585/WHT_20150331/fit_errors_2',plot_title+'_CIV.dat'), 'w') as f:
        f.write('Name Centroid FWHM Median p99 p95 p90 p80 p60 Mean Sigma EQW\n')

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA

    # Check if continuum is given in wavelength or doppler units
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    # Fit to median
    blue_inds = (wav > continuum_region[0][0]) & (wav < continuum_region[0][1])
    red_inds = (wav > continuum_region[1][0]) & (wav < continuum_region[1][1])

    blue_flux, red_flux = flux[blue_inds], flux[red_inds]
    blue_err, red_err = err[blue_inds], err[red_inds]
    blue_wav, red_wav = wav[blue_inds], wav[red_inds]

    n_samples = 1

    flux_array = np.zeros((len(flux), n_samples))
    for i in range(len(flux)):
        if np.abs(err[i]) > 0.0:
            flux_array[i,:] = np.random.normal(flux[i], np.abs(err[i]), n_samples)
        else:
            flux_array[i,:] = np.repeat(flux[i], n_samples)

    blue_flux_array = flux_array[blue_inds,:]
    red_flux_array =  flux_array[red_inds,:]

    xdat_bkgd = np.array([continuum_region[0].mean().value, continuum_region[1].mean().value] )
    ydat_bkgd = np.array([np.median(blue_flux_array,axis=0), np.median(red_flux_array,axis=0) ]).T

    # index of the region we want to fit
    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])

    xdat_fit = wav[fitting]
    ydat_fit = flux[fitting]
    yerr_fit = err[fitting]
    flux_array_fit = flux_array[fitting,:]


    # Transform to doppler shift
    vdat_fit = wave2doppler(xdat_fit, w0)

    # Add velocity shift
    vdat_fit = vdat_fit - velocity_shift

    """
    Remember that this is to velocity shifted array

    Accepts units km/s or angstrom (observed frame)
    """

    # mask out redshelf from fit
    mask = np.array([True] * len(vdat_fit))
    red_shelf_start = wave2doppler(red_shelf[0], w0)
    red_shelf_end = wave2doppler(red_shelf[1], w0)

    mask[(vdat_fit > red_shelf_start) & (vdat_fit < red_shelf_end)] = False

    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            for item in maskout:
                if verbose:
                    print 'Not fitting between {0} and {1}'.format(item[0], item[1])
                mask[(vdat_fit > item[0]) & (vdat_fit < item[1])] = False


        elif maskout.unit == (u.AA):

            for item in maskout:
                vlims = wave2doppler(item / (1.0 + z), w0) - velocity_shift
                if verbose:
                    print 'Not fitting between {0} ({1}) and {2} ({3})'.format(item[0], vlims[0], item[1], vlims[1])
                mask[(xdat_fit > (item[0] / (1.0 + z))) & (xdat_fit < (item[1] / (1.0 + z)))] = False

        else:
            if verbose:
                print "Units must be km/s or angstrom"

    vdat_fit = vdat_fit[mask]
    ydat_fit = ydat_fit[mask]
    yerr_fit = yerr_fit[mask]
    xdat_fit = xdat_fit[mask]
    flux_array_fit = flux_array_fit[mask,:]


    if plot:

        fig = plt.figure(figsize=(6,8))
        fit = fig.add_subplot(2,1,1)
        fit.set_xticklabels( () )
        residuals = fig.add_subplot(2,1,2)

        # Mark continuum fitting region
        blue_cont = wave2doppler(continuum_region[0], w0)
        red_cont = wave2doppler(continuum_region[1], w0)

        fit.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
        fit.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

        residuals.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
        residuals.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

        # Mark fitting region
        fr = wave2doppler(fitting_region, w0)

        # Mask out regions
        xdat_masking = np.arange(xdat_fit.min().value, xdat_fit.max().value, 0.05)*(u.AA)
        vdat_masking = wave2doppler(xdat_masking, w0)

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

        # mask out redshelf from fit
        mask = mask | ((xdat_masking.value > red_shelf[0].value) & (xdat_masking.value < red_shelf[1].value))

        vdat1_masking = ma.array(vdat_masking.value)
        vdat1_masking[mask] = ma.masked

        for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
            fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), color=sns.color_palette('deep')[4], alpha=0.4)
            residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), color=sns.color_palette('deep')[4], alpha=0.4)

        fit.set_ylabel(r'F$_\lambda$', fontsize=12)
        residuals.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=12)
        residuals.set_ylabel("Residual")

        if plot_region.unit == (u.km/u.s):
            plot_region = doppler2wave(plot_region, w0)

        plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])
        plotting_limits = wave2doppler(plot_region, w0)

        fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)
        residuals.set_xlim(fit.get_xlim())

    for k in range(n_samples):

        print k

        bkgdmod = PowerLawModel()
        bkgdpars = bkgdmod.make_params()
        bkgdpars['exponent'].value = 1.0
        bkgdpars['amplitude'].value = 1.0

        out = minimize(resid,
                       bkgdpars,
                       args=(xdat_bkgd, bkgdmod, ydat_bkgd[k,:]),
                       method='leastsq')


        if verbose:
            print fit_report(bkgdpars)

        # subtract continuum, define region for fitting
        flux_array_fit_bkgdsub = flux_array_fit[:,k] - resid(bkgdpars, xdat_fit.value, bkgdmod)

        # Calculate mean and variance
        p =  flux_array_fit_bkgdsub / np.sum(flux_array_fit_bkgdsub)
        m = np.sum(vdat_fit * p)
        v = np.sum(p * (vdat_fit-m)**2)
        sd = np.sqrt(v)
        print sd

        pars = Parameters()

        pars.add('amp0', value = 1.0, min=0.0)
        pars.add('sig0', value = 1.0, min=0.1)
        pars.add('cen0', value = 0.0, min=vdat_fit.min().value/sd.value, max=vdat_fit.max().value/sd.value)

        if order > 0:
            pars.add('amp1', value = 1.0, min=0.0)
            pars.add('sig1', value = 1.0, min=0.1)
            pars.add('cen1', value = 0.0, min=vdat_fit.min().value/sd.value, max=vdat_fit.max().value/sd.value)

        if order > 1:
            pars.add('amp2', value = 1.0, min=0.0)
            pars.add('sig2', value = 1.0, min=0.1)
            pars.add('cen2', value = 0.0, min=vdat_fit.min().value/sd.value, max=vdat_fit.max().value/sd.value)

        if order > 2:
            pars.add('amp3', value = 1.0, min=0.0)
            pars.add('sig3', value = 1.0, min=0.1)
            pars.add('cen3', value = 0.0, min=vdat_fit.min().value/sd.value, max=vdat_fit.max().value/sd.value)

        if order > 3:
            pars.add('amp4', value = 1.0, min=0.0)
            pars.add('sig4', value = 1.0, min=0.1)
            pars.add('cen4', value = 0.0, min=vdat_fit.min().value/sd.value, max=vdat_fit.max().value/sd.value)

        if order > 4:
            pars.add('amp5', value = 1.0, min=0.0)
            pars.add('sig5', value = 1.0, min=0.1)
            pars.add('cen5', value = 0.0, min=vdat_fit.min().value/sd.value, max=vdat_fit.max().value/sd.value)

        if order > 5:
            pars.add('amp6', value = 1.0, min=0.0)
            pars.add('sig6', value = 1.0, min=0.1)
            pars.add('cen6', value = 0.0, min=vdat_fit.min().value/sd.value, max=vdat_fit.max().value/sd.value)

        out = minimize(gh_resid,
                       pars,
                       args=(vdat_fit.value/sd.value, order, flux_array_fit_bkgdsub, yerr_fit),
                       method='leastsq')

        # fit.scatter(vdat_fit.value, flux_array_fit_bkgdsub)
        # save_dir = os.path.join('/data/lc585/WHT_20150331/NewLineFits3/SDSSJ110454.73+095714.8/GaussHermite/CIV')
        # parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
        # params_CIV = Parameters()
        # params_CIV.load(parfile)
        # parfile.close()
        # fit.plot(vdat_fit.value, gh_resid(params_CIV,vdat_fit.value/sd.value,order))


    #     if verbose:
    #         for key, value in pars.valuesdict().items():
    #             if 'cen' in key:
    #                 print key, value * sd.value
    #             else:
    #                 print key, value


    #     # Calculate FWHM of distribution

    #     integrand = lambda x: gh_resid(pars, x, order)

    #     # func_center = optimize.fmin(lambda x: -integrand(x) , 0)[0]

    #     # print 'Peak: {}'.format(func_center * sd.value)

    #     # half_max = integrand(func_center) / 2.0

    #     # root1 = optimize.brentq(lambda x: integrand(x) - half_max, vdat.min().value, func_center)
    #     # root2 = optimize.brentq(lambda x: integrand(x) - half_max, func_center, vdat.max().value)

    #     # print 'FWHM: {}'.format((root2 - root1)* sd.value)

    #     dv = 1.0

    #     xs = np.arange(vdat_fit.min().value, vdat_fit.max().value, dv) / sd.value
    #     # xs = np.arange(-2e4, 2e4, dv) / sd.value

    #     norm = np.sum(integrand(xs) * dv)
    #     pdf = integrand(xs) / norm
    #     cdf = np.cumsum(pdf)
    #     cdf_r = np.cumsum(pdf[::-1])[::-1] # reverse cumsum

    #     func_center = xs[np.argmax(pdf)]
    #     # print 'Peak: {}'.format(func_center * sd.value)

    #     half_max = np.max(pdf) / 2.0

    #     i = 0
    #     while pdf[i] < half_max:
    #         i+=1

    #     root1 = xs[i]

    #     i = 0
    #     while pdf[-i] < half_max:
    #         i+=1

    #     root2 = xs[-i]

    #     # print 'FWHM: {}'.format((root2 - root1)* sd.value)

    #     md = xs[np.argmin( np.abs( cdf - 0.5))]
    #     # print 'Median: {}'.format(md * sd.value)


    #     # Not sure this would work if median far from zero but probably would never happen.
    #     p99 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.005))] - xs[np.argmin(np.abs(cdf - 0.005))])* sd.value
    #     p95 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.025))] - xs[np.argmin(np.abs(cdf - 0.025))])* sd.value
    #     p90 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.05))] - xs[np.argmin(np.abs(cdf - 0.05))])* sd.value
    #     p80 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.1))] - xs[np.argmin(np.abs(cdf - 0.1))])* sd.value
    #     p60 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.2))] - xs[np.argmin(np.abs(cdf - 0.2))])* sd.value

    #     # print '99%: {}'.format(p99)
    #     # print '95%: {}'.format(p95)
    #     # print '90%: {}'.format(p90)
    #     # print '80%: {}'.format(p80)
    #     # print '60%: {}'.format(p60)

    #     m = np.sum(xs * pdf * dv)
    #     # print 'Mean: {}'.format(m * sd.value)

    #     """
    #     This is working, but for Lorentzian the second moment is not defined.
    #     It dies off much less quickly than the Gaussian so the sigma is much
    #     larger. I therefore need to think of the range in which I calcualte
    #     sigma
    #     """

    #     v = np.sum( (xs-m)**2 * pdf * dv )
    #     sigma = np.sqrt(v)
    #     # print 'Second moment: {}'.format(sigma * sd.value)

    #     xs = np.arange(vdat_fit.min().value, vdat_fit.max().value, dv) / sd.value
    #     flux_line = integrand(xs)
    #     xs_wav = doppler2wave(xs*sd,w0)
    #     flux_bkgd = bkgdmod.eval(params=bkgdpars, x=xs_wav.value)
    #     f = (flux_line + flux_bkgd) / flux_bkgd
    #     eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
    #     eqw = np.nansum(eqw)


    #     with open(os.path.join('/data/lc585/WHT_20150331/fit_errors_2',plot_title+'_CIV.dat'), 'a') as f:
    #         f.write('{0} {1:.2f} {2:.2f} {3:.2f} {4} {5} {6} {7} {8} {9:.2f} {10:.2f} {11:.4f} \n'.format(plot_title,
    #                                                                                             func_center * sd.value,
    #                                                                                             (root2 - root1)* sd.value,
    #                                                                                             md * sd.value,
    #                                                                                             p99,
    #                                                                                             p95,
    #                                                                                             p90,
    #                                                                                             p80,
    #                                                                                             p60,
    #                                                                                             m*sd.value,
    #                                                                                             sigma*sd.value,
    #                                                                                             eqw))

    #     if plot:

    #         xdat_plot = wav[plot_region_inds]
    #         vdat_plot = wave2doppler(xdat_plot, w0)
    #         ydat_plot = flux_array[plot_region_inds,k] - resid(bkgdpars, wav[plot_region_inds].value, bkgdmod)
    #         yerr_plot = err[plot_region_inds]

    #         if 'pts1' in locals():
    #             pts1.set_data((vdat_plot.value, ydat_plot))
    #         else:
    #             pts1, = fit.plot(vdat_plot.value, ydat_plot)

    #         vs = np.linspace(vdat_fit.min().value, vdat_fit.max().value, 1000)

    #         flux_mod = gausshermite(vs/sd.value, pars, order)

    #         if 'line' in locals():
    #             line.set_data((vs, flux_mod))
    #         else:
    #             line, = fit.plot(vs, flux_mod, color='black')


    #         if 'pts2' in locals():
    #             pts2.set_data((vdat_plot.value,ydat_plot - gausshermite(vdat_plot.value/sd.value, pars, order)))
    #         else:
    #             pts2, = residuals.plot(vdat_plot.value,
    #                                    ydat_plot- gausshermite(vdat_plot.value/sd.value, pars, order))


    #         fig.set_tight_layout(True)

    #         plt.pause(0.1)

    # plt.close()

    plt.show()

    print 'Done!'

    return None


if __name__ == '__main__':

    from get_spectra import get_boss_dr12_spec
    import matplotlib.pyplot as plt
    from spectra.get_wavelength import get_wavelength

    class PlotProperties(object):

        def __init__(self,
                     sdss_name,
                     boss_name,
                     z,
                     civ_gh_order,
                     civ_rebin,
                     civ_nGaussians,
                     civ_nLorentzians,
                     civ_continuum_region,
                     civ_fitting_region,
                     civ_maskout,
                     civ_red_shelf,
                     ha_gh_order,
                     ha_rebin,
                     ha_nGaussians,
                     ha_nLorentzians,
                     ha_continuum_region,
                     ha_fitting_region,
                     ha_maskout,
                     name):

            self.sdss_name = sdss_name
            self.boss_name = boss_name
            self.z = z
            self.civ_gh_order = civ_gh_order
            self.civ_rebin = civ_rebin
            self.civ_nGaussians = civ_nGaussians
            self.civ_nLorentzians = civ_nLorentzians
            self.civ_continuum_region = civ_continuum_region
            self.civ_fitting_region = civ_fitting_region
            self.civ_maskout = civ_maskout
            self.civ_red_shelf = civ_red_shelf
            self.ha_gh_order = ha_gh_order
            self.ha_rebin = ha_rebin
            self.ha_nGaussians = ha_nGaussians
            self.ha_nLorentzians = ha_nLorentzians
            self.ha_continuum_region = ha_continuum_region
            self.ha_fitting_region = ha_fitting_region
            self.ha_maskout =  ha_maskout
            self.name = name

    SDSSJ1104 = PlotProperties(sdss_name = 'SDSSJ110454.73+095714.8',
                            boss_name = 'SDSSJ110454.73+095714.8',
                            z = 2.421565,
                            civ_gh_order = 6,
                            civ_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            civ_continuum_region=[[1445.,1465.]*u.AA,[1700.,1705.]*u.AA],
                            civ_fitting_region=[1500.,1600.]*u.AA,
                            civ_maskout  = [[5238.3, 5250.4], [5254.0, 5263.7], [5573.14, 5583.42]]*u.AA,
                            civ_red_shelf = [1600,1690]*u.AA,
                            ha_gh_order = 4,
                            ha_rebin  = 2,
                            ha_nGaussians  = 2,
                            ha_nLorentzians = 0,
                            ha_continuum_region=[[6000.,6250.]*u.AA,[10800,12800]*(u.km/u.s)],
                            ha_fitting_region=[6400,6800]*u.AA,
                            ha_maskout  = None,
                            name = 'SDSSJ1104+0957')

    fname = os.path.join('/data/lc585/WHT_20150331/BOSS',SDSSJ1104.sdss_name + '.fits' )

    hdulist = fits.open(fname)
    data = hdulist[1].data
    hdr = hdulist[0].header
    hdulist.close()

    wav = 10**np.array([j[1] for j in data ])
    dw = wav * (10**hdr['COEFF1'] - 1.0 )
    flux = np.array([j[0] for j in data ])
    err = np.sqrt( np.array([j[6] for j in data ])) # square root of sky

    fit_line(wav,
            flux,
            err,
            z=SDSSJ1104.z,
            w0=np.mean([1548.202,1550.774])*u.AA,
            velocity_shift=0.0*(u.km / u.s),
            continuum_region=SDSSJ1104.civ_continuum_region,
            fitting_region=SDSSJ1104.civ_fitting_region,
            plot_region=[-30000,40000]*(u.km/u.s),
            red_shelf=SDSSJ1104.civ_red_shelf,
            maskout=SDSSJ1104.civ_maskout,
            order=SDSSJ1104.civ_gh_order,
            plot=True,
            verbose=False,
            plot_title=SDSSJ1104.sdss_name)

