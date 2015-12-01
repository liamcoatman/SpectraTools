# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:04:15 2015

@author: lc585

Fit emission line with many sixth order Gauss-Hermite polynomial
References: van der Marel & Franx (1993); Cappelari (2000)

"""

import numpy as np
import astropy.units as u
from lmfit import minimize, Parameters, fit_report
from lmfit.models import PowerLawModel
from lmfit.model import Model
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from SpectraTools.fit_line import wave2doppler, resid, doppler2wave
from scipy.interpolate import interp1d
import math
from scipy import integrate, optimize
from scipy.ndimage.filters import median_filter
from scipy.stats import norm
from palettable.colorbrewer.qualitative import Set2_5

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

def plot_fit(wav=None,
             flux=None,
             err=None,
             xscal = 1.0,
             pars = Parameters(),
             order = 6,
             plot_savefig=None,
             plot_title='',
             save_dir = None,
             maskout=None,
             red_shelf = [1580,1690]*u.AA,
             z=0.0,
             w0=6564.89*u.AA,
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=None,
             line_region=[-10000,10000]*(u.km/u.s)):


    # plotting region

    if plot_region.unit == (u.km/u.s):
        plot_region = doppler2wave(plot_region, w0)

    plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])

    # Transform to doppler shift
    vdat = wave2doppler(wav, w0)

    xdat = wav[plot_region_inds]
    vdat = vdat[plot_region_inds]
    ydat = flux[plot_region_inds]
    yerr = err[plot_region_inds]

    plt.rc('axes', color_cycle=Set2_5.mpl_colors) 
    fig = plt.figure(figsize=(6,18))

    fit = fig.add_subplot(5,1,1)
    fit.set_xticklabels( () )
    residuals = fig.add_subplot(5,1,2)
    residuals.set_xticklabels( () )
    eb = fig.add_subplot(5,1,3)

    fs = fig.add_subplot(5,1,5)  

    fit.axvline(line_region.value[0], color='black', linestyle='--')
    fit.axvline(line_region.value[1], color='black', linestyle='--')

    residuals.axvline(line_region.value[0], color='black', linestyle='--')
    residuals.axvline(line_region.value[1], color='black', linestyle='--')

    #fit.errorbar(vdat.value, ydat, yerr=yerr, linestyle='', alpha=0.4)
    #fit.scatter(vdat.value, ydat, edgecolor='None', alpha=0.5)
    fit.scatter(vdat.value, ydat, edgecolor='None', s=15, alpha=0.9, facecolor='black')

    # Mark continuum fitting region
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    
    # Mark fitting region
    fr = wave2doppler(fitting_region, w0)

    # Mask out regions
    xdat_masking = np.arange(xdat.min().value, xdat.max().value, 0.05)*(u.AA)
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
        fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='moccasin')
        residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='moccasin')
        eb.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='moccasin')
        fs.axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, alpha=0.4, color='moccasin')

    # Now do continuum regions, which is now in wavelength units 
    xdat_masking = np.arange(xdat.min().value, xdat.max().value, 0.05)*(u.AA)
    vdat_masking = wave2doppler(xdat_masking, w0)

    mask = (xdat_masking.value < continuum_region[0][0].value) | \
           ((xdat_masking.value > continuum_region[0][1].value) &  (xdat_masking.value < continuum_region[1][0].value))  | \
           (xdat_masking.value > continuum_region[1][1].value)

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
        fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')
        residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')
        eb.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')  
        fs.axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, alpha=0.4, color='powderblue')    

    plotting_limits = wave2doppler(plot_region, w0)

    vdat2 = wave2doppler(wav, w0)
    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])
    vs = np.linspace(vdat.min().value, vdat.max().value, 1000)
    #vs = np.linspace(-30000, 20000, 1000)

    flux_mod = gausshermite(vs/xscal, pars, order)

    line, = fit.plot(vs, flux_mod, color='black', lw=2)

    fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)

    fit.axhline(0, color='black', linestyle='--')

    eb.errorbar(vdat.value, ydat, yerr=yerr, linestyle='', alpha=0.5, color='grey')
    eb.plot(vs, flux_mod, color='black', lw=2)




    # residuals.errorbar(vdat.value,
    #                    ydat - gausshermite(vdat.value/xscal, pars, order),
    #                    yerr=yerr,
    #                    linestyle='',
    #                    alpha=0.4)

    residuals.scatter(vdat.value, (ydat - gausshermite(vdat.value/xscal, pars, order)) / yerr, edgecolor='None', s=15, alpha=0.9, facecolor='black')
    # residuals.plot(vdat.value, median_filter((ydat - gausshermite(vdat.value/xscal, pars, order)) / yerr, 3.0), color='black')


    residuals.set_ylim(-5,5)
    residuals.axhline(0.0, color='black', linestyle='--')


    # residuals.scatter(vdat.value,
    #                   ydat - gausshermite(vdat.value/xscal, pars, order))

    residuals.set_xlim(fit.get_xlim())

    # set y axis scale
    func_max = np.max( gausshermite(vdat2[fitting].value/xscal, pars, order) )
    fit.set_ylim(-0.2*func_max, 1.3*func_max)
    eb.set_xlim(fit.get_xlim())
    eb.set_ylim(fit.get_ylim())

    fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    residuals.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=12)
    residuals.set_ylabel("Residual")

    #######################################
    xdat = wav[fitting]
    vdat = wave2doppler(xdat, w0)
    ydat = flux[fitting]
    yerr = err[fitting]

    hg = fig.add_subplot(5,1,4)
    hg.hist((ydat - gausshermite(vdat.value/xscal, pars, order)) / yerr, bins=np.arange(-5,5,0.25), normed=True, edgecolor='None')
    x_axis = np.arange(-5, 5, 0.001)
    hg.plot(x_axis, norm.pdf(x_axis,0,1), color='black', lw=2)

    #########################################################

    fs.plot(wav, flux, color='black', lw=1)
  
    fig.tight_layout()

    if plot_savefig is not None:
        fig = fig.savefig(os.path.join(save_dir,plot_savefig))

    plt.show()

    plt.close()

    return None

def fit_line(wav,
             flux,
             err,
             z=0.0,
             w0=6564.89*u.AA,
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=[6000,7000]*u.AA,
             red_shelf = [1580,1690]*u.AA,
             line_region=[-10000,10000]*(u.km/u.s),
             maskout=None,
             order=6,
             plot=True,
             plot_savefig='something.png',
             plot_title='',
             verbose=True,
             save_dir=None,
             bkgd_median=True):

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

    # index of the region we want to fit
    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])


    # fit power-law to continuum region
    # For civ we use median because more robust for small wavelength intervals. 
    # Ha we will fit to the data points since windows are much larger. 

    # index of region for continuum fit 
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    bkgdmod = PowerLawModel()
    bkgdpars = bkgdmod.make_params()
    bkgdpars['exponent'].value = 1.0
    bkgdpars['amplitude'].value = 1.0

    blue_inds = (wav > continuum_region[0][0]) & (wav < continuum_region[0][1])
    red_inds = (wav > continuum_region[1][0]) & (wav < continuum_region[1][1])   

    xdat_blue = wav[blue_inds]
    ydat_blue = flux[blue_inds]
    yerr_blue = err[blue_inds]
    vdat_blue = wave2doppler(xdat_blue, w0)

    xdat_red = wav[red_inds]
    ydat_red = flux[red_inds]
    yerr_red = err[red_inds]
    vdat_red = wave2doppler(xdat_red, w0)

    if maskout is not None:

        mask_blue = np.array([True] * len(xdat_blue))
        mask_red = np.array([True] * len(xdat_red))

        if maskout.unit == (u.km/u.s):
            
            for item in maskout:
                mask_blue[(vdat_blue > item[0]) & (vdat_blue < item[1])] = False
                mask_red[(vdat_red > item[0]) & (vdat_red < item[1])] = False

        elif maskout.unit == (u.AA):

            for item in maskout:
                mask_blue[(xdat_blue > (item[0] / (1.0 + z))) & (xdat_blue < (item[1] / (1.0 + z)))] = False
                mask_red[(xdat_red > (item[0] / (1.0 + z))) & (xdat_red < (item[1] / (1.0 + z)))] = False

        else:
            print "Units must be km/s or angstrom"

        xdat_blue = xdat_blue[mask_blue]
        ydat_blue = ydat_blue[mask_blue]
        yerr_blue = yerr_blue[mask_blue]
        vdat_blue = vdat_blue[mask_blue]
        xdat_red = xdat_red[mask_red]
        ydat_red = ydat_red[mask_red]
        yerr_red = yerr_red[mask_red]
        vdat_red = vdat_red[mask_red]


    if bkgd_median is True:

        xdat_cont = np.array( [np.median(xdat_blue.value), np.median(xdat_red.value)] )
        ydat_cont = np.array( [np.median(ydat_blue), np.median(ydat_red)] )

        out = minimize(resid,
                       bkgdpars,
                       args=(xdat_cont, bkgdmod, ydat_cont),
                       method='leastsq')

    if bkgd_median is False:

        xdat_cont = np.concatenate((xdat_blue, xdat_red))
        ydat_cont = np.concatenate((ydat_blue, ydat_red))
        yerr_cont = np.concatenate((yerr_blue, yerr_red))

        out = minimize(resid,
                       bkgdpars,
                       args=(xdat_cont.value, bkgdmod, ydat_cont, yerr_cont),
                       method='leastsq')

    if verbose:
        print fit_report(bkgdpars)


    # subtract continuum, define region for fitting
    xdat = wav[fitting]
    ydat = flux[fitting] - resid(bkgdpars, wav[fitting].value, bkgdmod)
    yerr = err[fitting]

    # Transform to doppler shift
    vdat = wave2doppler(xdat, w0)

    """
    Remember that this is to velocity shifted array

    Accepts units km/s or angstrom (observed frame)
    """

    # mask out redshelf from fit
    mask = np.array([True] * len(vdat))
    red_shelf_start = wave2doppler(red_shelf[0], w0)
    red_shelf_end = wave2doppler(red_shelf[1], w0)

    mask[(vdat > red_shelf_start) & (vdat < red_shelf_end)] = False

    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            for item in maskout:
                if verbose:
                    print 'Not fitting between {0} and {1}'.format(item[0], item[1])
                mask[(vdat > item[0]) & (vdat < item[1])] = False


        elif maskout.unit == (u.AA):

            for item in maskout:
                vlims = wave2doppler(item / (1.0 + z), w0) 
                if verbose:
                    print 'Not fitting between {0} ({1}) and {2} ({3})'.format(item[0], vlims[0], item[1], vlims[1])
                mask[(xdat > (item[0] / (1.0 + z))) & (xdat < (item[1] / (1.0 + z)))] = False

        else:
            print "Units must be km/s or angstrom"

    vdat = vdat[mask]
    ydat = ydat[mask]
    yerr = yerr[mask]

    # Calculate mean and variance
    p = ydat / np.sum(ydat)
    m = np.sum(vdat * p)
    v = np.sum(p * (vdat-m)**2)
    sd = np.sqrt(v)

    vs = np.linspace(vdat.min(), vdat.max(), 1000)

    pars = Parameters()

    pars.add('amp0', value = 1.0, min=0.0)
    pars.add('sig0', value = 1.0, min=0.1)
    pars.add('cen0', value = 0.0, min=vdat.min().value/sd.value, max=vdat.max().value/sd.value)


    if order > 0:
        pars.add('amp1', value = 1.0, min=0.0)
        pars.add('sig1', value = 1.0, min=0.1)
        pars.add('cen1', value = 0.0, min=vdat.min().value/sd.value, max=vdat.max().value/sd.value)

    if order > 1:
        pars.add('amp2', value = 1.0, min=0.0)
        pars.add('sig2', value = 1.0, min=0.1)
        pars.add('cen2', value = 0.0, min=vdat.min().value/sd.value, max=vdat.max().value/sd.value)

    if order > 2:
        pars.add('amp3', value = 1.0, min=0.0)
        pars.add('sig3', value = 1.0, min=0.1)
        pars.add('cen3', value = 0.0, min=vdat.min().value/sd.value, max=vdat.max().value/sd.value)

    if order > 3:
        pars.add('amp4', value = 1.0, min=0.0)
        pars.add('sig4', value = 1.0, min=0.1)
        pars.add('cen4', value = 0.0, min=vdat.min().value/sd.value, max=vdat.max().value/sd.value)

    if order > 4:
        pars.add('amp5', value = 1.0, min=0.0)
        pars.add('sig5', value = 1.0, min=0.1)
        pars.add('cen5', value = 0.0, min=vdat.min().value/sd.value, max=vdat.max().value/sd.value)

    if order > 5:
        pars.add('amp6', value = 1.0, min=0.0)
        pars.add('sig6', value = 1.0, min=0.1)
        pars.add('cen6', value = 0.0, min=vdat.min().value/sd.value, max=vdat.max().value/sd.value)

    out = minimize(gh_resid,
                   pars,
                   args=(vdat.value/sd.value, order, ydat, yerr),
                   method='leastsq')

    # with open('file.txt', 'w') as f:
    #     for wl, fl, m in zip(np.asarray(vtemp), ydattemp, gh_resid(pars, np.asarray(vtemp)/sd.value, order)):
    #         f.write('{0:20} {1:20} {2:20} \n'.format(wl, fl, m))


    if verbose:
        for key, value in pars.valuesdict().items():
            if 'cen' in key:
                print key, value * sd.value
            else:
                print key, value


    # Save results

    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        param_file = os.path.join(save_dir, 'my_params.txt')
        parfile = open(param_file, 'w')
        pars.dump(parfile)
        parfile.close()

        wav_file = os.path.join(save_dir, 'wav.txt')
        parfile = open(wav_file, 'wb')
        pickle.dump(wav, parfile, -1)
        parfile.close()

        flx_file = os.path.join(save_dir, 'flx.txt')
        parfile = open(flx_file, 'wb')
        pickle.dump(flux - resid(bkgdpars, wav.value, bkgdmod), parfile, -1)
        parfile.close()

        err_file = os.path.join(save_dir, 'err.txt')
        parfile = open(err_file, 'wb')
        pickle.dump(err, parfile, -1)
        parfile.close()

        sd_file = os.path.join(save_dir, 'sd.txt')
        parfile = open(sd_file, 'wb')
        pickle.dump(sd.value, parfile, -1)
        parfile.close()

    # Calculate FWHM of distribution

    integrand = lambda x: gh_resid(pars, x, order)
    # func_center = optimize.fmin(lambda x: -integrand(x) , 0)[0]
    # print 'Peak: {}'.format(func_center * sd.value)

    # # half_max = integrand(func_center) / 2.0

    # # root1 = optimize.brentq(lambda x: integrand(x) - half_max, vdat.min().value, func_center)
    # # root2 = optimize.brentq(lambda x: integrand(x) - half_max, func_center, vdat.max().value)

    # # print 'FWHM: {}'.format((root2 - root1)* sd.value)

    dv = 1.0
    # xs = np.arange(vdat.min().value, vdat.max().value, dv) / sd.value

    if line_region.unit == u.AA:
        line_region = wave2doppler(line_region, w0)
 
    xs = np.arange(line_region.value[0], line_region[1].value, dv) / sd.value

    norm = np.sum(integrand(xs) * dv)
    pdf = integrand(xs) / norm
    cdf = np.cumsum(pdf)
    cdf_r = np.cumsum(pdf[::-1])[::-1] # reverse cumsum


    func_center = xs[np.argmax(pdf)]
    # print 'Peak: {}'.format(func_center * sd.value)

    half_max = np.max(pdf) / 2.0

    i = 0
    while pdf[i] < half_max:
        i+=1

    root1 = xs[i]

    i = 0
    while pdf[-i] < half_max:
        i+=1

    root2 = xs[-i]

    # print 'FWHM: {}'.format((root2 - root1)* sd.value)

    md = xs[np.argmin( np.abs( cdf - 0.5))]
    # print 'Median: {}'.format(md * sd.value)


    # # Not sure this would work if median far from zero but probably would never happen.
    p99 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.005))] - xs[np.argmin(np.abs(cdf - 0.005))])* sd.value
    p95 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.025))] - xs[np.argmin(np.abs(cdf - 0.025))])* sd.value
    p90 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.05))] - xs[np.argmin(np.abs(cdf - 0.05))])* sd.value
    p80 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.1))] - xs[np.argmin(np.abs(cdf - 0.1))])* sd.value
    p60 = np.abs(xs[np.argmin(np.abs(cdf_r - 0.2))] - xs[np.argmin(np.abs(cdf - 0.2))])* sd.value

    # # print '99%: {}'.format(p99)
    # # print '95%: {}'.format(p95)
    # # print '90%: {}'.format(p90)
    # # print '80%: {}'.format(p80)
    # # print '60%: {}'.format(p60)

    m = np.sum(xs * pdf * dv)
    v = np.sum( (xs-m)**2 * pdf * dv )
    sigma = np.sqrt(v)

    # xs_new = np.arange(line_region.value[0], line_region[1].value, dv) / sd.value
    # norm = np.sum(integrand(xs_new) * dv)
    # pdf = integrand(xs_new) / norm
    # m2 = np.sum(xs_new * pdf * dv)
    # v2 = np.sum( (xs_new-m2)**2 * pdf * dv )
    # sigma2 = np.sqrt(v2)

    # xs_new = np.arange(md - 10000.0, md + 10000, dv) / sd.value
    # norm = np.sum(integrand(xs_new) * dv)
    # pdf = integrand(xs_new) / norm
    # m3= np.sum(xs_new * pdf * dv)
    # v3 = np.sum( (xs_new-m3)**2 * pdf * dv )
    # sigma3 = np.sqrt(v3)

    # print 'Mean: {}'.format(m * sd.value)

    # """
    # This is working, but for Lorentzian the second moment is not defined.
    # It dies off much less quickly than the Gaussian so the sigma is much
    # larger. I therefore need to think of the range in which I calcualte
    # sigma
    # """

    print plot_title, (root2 - root1)* sd.value, sigma * sd.value, p99, p95, p90, p80, p60

    # print '{0} {1:.2f} {2:.2f} {3:.2f} {4} {5} {6} {7} {8} {9:.2f} {10:.2f}'.format(plot_title,
    #                                                                                 func_center * sd.value,
    #                                                                                 (root2 - root1)* sd.value,
    #                                                                                 md * sd.value,
    #                                                                                 p99,
    #                                                                                 p95,
    #                                                                                 p90,
    #                                                                                 p80,
    #                                                                                 p60,
    #                                                                                 m*sd.value,
    #                                                                                 sigma*sd.value)

    # print 'peak_civ = {0:.2f}*(u.km/u.s),'.format(func_center * sd.value)
    # print 'fwhm_civ = {0:.2f}*(u.km/u.s),'.format((root2 - root1) * sd.value)
    # print 'median_civ = {0:.2f}*(u.km/u.s),'.format(md * sd.value)
    # print 'sigma_civ = {0:.2f}*(u.km/u.s),'.format(sigma * sd.value)
    # print 'chired_civ = {0:.2f},'.format(out.redchi)


    # # Equivalent width
    # xs = np.arange(vdat.min().value, vdat.max().value, dv) / sd.value

    # flux_line = integrand(xs)
    # xs_wav = doppler2wave(xs*sd,w0)
    # flux_bkgd = bkgdmod.eval(params=bkgdpars, x=xs_wav.value)

    # f = (flux_line + flux_bkgd) / flux_bkgd

    # eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
    # print 'eqw_civ = {0:.2f},'.format(np.nansum(eqw))

    # """
    # Calculate S/N ratio per resolution element
    # Need to know what SDSS resolution is
    # """

    # vdat = wave2doppler(wav, w0)
    # vmin = md - 10000.0
    # vmax = md + 10000.0

    # i = np.argmin(np.abs(vdat.value - vmin))
    # j = np.argmin(np.abs(vdat.value - vmax))

    # fl = flux[i:j]
    # er = err[i:j]
    # w = wav[i:j]

    # good = (er > 0) & ~np.isnan(fl)
    # if len(good.nonzero()[0]) == 0:
    #     print('No good data in this range!')

    # fl = fl[good]
    # er = er[good]
    # w = w[good]
    # v = wave2doppler(w,w0)

    # snr = fl / er

    # # fwhm resolution in units of pixels
    # wres, res = np.genfromtxt('/data/lc585/SDSS/resolution.dat', unpack=True)
    # wres = wres / (1.0 + z)
    # vres = wave2doppler(wres*u.AA, w0)

    # f = interp1d(vres, res)
    # print 'snr_civ = {0:.2f},'.format(np.mean( np.sqrt(f(v)) * snr))

    if plot:
        plot_fit(wav=wav,
                 flux = flux - resid(bkgdpars, wav.value, bkgdmod),
                 err=err,
                 xscal = sd.value,
                 pars = pars,
                 order = order,
                 plot_savefig = plot_savefig,
                 save_dir = save_dir,
                 maskout = maskout,
                 red_shelf = red_shelf,
                 z=z,
                 w0=w0,
                 continuum_region=continuum_region,
                 fitting_region=fitting_region,
                 plot_region=plot_region,
                 plot_title=plot_title,
                 line_region=line_region)

    return None


if __name__ == '__main__':

    from get_spectra import get_boss_dr12_spec
    import matplotlib.pyplot as plt

    wav, dw, flux, err = get_boss_dr12_spec('SDSSJ123611.21+112921.6')

    fit_line(wav,
             flux,
             err,
             z=2.155473,
             w0=np.mean([1548.202,1550.774])*u.AA,
             continuum_region=[[1445.,1465.]*u.AA,[1700.,1705.]*u.AA],
             fitting_region=[1460.,1580.]*u.AA,
             plot_region=[1440,1720]*u.AA,
             maskout=[[-15610,-11900],[-3360,-2145],[135,485],[770,1024]]*(u.km/u.s),
             plot=True,
             save_dir='/data/lc585/WHT_20150331/NewLineFits/SDSSJ123611.21+112921.6/gausshermite/CIV')
    plt.show()

