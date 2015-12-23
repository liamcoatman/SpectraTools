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

def resid(p=None, x=None, model=None, data=None, sigma=None, **kwargs):

    mod = model.eval(params=p, x=x, **kwargs)
    
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


def fit_line_gh_errors(wav,
                       flux,
                       err,
                       z=0.0,
                       w0=np.mean([1548.202,1550.774])*u.AA,
                       continuum_region=[[1445.,1465.]*u.AA,[1700.,1705.]*u.AA],
                       fitting_region=[1500.,1600.]*u.AA,
                       plot_region=[1450.,1650.]*u.AA,
                       line_region=[1500.,1600.]*u.AA,
                       maskout=None,
                       order=6,
                       verbose=True,
                       plot=True,
                       save_dir=None,
                       plot_title='',
                       plot_savefig='',
                       bkgd_median=True,
                       fitting_method='leastsq',
                       mask_negflux=False):

    """
    Velocity shift added to doppler shift to change zero point (can do if HW10
    redshift does not agree with Halpha centroid)

    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    """
    with open(os.path.join('/data/lc585/WHT_20150331/fit_errors_3',plot_title+'_CIV.dat'), 'w') as f:
        f.write('Centroid FWHM Median Mean Sigma EQW \n')

    # Normalise spectrum
    spec_norm = 1.0 / np.median(flux)
    flux = flux * spec_norm 
    err = err * spec_norm  

    n_samples = 5000
    n_elements = len(wav)

    err[err <= 0] = np.median(err)

    flux_array = np.asarray(np.random.normal(flux,
                                             err,
                                             size=(n_samples, n_elements)),
                            dtype=np.float32)


    # flux_array = np.repeat(flux.reshape(1, n_elements), n_samples, axis=0)

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA

    # Check if continuum is given in wavelength or doppler units
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    # index of the region we want to fit
    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])

    # Fit to median
    blue_inds = (wav > continuum_region[0][0]) & (wav < continuum_region[0][1])
    red_inds = (wav > continuum_region[1][0]) & (wav < continuum_region[1][1])

    blue_wav, red_wav = wav[blue_inds], wav[red_inds]
    blue_vel, red_vel = wave2doppler(blue_wav, w0), wave2doppler(red_wav, w0)
    blue_err, red_err = err[blue_inds], err[red_inds]
    blue_flux_array = flux_array[:,blue_inds]
    red_flux_array =  flux_array[:,red_inds]

    if maskout is not None:

        mask_blue = np.array([True] * len(blue_wav))
        mask_red = np.array([True] * len(red_wav))

        if maskout.unit == (u.km/u.s):
            
            for item in maskout:
                mask_blue[(blue_vel > item[0]) & (blue_vel < item[1])] = False
                mask_red[(red_vel > item[0]) & (red_vel < item[1])] = False

        elif maskout.unit == (u.AA):

            for item in maskout:
                mask_blue[(blue_wav > (item[0] / (1.0 + z))) & (blue_wav < (item[1] / (1.0 + z)))] = False
                mask_red[(red_wav > (item[0] / (1.0 + z))) & (red_wav < (item[1] / (1.0 + z)))] = False

        else:
            print "Units must be km/s or angstrom"

        blue_wav = blue_wav[mask_blue]
        red_wav = red_wav[mask_red]
        blue_err = blue_err[mask_blue]
        red_err = red_err[mask_red]
        blue_flux_array = blue_flux_array[:,mask_blue]
        red_flux_array = red_flux_array[:,mask_red]


    if bkgd_median is True:

        xdat_bkgd = np.array( [blue_wav.mean().value,
                               red_wav.mean().value] )
        ydat_bkgd = np.array( [np.median(blue_flux_array,axis=1),
                               np.median(red_flux_array,axis=1)] ).T

    if bkgd_median is False: 

        xdat_bkgd = np.concatenate((blue_wav.value, red_wav.value))
        ydat_bkgd = np.concatenate((blue_flux_array, red_flux_array), axis=1)
        yerr_bkgd = np.concatenate((blue_err, red_err))        

    

    bkgdmod = PowerLawModel()
    bkgdpars = bkgdmod.make_params()
    bkgdpars['exponent'].value = 1.0
    bkgdpars['amplitude'].value = 1.0 / 5000.0  

    if plot:

        fig = plt.figure(figsize=(6,10))

        fit = fig.add_subplot(3,1,1)
        fit.set_xticklabels( () )
        residuals = fig.add_subplot(3,1,2)

        blue_cont = wave2doppler(continuum_region[0], w0)
        red_cont = wave2doppler(continuum_region[1], w0)

        fit.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
        fit.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

        residuals.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
        residuals.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

        # Mark fitting region
        fr = wave2doppler(fitting_region, w0) 

        # Mask out regions
        xdat_fit = wav[fitting]
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


        vdat1_masking = ma.array(vdat_masking.value)
        vdat1_masking[mask] = ma.masked

        for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
            fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), color='moccasin', alpha=0.4)
            residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), color='moccasin', alpha=0.4)

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

        if bkgd_median is True:
           

            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_bkgd, 'model':bkgdmod, 'data':ydat_bkgd[k]},
                           method='leastsq')

        if bkgd_median is False:

            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_bkgd, 'model':bkgdmod, 'data':ydat_bkgd[k,:], 'sigma':yerr_bkgd},
                           method='leastsq')    

        
        if verbose:
            print fit_report(bkgdpars)


        # subtract continuum, define region for fitting
        ydat_fit = flux_array[k,fitting] - resid(p=bkgdpars, x=wav[fitting].value, model=bkgdmod)

        xdat_fit = wav[fitting]
        yerr_fit = err[fitting]
        vdat_fit = wave2doppler(xdat_fit, w0)


        # Remove negative flux values which can mess up fit 
        if mask_negflux:
    
            posflux = (ydat_fit > 0.0) 
            
            ydat_fit = ydat_fit[posflux]
            yerr_fit = yerr_fit[posflux]
            xdat_fit = xdat_fit[posflux]
            vdat_fit = vdat_fit[posflux]


        """
        Remember that this is to velocity shifted array

        Accepts units km/s or angstrom (observed frame)
        """

        # mask out redshelf from fit
        mask = np.array([True] * len(vdat_fit))


        if maskout is not None:

            if maskout.unit == (u.km/u.s):

                for item in maskout:
                    if verbose:
                        print 'Not fitting between {0} and {1}'.format(item[0], item[1])
                    mask[(vdat_fit > item[0]) & (vdat_fit < item[1])] = False


            elif maskout.unit == (u.AA):

                for item in maskout:
                    vlims = wave2doppler(item / (1.0 + z), w0)
                    if verbose:
                        print 'Not fitting between {0} ({1}) and {2} ({3})'.format(item[0], vlims[0], item[1], vlims[1])
                    mask[(xdat_fit > (item[0] / (1.0 + z))) & (xdat_fit < (item[1] / (1.0 + z)))] = False

            else:
                print "Units must be km/s or angstrom"

        vdat_fit = vdat_fit[mask]
        ydat_fit = ydat_fit[mask]
        yerr_fit = yerr_fit[mask]

        # Calculate mean and variance
        p = ydat_fit / np.sum(ydat_fit)
        m = np.sum(vdat_fit * p)
        v = np.sum(p * (vdat_fit-m)**2)
        sd = np.sqrt(v)

        pars = Parameters()

        pars.add('amp0', value = 1.0, min=0.0)
        pars.add('sig0', value = 1.0, min=0.1)
        pars.add('cen0',
                 value = 0.0,
                 min=vdat_fit.min().value/sd.value,
                 max=vdat_fit.max().value/sd.value)


        if order > 0:
            pars.add('amp1', value = 1.0, min=0.0)
            pars.add('sig1', value = 1.0, min=0.1)
            pars.add('cen1',
                     value = 0.0,
                     min=vdat_fit.min().value/sd.value,
                     max=vdat_fit.max().value/sd.value)

        if order > 1:
            pars.add('amp2', value = 1.0, min=0.0)
            pars.add('sig2', value = 1.0, min=0.1)
            pars.add('cen2',
                     value = 0.0,
                     min=vdat_fit.min().value/sd.value,
                     max=vdat_fit.max().value/sd.value)

        if order > 2:
            pars.add('amp3', value = 1.0, min=0.0)
            pars.add('sig3', value = 1.0, min=0.1)
            pars.add('cen3',
                     value = 0.0,
                     min=vdat_fit.min().value/sd.value,
                     max=vdat_fit.max().value/sd.value)

        if order > 3:
            pars.add('amp4', value = 1.0, min=0.0)
            pars.add('sig4', value = 1.0, min=0.1)
            pars.add('cen4',
                     value = 0.0,
                     min=vdat_fit.min().value/sd.value,
                     max=vdat_fit.max().value/sd.value)

        if order > 4:
            pars.add('amp5', value = 1.0, min=0.0)
            pars.add('sig5', value = 1.0, min=0.1)
            pars.add('cen5',
                     value = 0.0,
                     min=vdat_fit.min().value/sd.value,
                     max=vdat_fit.max().value/sd.value)

        if order > 5:
            pars.add('amp6', value = 1.0, min=0.0)
            pars.add('sig6', value = 1.0, min=0.1)
            pars.add('cen6',
                      value = 0.0,
                      min=vdat_fit.min().value/sd.value,
                      max=vdat_fit.max().value/sd.value)

        out = minimize(gh_resid,
                       pars,
                       args=(vdat_fit.value/sd.value, order, ydat_fit, yerr_fit),
                       method='leastsq')

        if verbose:
            for key, value in pars.valuesdict().items():
                print key, value


        # Calculate FWHM of distribution
        integrand = lambda x: gh_resid(pars, x, order)
        
        dv = 1.0
    
        if line_region.unit == u.AA:
            line_region = wave2doppler(line_region, w0)
     
        xs = np.arange(line_region.value[0], line_region[1].value, dv) / sd.value
    
        norm = np.sum(integrand(xs) * dv)
        pdf = integrand(xs) / norm
        cdf = np.cumsum(pdf)
        cdf_r = np.cumsum(pdf[::-1])[::-1] # reverse cumsum
    
        func_center = xs[np.argmax(pdf)]
       
        half_max = np.max(pdf) / 2.0
    
        i = 0
        while pdf[i] < half_max:
            i+=1
    
        root1 = xs[i]
    
        i = 0
        while pdf[-i] < half_max:
            i+=1
    
        root2 = xs[-i]
    
        md = xs[np.argmin( np.abs( cdf - 0.5))]
    
        m = np.sum(xs * pdf * dv)
        v = np.sum( (xs-m)**2 * pdf * dv )
        sigma = np.sqrt(v)
        print sigma 

        flux_line = integrand(xs)
        xs_wav = doppler2wave(xs*sd,w0)
        flux_bkgd = bkgdmod.eval(params=bkgdpars, x=xs_wav.value)
    
        f = (flux_line + flux_bkgd) / flux_bkgd
    
        eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
        eqw = np.nansum(eqw)


        with open(os.path.join('/data/lc585/WHT_20150331/fit_errors_3',plot_title+'_CIV.dat'), 'a') as f:
            f.write('{0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.4f} \n'.format(func_center * sd.value,
                                                                                (root2 - root1)* sd.value,
                                                                                md * sd.value,
                                                                                m*sd.value,
                                                                                sigma*sd.value,
                                                                                eqw))

        print k

        if plot:

            xdat_plot = wav[plot_region_inds]
            vdat_plot = wave2doppler(xdat_plot, w0)
            ydat_plot = flux_array[k,plot_region_inds] - resid(bkgdpars, wav[plot_region_inds].value, bkgdmod)
            yerr_plot = err[plot_region_inds]

            if 'pts1' in locals():
                pts1.set_data((vdat_plot.value, ydat_plot))
            else:
                pts1, = fit.plot(vdat_plot.value, ydat_plot)

            vs = np.linspace(vdat_fit.min().value, vdat_fit.max().value, 1000)

            flux_mod = gausshermite(vs/sd.value, pars, order)

            if 'line' in locals():
                line.set_data((vs, flux_mod))
            else:
                line, = fit.plot(vs, flux_mod, color='black')


            if 'pts2' in locals():
                pts2.set_data((vdat_plot.value,ydat_plot - gausshermite(vdat_plot.value/sd.value, pars, order)))
            else:
                pts2, = residuals.plot(vdat_plot.value,
                                       ydat_plot- gausshermite(vdat_plot.value/sd.value, pars, order))


            fig.set_tight_layout(True)

            # plt.show()
            plt.pause(0.1)

    plt.close()

    return None

