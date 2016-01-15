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
from lmfit import minimize, Parameters, fit_report, Model
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import os
import cPickle as pickle
from scipy.stats import norm
from scipy.ndimage.filters import median_filter
from palettable.colorbrewer.qualitative import Set2_5
from time import gmtime, strftime
from astropy.cosmology import WMAP9 as cosmoWMAP
import math
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import interp1d 
from os.path import expanduser
from barak import spec 
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Simple mouse click function to s tore coordinates
def onclick(event):

    global ix
    
    ix = event.xdata

    coords.append(ix)

    if len(coords) % 2 == 0:
        print '[{0:.0f}, {1:.0f}]'.format(coords[-2], coords[-1])  
        # fig.canvas.mpl_disconnect(cid)
        
    return None 

def onclick2(event):

    global ix
    
    ix = event.xdata

    coords.append(ix)

    if len(coords) % 4 == 0:
        print '[[{0:.0f}, {1:.0f}]*u.AA,[{2:.0f}, {3:.0f}]*u.AA]'.format(coords[-4], coords[-3], coords[-2], coords[-1])  
        # fig.canvas.mpl_disconnect(cid)
        
    return None 

class line_props(object):
    
    def __init__(self, 
                 name, 
                 peak,
                 fwhm ,
                 median,
                 sigma,
                 redchi,
                 eqw):

        self.name = name
        self.peak = peak 
        self.fwhm = fwhm
        self.median = median 
        self.sigma = sigma 
        self.redchi = redchi
        self.eqw = eqw 
      
def resid(p=None, 
          x=None, 
          model=None, 
          data=None, 
          sigma=None, 
          **kwargs):

    # print p['amplitude'].value,    p['exponent'].value,    p['fe_norm'].value,    p['fe_sd'].value,    p['fe_shift'].value

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



def wave2doppler(w, w0):

    """
    function uses the Doppler equivalency between wavelength and velocity
    """
    w0_equiv = u.doppler_optical(w0)
    w_equiv = w.to(u.km/u.s, equivalencies=w0_equiv)

    return w_equiv

def doppler2wave(v, w0):

    """
    function uses the Doppler equivalency between wavelength and velocity
    """
    w0_equiv = u.doppler_optical(w0)
    w_equiv = v.to(u.AA, equivalencies=w0_equiv)

    return w_equiv

def PseudoContinuum(amplitude, 
                    exponent, 
                    fe_norm,
                    fe_sd,
                    fe_shift,
                    x,  
                    sp_fe):
 
 
    gauss = Gaussian1DKernel(stddev=fe_sd)

    fe_flux = convolve(sp_fe.fl, gauss)
    
    f = interp1d(sp_fe.wa - fe_shift, 
                 fe_flux, 
                 bounds_error=False, 
                 fill_value=0.0)

    fe_flux = f(x)   

    # so paramter is around 1
    amp = amplitude * 5000.0**(-exponent)

    return fe_norm * fe_flux + amp*x**exponent

def PLModel(amplitude,
            exponent,
            x):

    # should probably change this to 1350 when fitting CIV
    amp = amplitude * 5000.0**(-exponent) 

    return amp*x**exponent 

def plot_fit(wav=None,
             flux=None,
             err=None,
             pars=None,
             mod=None,
             out=None,
             plot_savefig=None,
             plot_title='',
             save_dir=None,
             maskout=None,
             z=0.0,
             w0=6564.89*u.AA,
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=None,
             line_region=[-10000,10000]*(u.km/u.s),
             mask_negflux = True,
             fit_model='MultiGauss',
             hb_narrow=True,
             verbose=True):


    if plot_region.unit == (u.km/u.s):
        plot_region = doppler2wave(plot_region, w0)

    # plotting region
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
    
    eb = fig.add_subplot(5,1,3)
    residuals.set_xticklabels( () )

    fs = fig.add_subplot(5,1,5)

    fit.scatter(vdat.value, ydat, edgecolor='None', s=15, alpha=0.9, facecolor='black')

    # Mark continuum fitting region
    # Doesn't make sense to transform to wave and then back to velocity but I'm being lazy.
    # Check if continuum is given in wavelength or doppler units
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    # Region where equivalent width etc. calculated.
    integrand = lambda x: mod.eval(params=pars, x=np.array(x))
    func_max = np.max(integrand(vdat))
    
    eb.axvline(line_region[0].value, color='black', linestyle='--')
    eb.axvline(line_region[1].value, color='black', linestyle='--')

    fit.axvline(line_region[0].value, color='black', linestyle='--')
    fit.axvline(line_region[1].value, color='black', linestyle='--')

    fs.axvline(doppler2wave(line_region[0], w0).value, color='black', linestyle='--')
    fs.axvline(doppler2wave(line_region[1], w0).value, color='black', linestyle='--')

    fit.axhline(0, color='black', linestyle='--')
    fs.axhline(0, color='black', linestyle='--')
    eb.axhline(0, color='black', linestyle='--')
    
    # Mark fitting region
    fr = wave2doppler(fitting_region, w0)

    # set y axis scale
    fit.set_ylim(-0.3*func_max, 1.3*func_max)

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
                xmin = item.value[0]
                xmax = item.value[1] 
                mask = mask | ((xdat_masking.value > xmin) & (xdat_masking.value < xmax))

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
                xmin = item.value[0] 
                xmax = item.value[1] 
                mask = mask | ((xdat_masking.value > xmin) & (xdat_masking.value < xmax))


    vdat1_masking = ma.array(vdat_masking.value)
    vdat1_masking[mask] = ma.masked 

    for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
        fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')
        residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')
        eb.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')  
        fs.axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, alpha=0.4, color='powderblue')    

    line, = fit.plot(np.sort(vdat.value), resid(pars, np.sort(vdat.value), mod), color='black', lw=2)

    plotting_limits = wave2doppler(plot_region, w0)
    fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)

    # residuals.errorbar(vdat.value, (ydat - resid(pars, vdat.value, mod)) / yerr, yerr=1, linestyle='', alpha=0.4)
    # residuals.plot(vdat.value, (ydat - resid(pars, vdat.value, mod)) / yerr, color='black', lw=1)
    residuals.scatter(vdat.value, 
                     (ydat - resid(pars, vdat.value, mod)) / yerr, 
                     alpha=0.9, 
                     edgecolor='None', 
                     s=15, 
                     facecolor='black')
    # residuals.plot(vdat.value, median_filter((ydat - resid(pars, vdat.value, mod)) / yerr, 3.0), color='black')

    residuals.axhline(0.0, color='black', linestyle='--')

    residuals.set_ylim(-8,8)
    residuals.set_xlim(fit.get_xlim())

    # plot model components

    if fit_model == 'MultiGauss':
    
        i, j = 0, 0
        for key, value in pars.valuesdict().iteritems():
            if key.startswith('g'):
                i += 1
            if key.startswith('l'):
                j += 1
    
        ngaussians, nlorentzians = int(float(i) / 4.0), int(float(j) / 4.0)
    
        if ngaussians > 1:
    
            for i in range(ngaussians):
                comp_mod = GaussianModel()
                comp_p = comp_mod.make_params()
    
                for key, v in comp_p.valuesdict().iteritems():
                    comp_p[key].value = pars['g' + str(i) + '_' + key].value
    
                fit.plot( np.sort(vdat.value), comp_mod.eval(comp_p, x=np.sort(vdat.value)) )
    
        if nlorentzians > 1:
    
            for i in range(nlorentzians):
                comp_mod = LorentzianModel()
                comp_p = comp_mod.make_params()
    
                for key, v in comp_p.valuesdict().iteritems():
                    comp_p[key].value = pars['l' + str(i) + '_' + key].value
    
                fit.plot( np.sort(vdat.value), comp_mod.eval(comp_p, x=np.sort(vdat.value)) )
    

    if fit_model == 'Hb': 
 


        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_5007_n_center'].value
        p['sigma'].value = pars['oiii_5007_n_sigma'].value
        p['amplitude'].value = pars['oiii_5007_n_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_4959_n_center'].value
        p['sigma'].value = pars['oiii_4959_n_sigma'].value
        p['amplitude'].value = pars['oiii_4959_n_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='blue',
                 linestyle='--')        

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_5007_b_center'].value
        p['sigma'].value = pars['oiii_5007_b_sigma'].value
        p['amplitude'].value = pars['oiii_5007_b_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='-')       

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_4959_b_center'].value
        p['sigma'].value = pars['oiii_4959_b_sigma'].value
        p['amplitude'].value = pars['oiii_4959_b_amplitude'].value   

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='blue',
                 linestyle='-')             


        i = 0
        for key, value in pars.valuesdict().iteritems():
            if key.startswith('hb_b'):
                i += 1
                
        nGaussians = int(float(i) / 4.0) 

        for i in range(nGaussians):

            g = GaussianModel()
            p = g.make_params()

            p['center'].value = pars['hb_b_{}_center'.format(i)].value
            p['sigma'].value = pars['hb_b_{}_sigma'.format(i)].value
            p['amplitude'].value = pars['hb_b_{}_amplitude'.format(i)].value  
    
            fit.plot(np.sort(vdat.value), 
                     g.eval(p, x=np.sort(vdat.value)),
                     c='orange')  

        mod_broad_hb = ConstantModel()

        for i in range(nGaussians):
            mod_broad_hb += GaussianModel(prefix='hb_b_{}_'.format(i))  

        pars_broad_hb = mod_broad_hb.make_params()
       
        pars_broad_hb['c'].value = 0.0
       
        for key, value in pars.valuesdict().iteritems():
            if key.startswith('hb_b_'):
                pars_broad_hb[key].value = value   
          
        
        fit.plot(np.sort(vdat.value), 
                 mod_broad_hb.eval(pars_broad_hb, x=np.sort(vdat.value)),
                 c='black',
                 linestyle='--',
                 lw=2)                          
    
        if hb_narrow is True: 

            g = GaussianModel()
            p = g.make_params()
    
            p['center'] = pars['hb_n_center']
            p['sigma'] = pars['hb_n_sigma']
            p['amplitude'] = pars['hb_n_amplitude']   
    
            fit.plot(np.sort(vdat.value), 
                     g.eval(p, x=np.sort(vdat.value)),
                     c='orange',
                     linestyle='--')                    

    if fit_model == 'Ha': 


        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['ha_n_center'].value
        p['sigma'].value = pars['ha_n_sigma'].value
        p['amplitude'].value = pars['ha_n_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['nii_6548_n_center'].value
        p['sigma'].value = pars['nii_6548_n_sigma'].value
        p['amplitude'].value = pars['nii_6548_n_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['nii_6584_n_center'].value
        p['sigma'].value = pars['nii_6584_n_sigma'].value
        p['amplitude'].value = pars['nii_6584_n_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['sii_6717_n_center'].value
        p['sigma'].value = pars['sii_6717_n_sigma'].value
        p['amplitude'].value = pars['sii_6717_n_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')      
                 
        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['sii_6731_n_center'].value
        p['sigma'].value = pars['sii_6731_n_sigma'].value
        p['amplitude'].value = pars['sii_6731_n_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--') 

        
        i = 0
        for key, value in pars.valuesdict().iteritems():
            if key.startswith('ha_b'):
                i += 1
                
        nGaussians = int(float(i) / 4.0) 

        for i in range(nGaussians):

            g = GaussianModel()
            p = g.make_params()

            p['center'].value = pars['ha_b_{}_center'.format(i)].value
            p['sigma'].value = pars['ha_b_{}_sigma'.format(i)].value
            p['amplitude'].value = pars['ha_b_{}_amplitude'.format(i)].value  
    
            fit.plot(np.sort(vdat.value), 
                     g.eval(p, x=np.sort(vdat.value)),
                     c='orange')  

        mod_broad_ha = ConstantModel()

        for i in range(nGaussians):
            mod_broad_ha += GaussianModel(prefix='ha_b_{}_'.format(i))  

        pars_broad_ha = mod_broad_ha.make_params()
       
        pars_broad_ha['c'].value = 0.0
       
        for key, value in pars.valuesdict().iteritems():
            if key.startswith('ha_b_'):
                pars_broad_ha[key].value = value   
          
        
        fit.plot(np.sort(vdat.value), 
                 mod_broad_ha.eval(pars_broad_ha, x=np.sort(vdat.value)),
                 c='black',
                 linestyle='--',
                 lw=2)                          
    
      
    eb.errorbar(vdat.value, ydat, yerr=yerr, linestyle='', alpha=0.5, color='grey')
    eb.plot(np.sort(vdat.value), resid(pars, np.sort(vdat.value), mod), color='black', lw=2)
    eb.set_xlim(fit.get_xlim())
    eb.set_ylim(fit.get_ylim())

    fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    eb.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=10)
    eb.set_ylabel(r'F$_\lambda$', fontsize=12)
    residuals.set_ylabel("Residual")

    #######################################

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])
    xdat = wav[fitting]
    vdat = wave2doppler(xdat, w0)
    ydat = flux[fitting]
    yerr = err[fitting]
    hg = fig.add_subplot(5,1,4)
    hg.hist((ydat - resid(pars, vdat.value, mod)) / yerr, bins=np.arange(-5,5,0.25), normed=True, edgecolor='None', facecolor='lightgrey')
    x_axis = np.arange(-5, 5, 0.001)
    hg.plot(x_axis, norm.pdf(x_axis,0,1), color='black', lw=2)

    #########################################

    fs.plot(wav, median_filter(flux,5.0), color='black', lw=1)
    fs.set_xlim(wav.min().value, wav.max().value)
    fs.set_ylim(-1*func_max, 2*func_max)


    ########################################################

    """
    If any flux values are negative could cause problems in fit.
    Highlight these by coloring points 
    """
    if mask_negflux:

        xdat = wav[fitting]
        vdat = wave2doppler(xdat, w0)
        ydat = flux[fitting]
        yerr = err[fitting]
    
        bad_points = (ydat < 0.0)
    
        xdat_bad = xdat[bad_points]
        vdat_bad = vdat[bad_points]
        ydat_bad = ydat[bad_points]
        yerr_bad = yerr[bad_points]
    
        fit.scatter(vdat_bad, ydat_bad, color='darkred', s=40, marker='x')
        residuals.scatter(vdat_bad, ydat_bad, color='darkred', s=40, marker='x')
        eb.scatter(vdat_bad, ydat_bad, color='darkred', s=40, marker='x')
        fs.scatter(xdat_bad, ydat_bad, color='darkred', s=40, marker='x')


    fig.tight_layout()

    # Call click func

    if verbose:

        global coords
        coords = [] 
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

    if plot_savefig is not None:
        fig.savefig(os.path.join(save_dir, plot_savefig))

    if verbose:

        plt.show(1)
        plt.close()

    return None

def fit_line(wav,
             dw,
             flux,
             err,
             z=0.0,
             w0=6564.89*u.AA,
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=[6000,7000]*u.AA,
             line_region=[6400,6800]*u.AA,
             nGaussians=0,
             nLorentzians=1,
             maskout=None,
             verbose=True,
             plot=True,
             plot_savefig='something.png',
             plot_title='',
             save_dir=None,
             bkgd_median=True,
             fitting_method='leastsq',
             mask_negflux = True,
             mono_lum_wav = 5100 * u.AA,
             fit_model='MultiGauss',
             subtract_fe=False,
             fe_FWHM=50*u.AA,
             fe_FWHM_vary=False,
             hb_narrow=True,
             narrow_fwhm=None):


    """
    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    """


    home_dir = expanduser("~")

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA
    dw = dw / (1.0 + z)

    # Normalise spectrum
    spec_norm = 1.0 / np.median(flux)
    flux = flux * spec_norm 
    err = err * spec_norm 
 
    

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
                mask_blue[(xdat_blue > item[0]) & (xdat_blue < item[1])] = False
                mask_red[(xdat_red > item[0]) & (xdat_red < item[1] )] = False

        else:
            print "Units must be km/s or angstrom"

        if mask_blue.size > 0:

            xdat_blue = xdat_blue[mask_blue]
            ydat_blue = ydat_blue[mask_blue]
            yerr_blue = yerr_blue[mask_blue]
            vdat_blue = vdat_blue[mask_blue]

        if mask_red.size > 0:

            xdat_red = xdat_red[mask_red]
            ydat_red = ydat_red[mask_red]
            yerr_red = yerr_red[mask_red]
            vdat_red = vdat_red[mask_red]


    if bkgd_median is True:

        xdat_cont = np.array( [np.median(xdat_blue.value), np.median(xdat_red.value)] )
        ydat_cont = np.array( [np.median(ydat_blue), np.median(ydat_red)] )

        if subtract_fe is True:
            print 'Cant fit iron if bkgd_median is True' 

        elif subtract_fe is False:

            bkgdmod = Model(PLModel, 
                            param_names=['amplitude','exponent'], 
                            independent_vars=['x']) 

            bkgdpars = bkgdmod.make_params()
            bkgdpars['exponent'].value = 1.0
            bkgdpars['amplitude'].value = 1.0 

            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_cont, 
                                'model':bkgdmod, 
                                'data':ydat_cont},
                           method='leastsq')

            if verbose:
                print out.message  
                print fit_report(bkgdpars)
    
    if bkgd_median is False:

        # Messes up fit if have negative/zero weights 
        good_pix = (yerr_blue > 0.0) & (~np.isnan(yerr_blue))

        xdat_blue = xdat_blue[good_pix]
        ydat_blue = ydat_blue[good_pix]
        yerr_blue = yerr_blue[good_pix]

        good_pix = (yerr_red > 0.0) & (~np.isnan(yerr_red))

        xdat_red = xdat_red[good_pix]
        ydat_red = ydat_red[good_pix]
        yerr_red = yerr_red[good_pix]

        if subtract_fe is True:

            ################################################################################
            """
            Optical FeII template
        
            Broaden by convolution with Gaussian using astropy 
            Add to power-law with normalisation
            Allow shifting as well?
        
            Shen does all fits in logarithmic wavelength space. 
            Does this make a difference? 
        
            Convolve with width of Balmer line - might have to do iteratively 
        
            Either use FWHM from fit to Ha or leave convolution as a free paramter 
            See how this compares. 
            """

            fname = os.path.join(home_dir,'SpectraTools/irontemplate.dat')
            fe_wav, fe_flux = np.genfromtxt(fname, unpack=True)
        
            fe_flux = fe_flux / np.median(fe_flux)
    
            # At least for FIRE, the spectrum has a resolution 12.517 km/s
            # Fe template has resolution 106.2 km/s 
            # We need to rebin to make these the same   

            sp_fe = spec.Spectrum(wa=10**fe_wav, fl=fe_flux)

            bkgdmod = Model(PseudoContinuum, 
                            param_names=['amplitude',
                                         'exponent',
                                         'fe_norm',
                                         'fe_sd',
                                         'fe_shift'], 
                            independent_vars=['x']) 
            
            bkgdpars = bkgdmod.make_params() 
            bkgdpars['exponent'].value = 1.0
            bkgdpars['amplitude'].value = 1.0
            bkgdpars['fe_norm'].value = 0.05 
            bkgdpars['fe_shift'].value = 0.0

            # Need to have something which can supply value for fwhm
            if fe_FWHM is None:
                fe_FWHM = 4200 * (u.km/u.s)
    
            fe_sigma = fe_FWHM / 2.35 
            fe_pix = fe_sigma / (sp_fe.dv * (u.km/u.s))

            bkgdpars['fe_sd'].value = fe_pix.value  
            bkgdpars['fe_sd'].vary = fe_FWHM_vary

            bkgdpars['fe_norm'].min = 0.0
            bkgdpars['fe_sd'].min = 2000.0 / 2.35 / sp_fe.dv
            bkgdpars['fe_sd'].max = 12000.0 / 2.35 / sp_fe.dv

            bkgdpars['fe_shift'].min = -20.0
            bkgdpars['fe_shift'].max = 20.0

            xdat_cont = np.concatenate((xdat_blue.value, xdat_red.value))
            ydat_cont = np.concatenate((ydat_blue, ydat_red))
            yerr_cont = np.concatenate((yerr_blue, yerr_red))

            # import ipdb; ipdb.set_trace()

            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_cont, 
                                'model':bkgdmod, 
                                'data':ydat_cont, 
                                'sigma':yerr_cont, 
                                'sp_fe':sp_fe},
                           method='slsqp')                           


             
            if verbose:
      
                print out.message, 'chi-squared = {}'.format(out.redchi)
                print fit_report(bkgdpars)
                print 'Fe FWHM = {} km/s'.format(bkgdpars['fe_sd'].value * 2.35 * sp_fe.dv)
            

            if plot:
       
                """
                Plotting continuum / iron fit
                """
    
                fig, axs = plt.subplots(4,1)
    
                axs[0].errorbar(wav.value, 
                                flux, 
                                yerr=err, 
                                linestyle='', 
                                alpha=0.5, 
                                color='grey')
    
                axs[1].errorbar(wav.value, 
                                median_filter(flux,51), 
                                color='grey')
    
    
                xdat_plotting = np.arange(xdat_cont.min(), xdat_cont.max(), 1)
    
                axs[0].plot(xdat_plotting, 
                            resid(p=bkgdpars, 
                                  x=xdat_plotting, 
                                  model=bkgdmod, 
                                  sp_fe=sp_fe), 
                            color='black', 
                            lw=2)
    
                axs[1].plot(xdat_plotting, 
                        resid(p=bkgdpars, 
                              x=xdat_plotting, 
                              model=bkgdmod, 
                              sp_fe=sp_fe), 
                        color='black', 
                        lw=2)
     
                gauss = Gaussian1DKernel(stddev=bkgdpars['fe_sd'].value)
            
                fe_flux = convolve(sp_fe.fl, gauss)
                fe_flux = np.roll(fe_flux, int(bkgdpars['fe_shift'].value))
                f = interp1d(sp_fe.wa, fe_flux) 
                fe_flux = f(xdat_cont)   
    
                axs[0].plot(xdat_cont, bkgdpars['fe_norm'].value * fe_flux, color='red')   
                
                axs[0].plot(xdat_cont, bkgdpars['amplitude']*xdat_cont**bkgdpars['exponent'], color='blue')   
    
                axs[0].set_xlim(xdat_cont.min() - 50.0, xdat_cont.max() + 50.0)
                axs[1].set_xlim(xdat_cont.min() - 50.0, xdat_cont.max() + 50.0)
                
                func_vals = resid(p=bkgdpars, 
                                  x=xdat_cont, 
                                  model=bkgdmod, 
                                  sp_fe=sp_fe)
    
                axs[0].set_ylim(np.median(ydat_cont) - 3.0*np.std(ydat_cont), np.median(ydat_cont) + 3.0*np.std(ydat_cont))
                axs[1].set_ylim(axs[0].get_ylim())
                
    
                xdat_masking = np.arange(wav.min().value, wav.max().value, 0.05)*(u.AA)
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
                            xmin = item.value[0] 
                            xmax = item.value[1] 
                            mask = mask | ((xdat_masking.value > xmin) & (xdat_masking.value < xmax))
                
                
                vdat1_masking = ma.array(vdat_masking.value)
                vdat1_masking[mask] = ma.masked 
    
                for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
                    axs[0].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
                               doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
                               alpha=0.4, 
                               color='powderblue')    
                    axs[1].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
                               doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
                               alpha=0.4, 
                               color='powderblue')    
    
                    axs[2].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
                               doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
                               alpha=0.4, 
                               color='powderblue')    
                    axs[3].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
                               doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
                               alpha=0.4, 
                               color='powderblue')    
    
                axs[2].scatter(xdat_cont, 
                              (ydat_cont - resid(p=bkgdpars, 
                                            x=xdat_cont, 
                                            model=bkgdmod, 
                                            sp_fe=sp_fe)) / yerr_cont, 
                              alpha=0.9, 
                              edgecolor='None', 
                              s=15, 
                              facecolor='black')
    
    
                axs[2].axhline(0.0, color='black', linestyle='--')
                axs[2].set_xlim(axs[0].get_xlim())
    
    
                axs[3].plot(wav.value, flux, color='grey')
    
                axs[3].plot(xdat_cont, 
                        resid(p=bkgdpars, 
                              x=xdat_cont, 
                              model=bkgdmod, 
                              sp_fe=sp_fe), 
                        color='black', 
                        lw=2)
    
                if verbose:

                    global coords
                    coords = [] 
                    cid = fig.canvas.mpl_connect('button_press_event', onclick2)
        
                    plt.show(1)
                    plt.close() 
                
                ##########################################################################

        elif subtract_fe is False: 
       
            xdat_cont = np.concatenate((xdat_blue, xdat_red))
            ydat_cont = np.concatenate((ydat_blue, ydat_red))
            yerr_cont = np.concatenate((yerr_blue, yerr_red))

            bkgdmod = Model(PLModel, 
                            param_names=['amplitude','exponent'], 
                            independent_vars=['x']) 

            bkgdpars = bkgdmod.make_params()
            bkgdpars['exponent'].value = 1.0
            bkgdpars['amplitude'].value = 1.0 

            
            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_cont.value, 
                                'model':bkgdmod, 
                                'data':ydat_cont},
                           method='leastsq') 

            

            if verbose:
                print out.message  
                print fit_report(bkgdpars)



    ####################################################################################################################
    """
    Calculate flux at wavelength mono_lum_wav
    """

    # Calculate just power-law continuum (no Fe)
    cont_mod = Model(PLModel, 
                     param_names=['amplitude','exponent'], 
                     independent_vars=['x']) 

    cont_pars = cont_mod.make_params()
    cont_pars['exponent'].value = bkgdpars['exponent'].value
    cont_pars['amplitude'].value = bkgdpars['amplitude'].value  

    mono_flux = resid(p=cont_pars, 
                      x=[mono_lum_wav.value], 
                      model=cont_mod)[0]
  
    mono_flux = mono_flux / spec_norm

    mono_flux = mono_flux * (u.erg / u.cm / u.cm / u.s / u.AA)

    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)

    mono_lum = mono_flux * (1.0 + z) * 4.0 * math.pi * lumdist**2 * mono_lum_wav 
    

    ######################################################################################################################

    # subtract continuum, define region for fitting
    xdat = wav[fitting]
    yerr = err[fitting]
    vdat = wave2doppler(xdat, w0)

    if subtract_fe is True:
        ydat = flux[fitting] - resid(p=bkgdpars, 
                                     x=xdat.value, 
                                     model=bkgdmod,
                                     sp_fe=sp_fe)
    
    if subtract_fe is False:
        ydat = flux[fitting] -  resid(p=bkgdpars, 
                                      x=xdat.value, 
                                      model=bkgdmod)
    
    """
    Remember that this is to velocity shifted array

    Accepts units km/s or angstrom (observed frame)
    """

    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            mask = np.array([True] * len(vdat))
            for item in maskout:
                if verbose:
                    print 'Not fitting between {0} and {1}'.format(item[0], item[1])
                mask[(vdat > item[0]) & (vdat < item[1])] = False


        elif maskout.unit == (u.AA):

            mask = np.array([True] * len(vdat))
            for item in maskout:
                vlims = wave2doppler(item, w0)
                if verbose:
                    print 'Not fitting between {0} ({1}) and {2} ({3})'.format(item[0], vlims[0], item[1], vlims[1])
                mask[(xdat > (item[0])) & (xdat < (item[1]))] = False

        else:
            print "Units must be km/s or angstrom"

        vdat = vdat[mask]
        ydat = ydat[mask]
        yerr = yerr[mask]
    
    if fit_model == 'MultiGauss':

        # Make model 
        bkgd = ConstantModel()
        mod = bkgd
        pars = bkgd.make_params()
        
        # A bit unnessesary, but I need a way to do += in the loop below
        pars['c'].value = 0.0
        pars['c'].vary = False

        for i in range(nGaussians):
            gmod = GaussianModel(prefix='g{}_'.format(i))
            mod += gmod
            pars += gmod.guess(ydat, x=vdat.value)
               
        for i in range(nGaussians):
            pars['g{}_center'.format(i)].value = 0.0
            pars['g{}_center'.format(i)].min = -5000.0
            pars['g{}_center'.format(i)].max = 5000.0
            pars['g{}_amplitude'.format(i)].min = 0.0
            #pars['g{}_sigma'.format(i)].min = 1000.0
            pars['g{}_sigma'.format(i)].max = 10000.0
        
        # if nGaussians == 2:
        #     pars['g1_center'].set(expr = 'g0_center')
        # if nGaussians == 3:
        #     pars['g1_center'].set(expr = 'g0_center')
        #     pars['g2_center'].set(expr = 'g0_center')


    elif fit_model == 'Ha':

        """
        Implement the Shen+15/11 fitting procedure
        The narrow components of Hα, [NII]λλ6548,6584, [SII]λλ6717,6731 are each fit with a single Gaussian. 
        Their velocity offsets from the systemic redshift and line widths are constrained to be the same
        The relative flux ratio of the two [NII] components is fixed to 2.96 - which way round is this? 
        We impose an upper limit on the narrow line FWHM < 1200 km/s 
        The broad Hα component is modelled in two different ways: 
        a) a single Gaussian with a FWHM > 1200 km/s; 
        b) multiple Gaussians with up to three Gaussians, each with a FWHM >1200 km/s. 
        The second method yields similar results to the fits with a truncated Gaussian-Hermite function. 
        During the fitting, all lines are restricted to be emission lines (i.e., positive flux)

        Also fit Boroson and Green iron template 

        """

        mod = GaussianModel(prefix='ha_n_')  

        mod += GaussianModel(prefix='nii_6548_n_')

        mod += GaussianModel(prefix='nii_6584_n_')

        mod += GaussianModel(prefix='sii_6717_n_')

        mod += GaussianModel(prefix='sii_6731_n_')


        for i in range(nGaussians):

            mod += GaussianModel(prefix='ha_b_{}_'.format(i))  

        pars = mod.make_params() 

        pars['nii_6548_n_amplitude'].value = 1000.0
        pars['nii_6584_n_amplitude'].value = 1000.0
        pars['sii_6717_n_amplitude'].value = 1000.0 
        pars['sii_6731_n_amplitude'].value = 1000.0 
        pars['ha_n_amplitude'].value = 1000.0  
        for i in range(nGaussians):
            pars['ha_b_{}_amplitude'.format(i)].value = 1000.0          

        pars['nii_6548_n_center'].value = wave2doppler(6548*u.AA, w0).value 
        pars['nii_6584_n_center'].value = wave2doppler(6584*u.AA, w0).value 
        pars['sii_6717_n_center'].value = wave2doppler(6717*u.AA, w0).value 
        pars['sii_6731_n_center'].value = wave2doppler(6731*u.AA, w0).value 
        pars['ha_n_center'].value = 100.0    
        for i in range(nGaussians): 
            pars['ha_b_{}_center'.format(i)].value = (-1)**i * 100.0  
            pars['ha_b_{}_center'.format(i)].min = -2000.0  
            pars['ha_b_{}_center'.format(i)].max = 2000.0  

        
        if narrow_fwhm is not None:

            pars['nii_6548_n_sigma'].value = narrow_fwhm.value / 2.35 
            pars['nii_6584_n_sigma'].value = narrow_fwhm.value / 2.35
            pars['sii_6717_n_sigma'].value = narrow_fwhm.value / 2.35
            pars['sii_6731_n_sigma'].value = narrow_fwhm.value / 2.35
            pars['ha_n_sigma'].value = narrow_fwhm.value / 2.35 

            pars['nii_6548_n_sigma'].vary = False
            pars['nii_6584_n_sigma'].vary = False
            pars['sii_6717_n_sigma'].vary = False
            pars['sii_6731_n_sigma'].vary = False
            pars['ha_n_sigma'].vary = False

        else:

            pars['nii_6548_n_sigma'].value = 700.0 / 2.35 
            pars['nii_6584_n_sigma'].value = 700.0 / 2.35 
            pars['sii_6717_n_sigma'].value = 700.0 / 2.35 
            pars['sii_6731_n_sigma'].value = 700.0 / 2.35 
            pars['ha_n_sigma'].value = 700.0 / 2.35 

            pars['ha_n_sigma'].max = 1000.0 / 2.35 
            pars['ha_n_sigma'].min = 400.0 / 2.35 
            pars['nii_6548_n_sigma'].set(expr='ha_n_sigma')
            pars['nii_6584_n_sigma'].set(expr='ha_n_sigma')
            pars['sii_6717_n_sigma'].set(expr='ha_n_sigma')
            pars['sii_6731_n_sigma'].set(expr='ha_n_sigma')

        
        for i in range(nGaussians): 
            pars['ha_b_{}_sigma'.format(i)].value = 1200.0

        pars['nii_6548_n_amplitude'].min = 0.0
        pars['nii_6584_n_amplitude'].min = 0.0
        pars['sii_6717_n_amplitude'].min = 0.0 
        pars['sii_6731_n_amplitude'].min = 0.0 
        pars['ha_n_amplitude'].min = 0.0 
        for i in range(nGaussians): 
            pars['ha_b_{}_amplitude'.format(i)].min = 0.0  
        

        pars['ha_n_center'].min = -2000.0
        pars['ha_n_center'].max = 2000.0

        pars['sii_6731_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6731*u.AA, w0).value))
        pars['sii_6717_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6717*u.AA, w0).value))
        pars['nii_6548_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6548*u.AA, w0).value))
        pars['nii_6584_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6584*u.AA, w0).value))

        pars['nii_6548_n_amplitude'].set(expr='0.333*nii_6584_n_amplitude')

        for i in range(nGaussians): 
            pars['ha_b_{}_sigma'.format(i)].min = 1200.0 / 2.35

    elif fit_model == 'Hb':

        """
        From Shen+15/11

        Need to get Boroson and Green (1992) optical iron template, 
        which people seem to convolve with a Gaussian
    
        Model each [OIII] line with Gaussian, one for the core and the other for the blue wing.
        Decide whether to fix flux ratio 3:1
        Velocity offset and FWHM of narrow Hb tied to the core [OIII] components
        Upper limit of 1200 km/s on the narrow line FWHM
        Broad Hb modelled by single gaussian, or up to 3 Gaussians each with FWHM > 1200 km/s

        It seams like a = b not the same as b = a, which is annoying. 
        Need to be careful

        """

        mod = GaussianModel(prefix='oiii_4959_n_')

        mod += GaussianModel(prefix='oiii_5007_n_')

        mod += GaussianModel(prefix='oiii_4959_b_')

        mod += GaussianModel(prefix='oiii_5007_b_')

        if hb_narrow is True: 
            mod += GaussianModel(prefix='hb_n_')  

        for i in range(nGaussians):

            mod += GaussianModel(prefix='hb_b_{}_'.format(i))  

        pars = mod.make_params() 

        pars['oiii_4959_n_amplitude'].value = 1000.0
        pars['oiii_5007_n_amplitude'].value = 1000.0
        pars['oiii_4959_b_amplitude'].value = 1000.0 
        pars['oiii_5007_b_amplitude'].value = 1000.0 
        if hb_narrow is True: 
            pars['hb_n_amplitude'].value = 1000.0  
        for i in range(nGaussians):
            pars['hb_b_{}_amplitude'.format(i)].value = 1000.0  

        pars['oiii_4959_n_center'].value = wave2doppler(4960.295*u.AA, w0).value 
        pars['oiii_5007_n_center'].value = wave2doppler(5008.239*u.AA, w0).value 
        pars['oiii_4959_b_center'].value = wave2doppler(4960.295*u.AA, w0).value 
        pars['oiii_5007_b_center'].value = wave2doppler(5008.239*u.AA, w0).value 
        if hb_narrow is True: 
            pars['hb_n_center'].value = 0.0    
        for i in range(nGaussians): 
            pars['hb_b_{}_center'.format(i)].value = -100.0  
            pars['hb_b_{}_center'.format(i)].min = -2000.0  
            pars['hb_b_{}_center'.format(i)].max = 2000.0  

        pars['oiii_4959_n_sigma'].value = 800 / 2.35
        pars['oiii_5007_n_sigma'].value = 800 / 2.35
        pars['oiii_4959_b_sigma'].value = 1200.0 
        pars['oiii_5007_b_sigma'].value = 1200.0 
        if hb_narrow is True: 
            pars['hb_n_sigma'].value = 800 / 2.35
        for i in range(nGaussians): 
            pars['hb_b_{}_sigma'.format(i)].value = 1200.0

        pars['oiii_4959_n_amplitude'].min = 0.0
        pars['oiii_5007_n_amplitude'].min = 0.0
        pars['oiii_4959_b_amplitude'].min = 0.0 
        pars['oiii_5007_b_amplitude'].min = 0.0 
        if hb_narrow is True:      
            pars['hb_n_amplitude'].min = 0.0 
        for i in range(nGaussians): 
            pars['hb_b_{}_amplitude'.format(i)].min = 0.0  
        

        pars['oiii_5007_n_center'].min = wave2doppler(5008.239*u.AA, w0).value - 2000.0
        pars['oiii_5007_n_center'].max = wave2doppler(5008.239*u.AA, w0).value + 2000.0
        if hb_narrow is True: 
            pars['hb_n_center'].set(expr = 'oiii_5007_n_center-{}'.format(wave2doppler(5008.239*u.AA, w0).value)) 
        pars['oiii_4959_n_center'].set(expr = 'oiii_5007_n_center+{}'.format(wave2doppler(4960.295*u.AA, w0).value - wave2doppler(5008.239*u.AA, w0).value))

        pars['oiii_5007_n_sigma'].max = 900.0 / 2.35 
        pars['oiii_5007_n_sigma'].min = 400.0 / 2.35 
        pars['oiii_4959_n_sigma'].set(expr='oiii_5007_n_sigma')
        if hb_narrow is True: 
            pars['hb_n_sigma'].set(expr='oiii_5007_n_sigma')

        pars.add('oiii_5007_b_center_delta') 
        pars['oiii_5007_b_center_delta'].value = 500.0 
        pars['oiii_5007_b_center_delta'].min = -1500.0 
        pars['oiii_5007_b_center_delta'].max = 1500.0

        pars['oiii_5007_b_center'].set(expr='oiii_5007_n_center-oiii_5007_b_center_delta')

        pars.add('oiii_4959_b_center_delta') 
        pars['oiii_4959_b_center_delta'].value = 500.0 
        pars['oiii_4959_b_center_delta'].min = -1500.0 
        pars['oiii_4959_b_center_delta'].max = 1000.0

        pars['oiii_4959_b_center'].set(expr='oiii_4959_n_center-oiii_4959_b_center_delta')

        for i in range(nGaussians): 
            pars['hb_b_{}_sigma'.format(i)].min = 1000.0 / 2.35
    
        pars['oiii_5007_b_sigma'].min = 1200.0 / 2.35 
        pars['oiii_5007_b_sigma'].max = 6000.0 / 2.35
        pars['oiii_4959_b_sigma'].set(expr='oiii_5007_b_sigma')

        # Set amplitude of [oiii] broad component less than narrow component 
        pars.add('oiii_5007_b_amp_delta')
        pars['oiii_5007_b_amp_delta'].value = 0.1
        pars['oiii_5007_b_amp_delta'].min = 0.0 

        pars['oiii_5007_b_amplitude'].set(expr='((oiii_5007_n_amplitude/oiii_5007_n_sigma)-oiii_5007_b_amp_delta)*oiii_5007_b_sigma')

        # Set 3:1 [OIII] peak ratio  
        pars['oiii_4959_n_amplitude'].set(expr='0.3333*oiii_5007_n_amplitude')

        # Also set 3:1 [OIII] peak ratio for broad components 
        pars.add('oiii_4959_b_amp_delta')   
        pars['oiii_4959_b_amp_delta'].value = 0.1 
        pars['oiii_4959_b_amp_delta'].min = 0.0

        pars['oiii_4959_b_amplitude'] .set(expr='oiii_4959_b_sigma*((oiii_5007_b_amplitude/oiii_5007_b_sigma)-oiii_4959_b_amp_delta)/3')

        # Constrain the luminosity of the narrow Hb component to be less than 1/8 the broad luminosity, which
        # is the minimum in Shen's fitting. The integral of a properly normalised Gaussian is one, so the luminosity
        # is proportional to the amplitude. 
        # will break if number of broad hb components isn't equal to 3. 
        
        # pars.add('hb_n_amplitude_delta')
        # pars['hb_n_amplitude_delta'].value = 100.0
        # pars['hb_n_amplitude_delta'].min = 80.0 

        # pars['hb_n_amplitude'].set(expr='(hb_b_0_amplitude + hb_b_1_amplitude + hb_b_2_amplitude) / hb_n_amplitude_delta') 

    if (verbose) and (any(y < 0.0 for y in ydat)):
        print 'Warning: Negative flux values in fitting region!'

    # Remove negative flux values which can mess up fit 

    if mask_negflux:

        posflux = (ydat > 0.0) 
    
        xdat = xdat[posflux] 
        ydat = ydat[posflux] 
        yerr = yerr[posflux]
        vdat = vdat[posflux] 

    # Messes up fit if have negative/zero weights 
    good_pix = (yerr > 0.0)

    xdat = xdat[good_pix]
    ydat = ydat[good_pix]
    yerr = yerr[good_pix]
    vdat = vdat[good_pix]


    out = minimize(resid,
                   pars,
                   args=(np.asarray(vdat), mod, ydat, yerr),
                   method = fitting_method)


    if verbose:
    
        print out.message
        print fit_report(pars)

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


        if subtract_fe is True:
            flux_dump = flux - resid(p=bkgdpars, 
                                     x=wav.value, 
                                     model=bkgdmod,
                                     sp_fe=sp_fe)
        
        if subtract_fe is False:
            flux_dump = flux - resid(p=bkgdpars, 
                                     x=wav.value, 
                                     model=bkgdmod)

        flx_file = os.path.join(save_dir, 'flx.txt')
        parfile = open(flx_file, 'wb')
        pickle.dump(flux_dump, parfile, -1)
        parfile.close()

        err_file = os.path.join(save_dir, 'err.txt')
        parfile = open(err_file, 'wb')
        pickle.dump(err, parfile, -1)
        parfile.close()

        fittxt = ''    
        fittxt += strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n \n'
        fittxt += plot_title + '\n \n'
        fittxt += r'Converged with chi-squared = ' + str(out.chisqr) + ', DOF = ' + str(out.nfree) + '\n \n'
        fittxt += fit_report(pars)
    

        with open(os.path.join(save_dir, 'fit.txt'), 'w') as f:
            f.write(fittxt) 

    # Calculate stats 

    if fit_model == 'Hb':

        """
        Only use broad Hb components to calculate stats 
        """

        mod_broad_hb = ConstantModel()

        for i in range(nGaussians):
            mod_broad_hb += GaussianModel(prefix='hb_b_{}_'.format(i))  

        pars_broad_hb = mod_broad_hb.make_params()
       
        pars_broad_hb['c'].value = 0.0
       
        for key, value in pars.valuesdict().iteritems():
            if key.startswith('hb_b_'):
                pars_broad_hb[key].value = value 
                

        integrand = lambda x: mod_broad_hb.eval(params=pars_broad_hb, x=np.array(x))

    elif fit_model == 'Ha':

        """
        Only use broad Hb components to calculate stats 
        """

        mod_broad_ha = ConstantModel()

        for i in range(nGaussians):
            mod_broad_ha += GaussianModel(prefix='ha_b_{}_'.format(i))  

        pars_broad_ha = mod_broad_ha.make_params()
       
        pars_broad_ha['c'].value = 0.0
       
        for key, value in pars.valuesdict().iteritems():
            if key.startswith('ha_b_'):
                pars_broad_ha[key].value = value 
                

        integrand = lambda x: mod_broad_ha.eval(params=pars_broad_ha, x=np.array(x))    

    else:

        integrand = lambda x: mod.eval(params=pars, x=np.array(x))

    # Calculate FWHM of distribution
    if line_region.unit == u.AA:
        line_region = wave2doppler(line_region, w0)

    dv = 1.0 # i think if this was anything else might need to change intergration.
    xs = np.arange(line_region.value[0], line_region[1].value, dv) 

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
    sd = np.sqrt(v)

    # Equivalent width
    flux_line = integrand(xs)
    xs_wav = doppler2wave(xs*(u.km/u.s), w0)
 
    if subtract_fe is True:
        flux_bkgd = resid(p=bkgdpars, 
                          x=xs_wav.value, 
                          model=bkgdmod,
                          sp_fe=sp_fe)
    
    if subtract_fe is False:
        flux_bkgd = resid(p=bkgdpars, 
                          x=xs_wav.value, 
                          model=bkgdmod)
    

    f = (flux_line + flux_bkgd) / flux_bkgd
    eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
    eqw = np.nansum(eqw)

    # Broad luminosity
    broad_lum = norm * (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm 
    if fit_model == 'Ha':
        narrow_lum = pars['ha_n_amplitude'].value * (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm 
        narrow_fwhm = pars['ha_n_fwhm'].value 
    elif fit_model == 'Hb':
        narrow_lum = pars['hb_n_amplitude'].value * (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm     
        if hb_narrow is True:
            narrow_fwhm = pars['hb_n_fwhm'].value 
        else:
            narrow_fwhm = 0.0 
    else: 
        narrow_lum = 0.0 * (u.erg / u.s)
        narrow_fwhm = 0.0 

    if verbose:
        print plot_title, \
             '{0:.2f},'.format(root2 - root1), \
             '{0:.2f},'.format(sd), \
             '{0:.2f},'.format(md), \
             '{0:.2f},'.format(func_center), \
             '{0:.2f},'.format(eqw), \
             '{0:.2f},'.format(np.log10(broad_lum.value)), \
             '{0:.2f},'.format(narrow_fwhm), \
             '{0:.2f},'.format(np.log10(narrow_lum.value)), \
             '{0:.2f}'.format(out.redchi)

    if verbose:
        print 'Monochomatic luminosity at {0} = {1:.3f}'.format(mono_lum_wav, np.log10(mono_lum.value)) 

    # """
    # Calculate S/N ratio per resolution element
    # """

    good = (yerr > 0) & ~np.isnan(ydat)
    if len(good.nonzero()[0]) == 0:
        print('No good data in this range!')

    ydat = ydat[good]
    yerr = yerr[good]
    
    if verbose:
        print 'Median SNR per pixel = {0:.2f}'.format(np.median(ydat / yerr))
  
    # # 33.02 is the A per resolution element I measured from the Arc spectrum
    # # print 'snr_ha = {0:.2f}'.format(np.mean(np.sqrt(33.02/dw1) * snr))

    fit_out = {'name':plot_title, 
               'fwhm':root2 - root1,
               'sigma': sd,
               'median': md,
               'cen': func_center,
               'eqw': eqw,
               'broad_lum':np.log10(broad_lum.value),
               'narrow_fwhm':narrow_fwhm,
               'narrow_lum':np.log10(narrow_lum.value),
               'redchi':out.redchi,
               'snr':np.median(ydat / yerr),
               'monolum':np.log10(mono_lum.value)}

    # print plot_title 
    # print 'peak_ha = {0:.2f}*(u.km/u.s),'.format(func_center)
    # print 'fwhm_ha = {0:.2f}*(u.km/u.s),'.format(root2 - root1)
    # print 'median_ha = {0:.2f}*(u.km/u.s),'.format(md)
    # print 'sigma_ha = {0:.2f}*(u.km/u.s),'.format(sd)
    # print 'chired_ha = {0:.2f},'.format(out.redchi)
    # print 'eqw_ha = {0:.2f}*u.AA,'.format(eqw)

    if save_dir is not None:

        with open(os.path.join(save_dir, 'fit.txt'), 'a') as f:
            f.write('\n \n')
            f.write('Peak: {0:.2f} km/s \n'.format(func_center))
            f.write('FWHM: {0:.2f} km/s \n'.format(root2 - root1))
            f.write('Median: {0:.2f} km/s \n'.format(md))
            f.write('Sigma: {0:.2f} km/s \n'.format(sd))
            f.write('Reduced chi-squared: {0:.2f} \n'.format(out.redchi))
            f.write('EQW: {0:.2f} A \n'.format(eqw))      

        my_line_props = line_props(plot_title, func_center, root2 - root1, md, sd, out.redchi, eqw)  
        param_file = os.path.join(save_dir, 'line_props.txt')
        parfile = open(param_file, 'w')
        pickle.dump(my_line_props, parfile, -1)
        parfile.close()
      


    if plot is True:

        if subtract_fe is True:
            flux_plot = flux - resid(p=bkgdpars, 
                                     x=wav.value, 
                                     model=bkgdmod,
                                     sp_fe=sp_fe)
        
        if subtract_fe is False:
            flux_plot = flux - resid(p=bkgdpars, 
                                     x=wav.value, 
                                     model=bkgdmod)

        plot_fit(wav=wav,
                 flux = flux_plot,
                 err=err,
                 pars=pars,
                 mod=mod,
                 out=out,
                 plot_savefig = plot_savefig,
                 save_dir = save_dir,
                 maskout = maskout,
                 z=z,
                 w0=w0,
                 continuum_region=continuum_region,
                 fitting_region=fitting_region,
                 plot_region=plot_region,
                 plot_title=plot_title,
                 line_region=line_region,
                 mask_negflux = mask_negflux,
                 fit_model=fit_model,
                 hb_narrow=hb_narrow,
                 verbose=verbose)


    return fit_out 


if __name__ == '__main__':

    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    p0 = [1., 0, 500.0]

    dv = 1 
    x = np.arange(-10000,10000, dv)

    flux = gauss(x, *p0)
    
    norm = np.sum(flux * dv)
    pdf = flux / norm
    cdf = np.cumsum(pdf)
    cdf_r = np.cumsum(pdf[::-1])[::-1] # reverse cumsum


    m = np.sum(x * pdf * dv)

    v = np.sum( (x-m)**2 * pdf * dv )
    sd = np.sqrt(v)

    print sd 
