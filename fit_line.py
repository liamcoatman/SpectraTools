# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:13:56 2015

@author: lc585

Fit emission line with model.

"""
from __future__ import division

from termcolor import colored
import numpy as np
import astropy.units as u
from lmfit.models import GaussianModel, LorentzianModel, PowerLawModel, ConstantModel
from lmfit import minimize, Parameters, fit_report, Model, Minimizer
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
from astropy.convolution import Gaussian1DKernel, convolve, Box2DKernel
from scipy.interpolate import interp1d 
from os.path import expanduser
from barak import spec 
from astropy import constants as const
import mpld3
from scipy.signal import medfilt2d, medfilt
from functools import partial 
import pandas as pd
from multiprocessing import Pool 
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning) # I get "elementwise comparison failed" during plotting, but doesn't seem important
# warnings.simplefilter(action = "error", category = RuntimeWarning)
np.set_printoptions(threshold='nan')

def gausshermite_component(x, amp, sig, cen, order):

    if order == 0:
        return (amp/(np.sqrt(2*math.pi)*sig)) * np.exp(-(x-cen)**2 /(2*sig**2))
    if order == 1:
        return np.sqrt(2.0) * x * (amp/(np.sqrt(2*math.pi)*sig)) * np.exp(-(x-cen)**2 /(2*sig**2))
    if order == 2:    
        return (2.0*x*x - 1.0) / np.sqrt(2.0) * (amp/(np.sqrt(2*math.pi)*sig)) * np.exp(-(x-cen)**2 /(2*sig**2))
    if order == 3:   
        return x * (2.0*x*x - 3.0) / np.sqrt(3.0) * (amp/(np.sqrt(2*math.pi)*sig)) * np.exp(-(x-cen)**2 /(2*sig**2))
    if order == 4:   
        return (x*x*(4.0*x*x-12.0)+3.0) / (2.0*np.sqrt(6.0)) * (amp/(np.sqrt(2*math.pi)*sig)) * np.exp(-(x-cen)**2 /(2*sig**2))
    if order == 5:    
        return (x*(x*x*(4.0*x*x-20.0) + 15.0)) / (2.0*np.sqrt(15.0)) * (amp/(np.sqrt(2*math.pi)*sig)) * np.exp(-(x-cen)**2 /(2*sig**2))
    if order == 6:    
        return (x*x*(x*x*(8.0*x*x-60.0) + 90.0) - 15.0) / (12.0*np.sqrt(5.0)) * (amp/(np.sqrt(2*math.pi)*sig)) * np.exp(-(x-cen)**2 /(2*sig**2))

def gausshermite_0(x, amp0, sig0, cen0):

    h0 = gausshermite_component(x, amp0, sig0, cen0, 0)

    return h0 

def gausshermite_1(x, 
                   amp0, 
                   sig0, 
                   cen0, 
                   amp1, 
                   sig1, 
                   cen1):

    h0 = gausshermite_component(x, amp0, sig0, cen0, 0)
    h1 = gausshermite_component(x, amp1, sig1, cen1, 1)

    return h0 + h1

def gausshermite_2(x, 
                   amp0, 
                   sig0, 
                   cen0, 
                   amp1, 
                   sig1, 
                   cen1, 
                   amp2, 
                   sig2, 
                   cen2):

    h0 = gausshermite_component(x, amp0, sig0, cen0, 0)
    h1 = gausshermite_component(x, amp1, sig1, cen1, 1)
    h2 = gausshermite_component(x, amp2, sig2, cen2, 2)

    return h0 + h1 + h2 

def gausshermite_3(x, 
                   amp0, 
                   sig0, 
                   cen0, 
                   amp1, 
                   sig1, 
                   cen1, 
                   amp2, 
                   sig2, 
                   cen2, 
                   amp3, 
                   sig3, 
                   cen3):   

    h0 = gausshermite_component(x, amp0, sig0, cen0, 0)
    h1 = gausshermite_component(x, amp1, sig1, cen1, 1)
    h2 = gausshermite_component(x, amp2, sig2, cen2, 2)
    h3 = gausshermite_component(x, amp3, sig3, cen3, 3)

    return h0 + h1 + h2 + h3 

def gausshermite_4(x, 
                   amp0, 
                   sig0, 
                   cen0, 
                   amp1, 
                   sig1, 
                   cen1, 
                   amp2, 
                   sig2, 
                   cen2, 
                   amp3, 
                   sig3, 
                   cen3, 
                   amp4, 
                   sig4, 
                   cen4):

    h0 = gausshermite_component(x, amp0, sig0, cen0, 0)
    h1 = gausshermite_component(x, amp1, sig1, cen1, 1)
    h2 = gausshermite_component(x, amp2, sig2, cen2, 2)
    h3 = gausshermite_component(x, amp3, sig3, cen3, 3)
    h4 = gausshermite_component(x, amp4, sig4, cen4, 4)

    return h0 + h1 + h2 + h3 + h4 

def gausshermite_5(x, 
                   amp0, 
                   sig0, 
                   cen0, 
                   amp1, 
                   sig1, 
                   cen1, 
                   amp2, 
                   sig2, 
                   cen2, 
                   amp3, 
                   sig3, 
                   cen3, 
                   amp4, 
                   sig4, 
                   cen4, 
                   amp5, 
                   sig5, 
                   cen5):   

    h0 = gausshermite_component(x, amp0, sig0, cen0, 0)
    h1 = gausshermite_component(x, amp1, sig1, cen1, 1)
    h2 = gausshermite_component(x, amp2, sig2, cen2, 2)
    h3 = gausshermite_component(x, amp3, sig3, cen3, 3)
    h4 = gausshermite_component(x, amp4, sig4, cen4, 4)
    h5 = gausshermite_component(x, amp5, sig5, cen5, 5)

    return h0 + h1 + h2 + h3 + h4 + h5 

def gausshermite_6(x, 
                   amp0, 
                   sig0, 
                   cen0, 
                   amp1, 
                   sig1, 
                   cen1, 
                   amp2, 
                   sig2, 
                   cen2, 
                   amp3, 
                   sig3, 
                   cen3, 
                   amp4, 
                   sig4, 
                   cen4, 
                   amp5, 
                   sig5, 
                   cen5, 
                   amp6, 
                   sig6, 
                   cen6):

    h0 = gausshermite_component(x, amp0, sig0, cen0, 0)
    h1 = gausshermite_component(x, amp1, sig1, cen1, 1)
    h2 = gausshermite_component(x, amp2, sig2, cen2, 2)
    h3 = gausshermite_component(x, amp3, sig3, cen3, 3)
    h4 = gausshermite_component(x, amp4, sig4, cen4, 4)
    h5 = gausshermite_component(x, amp5, sig5, cen5, 5)
    h6 = gausshermite_component(x, amp6, sig6, cen6, 6)

    return h0 + h1 + h2 + h3 + h4 + h5 + h6 


def rebin(wa, fl, er, n):
    
    """ 
    Bins up the spectrum by averaging the values of every n pixels. 
    Sometimes get negative or zero errors in array
    Assign these large weights 
    """ 
    
    # not sure if this is a good idea or not
    er[er <= 0.0] = 1.e6 

    remain = -(len(wa) % n) or None

    wa = wa[:remain].reshape(-1, n)
    fl = fl[:remain].reshape(-1, n)
    er = er[:remain].reshape(-1, n)
   
    n = float(n)
   
    wa = np.nansum(wa, axis=1) / n
    fl = np.nansum(fl / er**2, axis=1) / np.nansum(1.0 / er**2, axis=1)  
    er = 1.0 / np.sqrt(np.nansum(1 / er**2, axis=1) ) 
    
    return wa, fl, er 

# Simple mouse click function to store coordinates
def onclick(event):

    global ix

    if event.button == 2:
    
        ix = event.xdata
    
        coords.append(ix)
    
        # if len(coords) % 2 == 0:
        #     print '[{0:.0f}, {1:.0f}]'.format(coords[-2], coords[-1])  
        
        # fig.canvas.mpl_disconnect(cid)
    
    if event.button == 3:

        o = '['
        for i in range(int(len(coords) / 2)):  
            if i == int(len(coords) / 2) - 1:
                o += '[{0:.0f}, {1:.0f}]'.format(coords[2*i], coords[2*i+1])  
            else:
                o += '[{0:.0f}, {1:.0f}],'.format(coords[2*i], coords[2*i+1])  
        o += ']*(u.km/u.s)'
        print o 




    return None 

def onclick2(event, w0=4862.721*u.AA):

    global ix
    
    ix = event.xdata
    ix = wave2doppler(ix*u.AA, w0).value 
    coords.append(ix)

    if len(coords) % 4 == 0:
        print '[[{0:.0f}, {1:.0f}]*(u.km/u.s),[{2:.0f}, {3:.0f}]*(u.km/u.s)]'.format(coords[-4], coords[-3], coords[-2], coords[-1])  
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
      
def resid(params=None, 
          x=None, 
          model=None, 
          data=None, 
          sigma=None, 
          **kwargs):

    mod = model.eval(params=params, x=x, **kwargs)

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

def PseudoContinuum(x, 
                    amplitude,
                    exponent,
                    fe_norm,
                    fe_sd,
                    fe_shift,
                    sp_fe):
 
    gauss = Gaussian1DKernel(stddev=fe_sd)

    fe_flux = convolve(sp_fe.fl, gauss)
    
    f = interp1d(sp_fe.wa - fe_shift, 
                 fe_flux, 
                 bounds_error=False, 
                 fill_value=0.0)

    fe_flux = f(x)   

    # so parameter is around 1
    amp = amplitude * 5000.0**(-exponent)

    return fe_norm * fe_flux + amp*x**exponent

def PLModel(x, amplitude, exponent):

    # should probably change this to 1350 when fitting CIV
    amp = amplitude * 5000.0**(-exponent) 

    return amp*x**exponent 

def plot_spec(wav=None,
              flux=None,
              plot_savefig=None,
              plot_title='',
              save_dir=None,
              z=0.0,
              w0=6564.89*u.AA,
              continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
              fitting_region=[6400,6800]*u.AA,
              plot_region=None,
              verbose=True):    

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(wav.value, 
               flux, 
               edgecolor='None', 
               s=15, 
               alpha=0.9, 
               facecolor='black')

    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    ax.axvspan(fitting_region[0].value, 
               fitting_region[1].value, 
               alpha=0.4, 
               color='moccasin')
       
    ax.axvspan(continuum_region[0][0].value, 
               continuum_region[0][1].value, 
               alpha=0.4, 
               color='powderblue')

    ax.axvspan(continuum_region[1][0].value, 
               continuum_region[1][1].value, 
               alpha=0.4, 
               color='powderblue')

    ax.set_ylim(np.median(flux) - 3.0*np.std(flux), np.median(flux) + 3.0*np.std(flux))

    ax.set_ylabel(r'F$_\lambda$', fontsize=12)
    ax.set_xlabel(r'Wavelength [$\AA$]', fontsize=12)
    
    fig.tight_layout() 
    
    if plot_savefig is not None:
        fig.savefig(os.path.join(save_dir, plot_savefig))

    if verbose:
        plt.show(1)
    
    plt.close()
    
    return None    


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
             verbose=True,
             reject_outliers = False,  
             reject_width = 20,
             reject_sigma = 3.0,
             gh_order=None,
             xscale=1.0):

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

    fit.scatter(vdat.value, ydat, edgecolor='None', s=5, facecolor='grey')

    # Mark continuum fitting region
    # Doesn't make sense to transform to wave and then back to velocity but I'm being lazy.
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    # set yaxis range 
    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])
    integrand = lambda x: mod.eval(params=pars, x=np.array(x))
    func_max = np.max(integrand(wave2doppler(wav[fitting], w0).value/xscale))
    fit.set_ylim(-0.3*func_max, 1.3*func_max)
    eb.set_ylim(fit.get_ylim())
    residuals.set_ylim(-8,8)
    fs.set_ylim(-1*func_max, 2*func_max)

    # set xaxis range 
    plotting_limits = wave2doppler(plot_region, w0)
    fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)
    residuals.set_xlim(fit.get_xlim())
    eb.set_xlim(fit.get_xlim())
    fs.set_xlim(wav.min().value, wav.max().value)

    # Lines to mark region where parameters calculated 
    eb.axvline(line_region[0].value, color='black', linestyle='--')
    eb.axvline(line_region[1].value, color='black', linestyle='--')

    fit.axvline(line_region[0].value, color='black', linestyle='--')
    fit.axvline(line_region[1].value, color='black', linestyle='--')

    fs.axvline(doppler2wave(line_region[0], w0).value, color='black', linestyle='--')
    fs.axvline(doppler2wave(line_region[1], w0).value, color='black', linestyle='--')

    # horizontal line at zero 
    fit.axhline(0, color='black', linestyle='--')
    fs.axhline(0, color='black', linestyle='--')
    eb.axhline(0, color='black', linestyle='--')
    residuals.axhline(0.0, color='black', linestyle='--')

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

    line, = fit.plot(np.sort(vdat.value), resid(pars, np.sort(vdat.value/xscale), mod), color='black', lw=2)

    residuals.scatter(vdat.value, 
                     (ydat - resid(pars, vdat.value/xscale, mod)) / yerr, 
                      alpha=0.9, 
                      edgecolor='None', 
                      s=5, 
                      facecolor='black')
    

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
    
            p['center'].value = pars['hb_n_center'].value
            p['sigma'].value = pars['hb_n_sigma'].value
            p['amplitude'].value = pars['hb_n_amplitude'].value
    
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
                 c='orange',
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

    if fit_model == 'siiv': 

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['g0_center'].value
        p['sigma'].value = pars['g0_sigma'].value
        p['amplitude'].value = pars['g0_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='-')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['g1_center'].value
        p['sigma'].value = pars['g1_sigma'].value
        p['amplitude'].value = pars['g1_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='-')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['g2_center'].value
        p['sigma'].value = pars['g2_sigma'].value
        p['amplitude'].value = pars['g2_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='orange',
                 linestyle='-')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['g3_center'].value
        p['sigma'].value = pars['g3_sigma'].value
        p['amplitude'].value = pars['g3_amplitude'].value

        fit.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='orange',
                 linestyle='-')



    if fit_model == 'GaussHermite':

        i = 0 
        
        while i <= gh_order:
            
            flux_com = gausshermite_component(vdat.value/xscale, 
                                              pars['amp{}'.format(i)].value, 
                                              pars['sig{}'.format(i)].value, 
                                              pars['cen{}'.format(i)].value, 
                                              i)
    
            fit.plot(vdat.value, flux_com, color='gold', lw=1)
             
            i += 1 

    # plot error bars and model  
    eb.errorbar(vdat.value, 
                ydat, 
                yerr=yerr, 
                linestyle='', 
                alpha=0.5, 
                color='grey')

    eb.plot(np.sort(vdat.value), 
            resid(pars, np.sort(vdat.value/xscale), mod), 
            color='black', 
            lw=2)
    
    
    # Set labels
    fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    eb.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=10)
    eb.set_ylabel(r'F$_\lambda$', fontsize=12)
    residuals.set_ylabel("Residual")

    #######################################

    # histogram of residuals 
    # this contains pixels which have been masked out   

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])
    xdat = wav[fitting]
    vdat = wave2doppler(xdat, w0)
    ydat = flux[fitting]
    yerr = err[fitting]
    hg = fig.add_subplot(5,1,4)

    bad = np.isnan(ydat) | np.isinf(yerr)
    ydat = ydat[~bad]
    vdat = vdat[~bad]
    yerr = yerr[~bad]
  
    hg.hist((ydat - resid(pars, vdat.value/xscale, mod)) / yerr, 
            bins=np.arange(-5,5,0.25), 
            normed=True, 
            edgecolor='None', 
            facecolor='lightgrey')
    x_axis = np.arange(-5, 5, 0.001)
    hg.plot(x_axis, norm.pdf(x_axis, 0, 1), color='black', lw=2)


    #########################################

    # plt full spectrum 
    fs.plot(wav, median_filter(flux, 5), color='black', lw=1)
    

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
        residuals.scatter(vdat_bad, (ydat_bad - resid(pars, vdat_bad.value, mod)) / yerr_bad, color='darkred', s=40, marker='x')
        eb.scatter(vdat_bad, ydat_bad, color='darkred', s=40, marker='x')
        fs.scatter(xdat_bad, ydat_bad, color='darkred', s=40, marker='x')

    if reject_outliers:

        # xdat = wav
        # vdat = wave2doppler(xdat, w0)
        # ydat = flux
        # yerr = err
    
        # flux_smooth = median_filter(flux[fitting], size=reject_width)
        # bad_pix = np.where((flux[fitting] - flux_smooth) / err[fitting] > reject_sigma)[0]

        # bad_pix = np.concatenate((bad_pix - 2,
        #                           bad_pix - 1,
        #                           bad_pix, 
        #                           bad_pix + 1, 
        #                           bad_pix + 2))

        # # if we go over the edge then remove these 
        # bad_pix = bad_pix[(bad_pix > 0) & (bad_pix < len(wav[fitting].value) - 1)]

        # mask = np.ones(wav[fitting].value.shape, dtype=bool)
        # mask[bad_pix] = False 

        # fit.scatter(wave2doppler(wav[fitting][bad_pix], w0).value, flux[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')
        # residuals.scatter(wave2doppler(wav[fitting][bad_pix], w0).value, (flux[fitting][bad_pix] - resid(pars, wave2doppler(wav[fitting][bad_pix], w0).value, mod)) / err[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')
        # eb.scatter(wave2doppler(wav[fitting][bad_pix], w0).value, flux[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')
        # fs.scatter(wav[fitting][bad_pix].value, flux[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')



        flux_smooth = medfilt(flux[fitting], kernel_size=reject_width)
        bad_pix = np.where((flux_smooth - flux[fitting]) / err[fitting] > reject_sigma)[0]

        bad_pix = np.concatenate((bad_pix - 2,
                                  bad_pix - 1,
                                  bad_pix, 
                                  bad_pix + 1, 
                                  bad_pix + 2))

        # if we go over the edge then remove these 
        bad_pix = bad_pix[(bad_pix > 0) & (bad_pix < len(wav[fitting].value) - 1)]

        mask = np.ones(wav[fitting].value.shape, dtype=bool)
        mask[bad_pix] = False 

        fit.scatter(wave2doppler(wav[fitting][bad_pix], w0).value, flux[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')
        residuals.scatter(wave2doppler(wav[fitting][bad_pix], w0).value, (flux[fitting][bad_pix] - resid(pars, wave2doppler(wav[fitting][bad_pix], w0).value, mod)) / err[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')
        eb.scatter(wave2doppler(wav[fitting][bad_pix], w0).value, flux[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')
        fs.scatter(wav[fitting][bad_pix].value, flux[fitting][bad_pix], color='mediumseagreen', s=40, marker='x')







    fig.tight_layout()

    # Call click func

    if verbose:

        global coords
        coords = [] 
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

    if plot_savefig is not None:
        fig.savefig(os.path.join(save_dir, plot_savefig))
        # mpld3.save_html(fig, os.path.join(save_dir, 'plot.html')) 
      

    if verbose:
        plt.show(1)
    
    plt.close()

    return None

# Make a simpler plot to display with d3. 

def plot_fit_d3(wav=None,
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
                verbose=True,
                xscale=1.0,
                gh_order=None):


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
    
    fig = plt.figure(figsize=(6,4))

    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(vdat.value, 
               ydat, 
               edgecolor='None', 
               s=5, 
               facecolor='grey')

    # Region where equivalent width etc. calculated.
    integrand = lambda x: mod.eval(params=pars, x=np.array(x))
    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])
    func_max = np.max(integrand(wave2doppler(wav[fitting], w0).value/xscale))
   
    # set y axis scale
    ax.set_ylim(-0.3*func_max, 1.3*func_max)

    line, = ax.plot(np.sort(vdat.value), 
                     resid(pars, np.sort(vdat.value) / xscale, mod),
                     color='black', 
                     lw=2)

    plotting_limits = wave2doppler(plot_region, w0)
    ax.set_xlim(plotting_limits[0].value, plotting_limits[1].value)

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
    
                ax.plot( np.sort(vdat.value), comp_mod.eval(comp_p, x=np.sort(vdat.value)) )
    
        if nlorentzians > 1:
    
            for i in range(nlorentzians):
                comp_mod = LorentzianModel()
                comp_p = comp_mod.make_params()
    
                for key, v in comp_p.valuesdict().iteritems():
                    comp_p[key].value = pars['l' + str(i) + '_' + key].value
    
                ax.plot( np.sort(vdat.value), comp_mod.eval(comp_p, x=np.sort(vdat.value)) )
    

    if fit_model == 'Hb': 
 
        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_5007_n_center'].value
        p['sigma'].value = pars['oiii_5007_n_sigma'].value
        p['amplitude'].value = pars['oiii_5007_n_amplitude'].value

        ax.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_4959_n_center'].value
        p['sigma'].value = pars['oiii_4959_n_sigma'].value
        p['amplitude'].value = pars['oiii_4959_n_amplitude'].value

        ax.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='blue',
                 linestyle='--')        

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_5007_b_center'].value
        p['sigma'].value = pars['oiii_5007_b_sigma'].value
        p['amplitude'].value = pars['oiii_5007_b_amplitude'].value

        ax.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='-')       

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['oiii_4959_b_center'].value
        p['sigma'].value = pars['oiii_4959_b_sigma'].value
        p['amplitude'].value = pars['oiii_4959_b_amplitude'].value   

        ax.plot(np.sort(vdat.value), 
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
    
            ax.plot(np.sort(vdat.value), 
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
          
        
        ax.plot(np.sort(vdat.value), 
                 mod_broad_hb.eval(pars_broad_hb, x=np.sort(vdat.value)),
                 c='black',
                 linestyle='--',
                 lw=2)                          
    
        if hb_narrow is True: 

            g = GaussianModel()
            p = g.make_params()
    
            p['center'].value = pars['hb_n_center'].value
            p['sigma'].value = pars['hb_n_sigma'].value
            p['amplitude'].value = pars['hb_n_amplitude'].value
    
            ax.plot(np.sort(vdat.value), 
                     g.eval(p, x=np.sort(vdat.value)),
                     c='orange',
                     linestyle='--')                

    if fit_model == 'Ha': 


        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['ha_n_center'].value
        p['sigma'].value = pars['ha_n_sigma'].value
        p['amplitude'].value = pars['ha_n_amplitude'].value

        ax.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['nii_6548_n_center'].value
        p['sigma'].value = pars['nii_6548_n_sigma'].value
        p['amplitude'].value = pars['nii_6548_n_amplitude'].value

        ax.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['nii_6584_n_center'].value
        p['sigma'].value = pars['nii_6584_n_sigma'].value
        p['amplitude'].value = pars['nii_6584_n_amplitude'].value

        ax.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')

        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['sii_6717_n_center'].value
        p['sigma'].value = pars['sii_6717_n_sigma'].value
        p['amplitude'].value = pars['sii_6717_n_amplitude'].value

        ax.plot(np.sort(vdat.value), 
                 g.eval(p, x=np.sort(vdat.value)),
                 c='red',
                 linestyle='--')      
                 
        g = GaussianModel()
        p = g.make_params()

        p['center'].value = pars['sii_6731_n_center'].value
        p['sigma'].value = pars['sii_6731_n_sigma'].value
        p['amplitude'].value = pars['sii_6731_n_amplitude'].value

        ax.plot(np.sort(vdat.value), 
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
    
            ax.plot(np.sort(vdat.value), 
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
          
        
        ax.plot(np.sort(vdat.value), 
                 mod_broad_ha.eval(pars_broad_ha, x=np.sort(vdat.value)),
                 c='black',
                 linestyle='--',
                 lw=2)          

    if fit_model == 'GaussHermite':

        i = 0 
        
        while i <= gh_order:
            
            flux_com = gausshermite_component(vdat.value/xscale, 
                                              pars['amp{}'.format(i)].value, 
                                              pars['sig{}'.format(i)].value, 
                                              pars['cen{}'.format(i)].value, 
                                              i)
    
            ax.plot(vdat.value, flux_com, color='gold', lw=1)
             
            i += 1                 
  
    fig.tight_layout()

    if plot_savefig is not None:
        mpld3.save_html(fig, os.path.join(save_dir, 'plot.html')) 
      
    plt.close()

    return None

def plot_continum(xdat_cont, 
                  ydat_cont, 
                  yerr_cont, 
                  wav, 
                  flux, 
                  err, 
                  subtract_fe, 
                  bkgdpars, 
                  bkgdmod, 
                  sp_fe, 
                  continuum_region, 
                  maskout, 
                  w0, 
                  reject_outliers, 
                  wav_array_blue, 
                  flux_array_blue, 
                  wav_array_red, 
                  flux_array_red, 
                  verbose):

  
    """
    Plotting continuum / iron fit
    """

    xdat_cont_plot = ma.getdata(xdat_cont[~ma.getmaskarray(xdat_cont)]).value
    ydat_cont_plot = ma.getdata(ydat_cont[~ma.getmaskarray(ydat_cont)])
    yerr_cont_plot = ma.getdata(yerr_cont[~ma.getmaskarray(yerr_cont)])

    fig, axs = plt.subplots(4, 1, figsize=(20,20))

    axs[0].errorbar(wav.value, 
                    flux, 
                    yerr=err, 
                    linestyle='', 
                    alpha=0.5,
                    color='grey')

    axs[1].errorbar(wav.value, 
                    median_filter(flux, 51), 
                    color='grey')

    xdat_plotting = np.arange(xdat_cont_plot.min(), xdat_cont_plot.max(), 1)

    if subtract_fe is True: 

        axs[0].plot(xdat_plotting, 
                    resid(params=bkgdpars, 
                          x=xdat_plotting, 
                          model=bkgdmod, 
                          sp_fe=sp_fe), 
                    color='black', 
                    lw=2)

        axs[1].plot(xdat_plotting, 
                    resid(params=bkgdpars, 
                          x=xdat_plotting, 
                          model=bkgdmod, 
                          sp_fe=sp_fe), 
                    color='black', 
                    lw=2)

        gauss = Gaussian1DKernel(stddev=bkgdpars['fe_sd'].value) 

        fe_flux = convolve(sp_fe.fl, gauss)
        fe_flux = np.roll(fe_flux, int(bkgdpars['fe_shift'].value))
        f = interp1d(sp_fe.wa, fe_flux) 
        fe_flux = f(xdat_cont_plot)   

        axs[0].plot(xdat_cont_plot, 
                    bkgdpars['fe_norm'].value * fe_flux, 
                    color='red')   

        
        amp = bkgdpars['amplitude'].value * 5000.0**(-bkgdpars['exponent'].value)    
        
        axs[0].plot(xdat_cont_plot, 
                    amp*xdat_cont_plot**bkgdpars['exponent'].value, 
                    color='blue')   


        fe_flux = np.roll(sp_fe.fl, int(bkgdpars['fe_shift'].value))
       
        axs[1].plot(sp_fe.wa, 
                    bkgdpars['fe_norm'].value * fe_flux, 
                    color='red')  


    if subtract_fe is False: 

        axs[0].plot(xdat_plotting, 
                    resid(params=bkgdpars, 
                          x=xdat_plotting, 
                          model=bkgdmod), 
                    color='black', 
                    lw=2)

        axs[1].plot(xdat_plotting, 
                    resid(params=bkgdpars, 
                          x=xdat_plotting, 
                          model=bkgdmod),
                    color='black', 
                    lw=2)

        bkgdpars_tmp = bkgdpars.copy() 


        axs[1].plot(xdat_plotting, 
                    resid(params=bkgdpars, 
                          x=xdat_plotting, 
                          model=bkgdmod),
                    color='black', 
                    lw=2)
    


    axs[0].set_xlim(xdat_cont_plot.min() - 50.0, xdat_cont_plot.max() + 50.0)
    axs[1].set_xlim(axs[0].get_xlim())

    if subtract_fe is True: 
    
        func_vals = resid(params=bkgdpars, 
                          x=xdat_cont_plot, 
                          model=bkgdmod, 
                          sp_fe=sp_fe)

    if subtract_fe is False: 

        func_vals = resid(params=bkgdpars, 
                          x=xdat_cont_plot, 
                          model=bkgdmod)

    axs[0].set_ylim(np.median(ydat_cont_plot) - 3.0*np.std(ydat_cont_plot), np.median(ydat_cont_plot) + 3.0*np.std(ydat_cont_plot))
    axs[1].set_ylim(axs[0].get_ylim())
    
    xdat_masking = np.arange(wav.min().value, wav.max().value, 0.05)*(u.AA)
    vdat_masking = wave2doppler(xdat_masking, w0)
    
    mask = (xdat_masking.value < continuum_region[0][0].value) | \
           ((xdat_masking.value > continuum_region[0][1].value) &  (xdat_masking.value < continuum_region[1][0].value))  | \
           (xdat_masking.value > continuum_region[1][1].value)
    
    if maskout is not None:
        for item in maskout:
            mask = mask | ((vdat_masking.value > item.value[0]) & (vdat_masking.value < item.value[1]))
    
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

    if subtract_fe is True: 
    
        axs[2].scatter(xdat_cont_plot, 
                      (ydat_cont_plot - resid(params=bkgdpars, 
                                              x=xdat_cont_plot, 
                                              model=bkgdmod, 
                                              sp_fe=sp_fe)) / yerr_cont_plot, 
                      edgecolor='None', 
                      s=15, 
                      facecolor='black')

    if subtract_fe is False: 

        axs[2].scatter(xdat_cont_plot, 
                      (ydat_cont_plot - resid(params=bkgdpars, 
                                              x=xdat_cont_plot, 
                                              model=bkgdmod)) / yerr_cont_plot, 
                      edgecolor='None', 
                      s=15, 
                      facecolor='black')

    
    axs[2].axhline(0.0, color='black', linestyle='--')
    axs[2].set_xlim(axs[0].get_xlim())

    axs[3].plot(wav.value, flux, color='grey')

    if subtract_fe is True:
    
        axs[3].plot(xdat_cont_plot, 
                    resid(params=bkgdpars, 
                          x=xdat_cont_plot, 
                          model=bkgdmod, 
                          sp_fe=sp_fe), 
                    color='black', 
                    lw=2)

    if subtract_fe is False: 

        axs[3].plot(xdat_cont_plot, 
                    resid(params=bkgdpars, 
                          x=xdat_cont_plot, 
                          model=bkgdmod), 
                    color='black', 
                    lw=2)

    if reject_outliers: 

        mask = ma.getmaskarray(wav_array_blue).flatten() 

        mask[ma.getdata(wav_array_blue.flatten()).value < continuum_region[0][0].value] = False
        mask[ma.getdata(wav_array_blue.flatten()).value > continuum_region[0][1].value] = False

        xdat_cont_outliers = ma.masked_where(~mask, ma.getdata(wav_array_blue).flatten())
        ydat_cont_outliers = ma.masked_where(~mask, ma.getdata(flux_array_blue).flatten())

        axs[0].scatter(xdat_cont_outliers,
                       ydat_cont_outliers,
                       color='firebrick', 
                       s=40, 
                       marker='x') 

        mask = ma.getmaskarray(wav_array_red).flatten() 

        mask[ma.getdata(wav_array_red.flatten()).value < continuum_region[1][0].value] = False
        mask[ma.getdata(wav_array_red.flatten()).value > continuum_region[1][1].value] = False

        xdat_cont_outliers = ma.masked_where(~mask, ma.getdata(wav_array_red).flatten())
        ydat_cont_outliers = ma.masked_where(~mask, ma.getdata(flux_array_red).flatten())

        axs[0].scatter(xdat_cont_outliers,
                       ydat_cont_outliers,
                       color='firebrick', 
                       s=40, 
                       marker='x') 

    if verbose:

        global coords
        coords = [] 
        cid = fig.canvas.mpl_connect('button_press_event', lambda x: onclick2(x, w0=w0))

        plt.show(1)

    plt.close() 

    return None 


def fit1(obj,
         n_samples, 
         plot_title, 
         verbose, 
         mono_lum_wav, 
         spec_norm, 
         z):


    k = obj[0]
    x = obj[1]
    y = obj[2]
    flux_array_fit_k = obj[3]
    flux_array_plot_k = obj[4] 
    wav_array_fit_k = obj[5] 
    wav_array_plot_k = obj[6] 

    bkgdmod = Model(PLModel, 
                    param_names=['amplitude','exponent'], 
                    independent_vars=['x']) 

    bkgdpars = bkgdmod.make_params()

    bkgdpars['exponent'].value = 1.0
    bkgdpars['amplitude'].value = 1.0 
    
    out = minimize(resid,
                   bkgdpars,
                   kws={'x':x, 
                        'model':bkgdmod, 
                        'data':y},
                   method='leastsq')

    flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] = flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] - resid(params=out.params, x=ma.getdata(wav_array_fit_k[~ma.getmaskarray(wav_array_fit_k)]).value, model=bkgdmod)
    flux_array_plot_k[~ma.getmaskarray(wav_array_plot_k)] = flux_array_plot_k[~ma.getmaskarray(wav_array_plot_k)] - resid(params=out.params, x=ma.getdata(wav_array_plot_k[~ma.getmaskarray(wav_array_plot_k)]).value, model=bkgdmod)                  

    if verbose:
        if n_samples == 1:
            print out.message  
            print fit_report(out.params)

    ####################################################################################################################
    """
    Calculate flux at wavelength mono_lum_wav
    """
    # Calculate just power-law continuum (no Fe)
    cont_mod = Model(PLModel, 
                     param_names=['amplitude','exponent'], 
                     independent_vars=['x']) 

    cont_pars = cont_mod.make_params()
    cont_pars['exponent'].value = out.params['exponent'].value
    cont_pars['amplitude'].value = out.params['amplitude'].value  

    mono_flux = resid(params=cont_pars, 
                      x=np.array([mono_lum_wav.value]), 
                      model=cont_mod)[0]
    
    mono_flux = mono_flux / spec_norm
    mono_flux = mono_flux * (u.erg / u.cm / u.cm / u.s / u.AA)
    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)

    mono_lum = mono_flux * (1.0 + z) * 4.0 * math.pi * lumdist**2 * mono_lum_wav

    eqw_fe = 0.0 

    if n_samples > 1: 
        print plot_title, k

    return (flux_array_fit_k, flux_array_plot_k, mono_lum, eqw_fe, out.params)

def fit2(obj, 
         n_samples, 
         subtract_fe, 
         home_dir, 
         fe_FWHM, 
         fe_FWHM_vary, 
         spec_norm, 
         mono_lum_wav, 
         z,
         wav, 
         flux, 
         err, 
         verbose, 
         wav_array_blue, 
         flux_array_blue, 
         wav_array_red, 
         flux_array_red, 
         reject_outliers, 
         w0, 
         maskout, 
         continuum_region,
         plot_title,
         plot): 

    k = obj[0]
    x = obj[1]
    y = obj[2]
    er = obj[3]
    flux_array_fit_k = obj[4]
    flux_array_plot_k = obj[5] 
    wav_array_fit_k = obj[6] 
    wav_array_plot_k = obj[7] 

    if len(ma.getdata(x[~ma.getmaskarray(x)]).value) == 0: 
        if verbose:
            print 'No flux in continuum region'

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

        fname = os.path.join(home_dir, 'SpectraTools/irontemplate.dat')
        fe_wav, fe_flux = np.genfromtxt(fname, unpack=True)
    
        fe_flux = fe_flux / np.median(fe_flux)
    
        sp_fe = spec.Spectrum(wa=10**fe_wav, fl=fe_flux)

        bkgdmod = Model(PseudoContinuum, 
                        param_names=['amplitude',
                                     'exponent',
                                     'fe_norm',
                                     'fe_sd',
                                     'fe_shift'], 
                        independent_vars=['x']) 
        
        bkgdpars = bkgdmod.make_params() 
        
        fe_sigma = fe_FWHM / 2.35 
        fe_pix = fe_sigma / (sp_fe.dv * (u.km/u.s))

        bkgdpars['fe_sd'].vary = fe_FWHM_vary

        bkgdpars['fe_norm'].min = 0.0
        bkgdpars['fe_sd'].min = 2000.0 / 2.35 / sp_fe.dv
        bkgdpars['fe_sd'].max = 12000.0 / 2.35 / sp_fe.dv

        bkgdpars['fe_shift'].min = -20.0
        bkgdpars['fe_shift'].max = 20.0

        # I think that constraining the power-law slope to be negative
        # helps break some of the degeneracy between the fe template
        # and the continuum. I don't want to change any of my fits
        # so for now just do this for the pseudo-continuum fit. 

        # Seems to break if max is set to 0.0

        # if pseudo_continuum_fit:

        #     bkgdpars['exponent'].max = 0.5 

        bkgdpars['fe_sd'].value = fe_pix.value  
        bkgdpars['exponent'].value = -1.0
        bkgdpars['amplitude'].value = 1.0
        bkgdpars['fe_norm'].value = 0.05 
        bkgdpars['fe_shift'].value = 0.0

        out = minimize(resid,
                       bkgdpars,
                       kws={'x':ma.getdata(x[~ma.getmaskarray(x)]).value, 
                            'model':bkgdmod, 
                            'data':ma.getdata(y[~ma.getmaskarray(y)]), 
                            'sigma':ma.getdata(er[~ma.getmaskarray(er)]), 
                            'sp_fe':sp_fe},
                       method='powell',
                       options={'maxiter':1e6, 'maxfev':1e6})

   
        flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] = flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] - resid(params=out.params, x=ma.getdata(wav_array_fit_k[~ma.getmaskarray(wav_array_fit_k)]).value, model=bkgdmod, sp_fe=sp_fe)
        flux_array_plot_k[~ma.getmaskarray(wav_array_plot_k)] = flux_array_plot_k[~ma.getmaskarray(wav_array_plot_k)] - resid(params=out.params, x=ma.getdata(wav_array_plot_k[~ma.getmaskarray(wav_array_plot_k)]).value, model=bkgdmod, sp_fe=sp_fe)                  

        ##############################################################################
        
        """
        Calculate EQW of Fe between 4435 and 4684A - same as Shen (2016)
        """
        fe_fl = PseudoContinuum(np.arange(4434,4684,1), 
                                0.0,
                                0.0,
                                out.params['fe_norm'].value,
                                out.params['fe_sd'].value,
                                out.params['fe_shift'].value,
                                sp_fe)

        fe_fl = fe_fl / spec_norm 
        fe_fl = fe_fl * (u.erg / u.cm / u.cm / u.s / u.AA) 

        bg_fl = PseudoContinuum(np.arange(4434,4684,1), 
                                out.params['amplitude'].value,
                                out.params['exponent'].value,
                                0.0,
                                out.params['fe_sd'].value,
                                out.params['fe_shift'].value,
                                sp_fe)      

        bg_fl = bg_fl / spec_norm 
        bg_fl = bg_fl * (u.erg / u.cm / u.cm / u.s / u.AA) 
    
        f = (fe_fl + bg_fl) / bg_fl
        eqw_fe = np.nansum((f[:-1] - 1.0) * 1.0) 

        if verbose:
            print 'Fe EW: {}'.format(eqw_fe)  

    if subtract_fe is False:

        sp_fe = None

        bkgdmod = Model(PLModel, 
                        param_names=['amplitude','exponent'], 
                        independent_vars=['x']) 

        bkgdpars = bkgdmod.make_params()    
        bkgdpars['exponent'].value = 1.0
        bkgdpars['amplitude'].value = 1.0 

        out = minimize(resid,
                       bkgdpars,
                       kws={'x':ma.getdata(x[~ma.getmaskarray(x)]).value, 
                            'model':bkgdmod, 
                            'data':ma.getdata(y[~ma.getmaskarray(y)]),
                            'sigma':ma.getdata(er[~ma.getmaskarray(er)])},  
                       method='leastsq') 


        flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] = flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] - resid(params=out.params, x=ma.getdata(wav_array_fit_k[~ma.getmaskarray(wav_array_fit_k)]).value, model=bkgdmod)
        flux_array_plot_k[~ma.getmaskarray(wav_array_plot_k)] = flux_array_plot_k[~ma.getmaskarray(wav_array_plot_k)] - resid(params=out.params, x=ma.getdata(wav_array_plot_k[~ma.getmaskarray(wav_array_plot_k)]).value, model=bkgdmod)                  
    
        eqw_fe = 0.0

    ####################################################################################################################
    """
    Calculate flux at wavelength mono_lum_wav
    """

    # Calculate just power-law continuum (no Fe)
    cont_mod = Model(PLModel, 
                     param_names=['amplitude','exponent'], 
                     independent_vars=['x']) 


    cont_pars = cont_mod.make_params()
    cont_pars['exponent'].value = out.params['exponent'].value
    cont_pars['amplitude'].value = out.params['amplitude'].value  

    mono_flux = resid(params=cont_pars, 
                      x=np.array([mono_lum_wav.value]), 
                      model=cont_mod)[0]

    mono_flux = mono_flux / spec_norm

    mono_flux = mono_flux * (u.erg / u.cm / u.cm / u.s / u.AA)

    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)

    mono_lum = mono_flux * (1.0 + z) * 4.0 * math.pi * lumdist**2 * mono_lum_wav

    if n_samples > 1: 
        print plot_title, k 


    ######################################################################################################################

    if n_samples == 1:  

        if plot:

            plot_continum(x, 
                          y, 
                          er,
                          wav, 
                          flux, 
                          err, 
                          subtract_fe, 
                          out.params, 
                          bkgdmod, 
                          sp_fe, 
                          continuum_region, 
                          maskout, 
                          w0, 
                          reject_outliers, 
                          wav_array_blue, 
                          flux_array_blue, 
                          wav_array_red, 
                          flux_array_red, 
                          verbose)


        if verbose:

            print colored(out.message, 'red'), colored('Number of function evaluations: {}'.format(out.nfev), 'red') 
            print fit_report(out.params)
    
            if subtract_fe is True:
                print 'Fe FWHM = {0:.1f} km/s (initial = {1:.1f} km/s)'.format(out.params['fe_sd'].value * 2.35 * sp_fe.dv, out.params['fe_sd'].init_value * 2.35 * sp_fe.dv)            
            

    # if pseudo_continuum_fit:

    #     """
    #     Only fitting PseudoContinuum - BROKEN SINCE MADE IN TO A FUNCTION! 
    #     """

    #     fit_out = {'name':plot_title,
    #                'monolum':np.log10(mono_lum.value),
    #                'fe_ew':eqw_fe} 

    #     if n_samples == 1:       

    #         return fit_out

    #     else: 
       
    #         if save_dir is not None:
    #             df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 

    #         return None 

    return (flux_array_fit_k, flux_array_plot_k, mono_lum, eqw_fe, out.params)


def fit3(obj, 
         fit_model, 
         nGaussians, 
         gh_order, 
         fitting_region, 
         ha_narrow_fwhm, 
         ha_narrow_vary, 
         ha_narrow_voff, 
         w0, 
         hb_narrow, 
         n_samples, 
         plot, 
         continuum_region, 
         fitting_method, 
         save_dir, 
         subtract_fe, 
         home_dir,
         z, 
         line_region, 
         spec_norm, 
         plot_title,
         verbose,
         spec_dv,
         wav, 
         flux, 
         err,
         mono_lum_wav,
         emission_line,
         plot_savefig,
         maskout,
         plot_region,
         mask_negflux,
         reject_outliers,
         reject_width,
         reject_sigma,
         fig,
         fit,
         residuals,
         fix_broad_peaks,
         oiii_broad_off):


    k = obj[0]
    x = obj[1]
    y = obj[2]
    er = obj[3]
    wav_array_plot_k = obj[4]
    vdat_array_plot_k = obj[5]
    flux_array_plot_k = obj[6]
    err_array_plot_k = obj[7]
    bkgdpars_k = obj[8]
    mono_lum_k = obj[9]
    eqw_fe_k = obj[10]
    snr_k = obj[11]

    if subtract_fe is True:

        bkgdmod = Model(PseudoContinuum, 
                        param_names=['amplitude',
                                     'exponent',
                                     'fe_norm',
                                     'fe_sd',
                                     'fe_shift'], 
                        independent_vars=['x']) 

        fname = os.path.join(home_dir, 'SpectraTools/irontemplate.dat')
        fe_wav, fe_flux = np.genfromtxt(fname, unpack=True)
        fe_flux = fe_flux / np.median(fe_flux)
        sp_fe = spec.Spectrum(wa=10**fe_wav, fl=fe_flux)


    if subtract_fe is False:                

        bkgdmod = Model(PLModel, 
                        param_names=['amplitude','exponent'], 
                        independent_vars=['x'])

    if fit_model == 'MultiGauss':

        xscale = 1.0 

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
            pars += gmod.guess(y[~ma.getmaskarray(y)], x=ma.getdata(x[~ma.getmaskarray(x)]).value)
               
        for i in range(nGaussians):
            pars['g{}_center'.format(i)].value = 0.0
            pars['g{}_center'.format(i)].min = -2000.0
            pars['g{}_center'.format(i)].max = 2000.0 
            pars['g{}_amplitude'.format(i)].min = 0.0
            # pars['g{}_sigma'.format(i)].min = 1000.0 # sometimes might be better if this is relaxed 
            # pars['g{}_sigma'.format(i)].max = 10000.0 
            # pars['g0_center'].set(expr='g1_center')
            pars['g{}_amplitude'.format(i)].value =  1.0 
            pars['g{}_sigma'.format(i)].value = 1200.0 

        if fix_broad_peaks:

                for i in range(1, nGaussians):
                    pars['g{}_center'.format(i)].set(expr = 'g0_center')


    elif fit_model == 'GaussHermite':

        # Calculate mean and variance

        """
        Because p isn't a real probability distribution we sometimes get negative 
        variances if a lot of the flux is negative. 
        Therefore confine this to only work on the bit of the 
        spectrum that is positive
        """

        ydat = ma.getdata(y[~ma.getmaskarray(y)])
        vdat = ma.getdata(x[~ma.getmaskarray(x)])

        p = ydat[ydat > 0.0] / np.sum(ydat[ydat > 0.0])
        m = np.sum(vdat[ydat > 0.0] * p)
        v = np.sum(p * (vdat[ydat > 0.0] - m)**2)
        xscale = np.sqrt(v).value 

        if xscale == 0.0:

            """ 
            Breaks when I'm calculating errors, but the fit is very bad. 
            Just exit and return empty list - 
            """

            fit_out = {'name':plot_title, 
                       'fwhm':np.nan,
                       'fwhm_1':np.nan,
                       'fwhm_2':np.nan,
                       'fwhm_3':np.nan,
                       'sigma':np.nan,
                       'median':np.nan,
                       'cen':np.nan,
                       'cen_1':np.nan,
                       'cen_2':np.nan,
                       'cen_3':np.nan,
                       'eqw':np.nan,
                       'broad_lum':np.nan,
                       'peak':np.nan, 
                       'amplitude_1':np.nan,
                       'amplitude_2':np.nan,
                       'amplitude_3':np.nan,
                       'narrow_fwhm':np.nan,
                       'narrow_lum':np.nan,
                       'narrow_voff':np.nan, 
                       'oiii_5007_eqw':np.nan,
                       'oiii_5007_lum':np.nan,
                       'oiii_5007_n_lum':np.nan,
                       'oiii_5007_b_lum':np.nan,
                       'oiii_fwhm':np.nan,
                       'oiii_n_fwhm':np.nan,
                       'oiii_b_fwhm':np.nan,
                       'oiii_5007_b_voff':np.nan,
                       'oiii_5007_blueshift':np.nan,
                       'redchi':np.nan,
                       'snr':np.nan,
                       'dv':np.nan, 
                       'monolum':np.nan,
                       'fe_ew':np.nan}

            return fit_out 

        param_names = []

        for i in range(gh_order + 1):
            
            param_names.append('amp{}'.format(i))
            param_names.append('sig{}'.format(i))
            param_names.append('cen{}'.format(i))

        if gh_order == 0: 
 
            mod = Model(gausshermite_0, independent_vars=['x'], param_names=param_names) 
    
        if gh_order == 1: 
 
            mod = Model(gausshermite_1, independent_vars=['x'], param_names=param_names) 
     
        if gh_order == 2: 
 
            mod = Model(gausshermite_2, independent_vars=['x'], param_names=param_names) 
     
        if gh_order == 3: 
 
            mod = Model(gausshermite_3, independent_vars=['x'], param_names=param_names) 
     
        if gh_order == 4: 
 
            mod = Model(gausshermite_4, independent_vars=['x'], param_names=param_names) 
     
        if gh_order == 5: 
 
            mod = Model(gausshermite_5, independent_vars=['x'], param_names=param_names) 

        if gh_order == 6: 
 
            mod = Model(gausshermite_6, independent_vars=['x'], param_names=param_names) 

        pars = mod.make_params()

        for i in range(gh_order + 1):

            pars['amp{}'.format(i)].value = 1.0
            pars['sig{}'.format(i)].value = 1.0
            pars['cen{}'.format(i)].value = 0.0
 
            # remember that wav_fitting_array mask includes the background windows, 
            # so don't use this 
            # already converted fitting_region in to units of wavelength. 

            pars['cen{}'.format(i)].min = wave2doppler(fitting_region[0], w0).value / xscale
            pars['cen{}'.format(i)].max = wave2doppler(fitting_region[1], w0).value / xscale

            pars['sig{}'.format(i)].min = 0.1

            pars['amp{}'.format(i)].min = 0.0    

        
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

        xscale = 1.0 

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

        for i in range(nGaussians): 
            pars['ha_b_{}_center'.format(i)].value = (-1)**i * 100.0  
            pars['ha_b_{}_center'.format(i)].min = -2000.0  
            pars['ha_b_{}_center'.format(i)].max = 2000.0  

        if fix_broad_peaks:

            if nGaussians == 2:
                pars['ha_b_1_center'].set(expr = 'ha_b_0_center')
            elif nGaussians == 3:
                pars['ha_b_2_center'].set(expr = 'ha_b_0_center')

        pars['nii_6548_n_sigma'].value = ha_narrow_fwhm / 2.35 
        pars['nii_6584_n_sigma'].value = ha_narrow_fwhm / 2.35
        pars['sii_6717_n_sigma'].value = ha_narrow_fwhm / 2.35
        pars['sii_6731_n_sigma'].value = ha_narrow_fwhm / 2.35
        pars['ha_n_sigma'].value = ha_narrow_fwhm / 2.35 

        if ha_narrow_vary is False:

            pars['nii_6548_n_sigma'].vary = False
            pars['nii_6584_n_sigma'].vary = False
            pars['sii_6717_n_sigma'].vary = False
            pars['sii_6731_n_sigma'].vary = False
            pars['ha_n_sigma'].vary = False

        else:

            pars['ha_n_sigma'].max = 1200.0 / 2.35 
            pars['ha_n_sigma'].min = 400.0 / 2.35 
            pars['nii_6548_n_sigma'].set(expr='ha_n_sigma')
            pars['nii_6584_n_sigma'].set(expr='ha_n_sigma')
            pars['sii_6717_n_sigma'].set(expr='ha_n_sigma')
            pars['sii_6731_n_sigma'].set(expr='ha_n_sigma')
   
        for i in range(nGaussians): 
            pars['ha_b_{}_sigma'.format(i)].value = 1200.0

        for i in range(nGaussians): 
            pars['ha_b_{}_sigma'.format(i)].min = 1200.0 / 2.35

        pars['nii_6548_n_amplitude'].min = 0.0
        pars['nii_6584_n_amplitude'].min = 0.0
        pars['sii_6717_n_amplitude'].min = 0.0 
        pars['sii_6731_n_amplitude'].min = 0.0 
        pars['ha_n_amplitude'].min = 0.0 
        for i in range(nGaussians): 
            pars['ha_b_{}_amplitude'.format(i)].min = 0.0  
        
        pars['ha_n_center'].value = ha_narrow_voff 
        pars['sii_6731_n_center'].value = ha_narrow_voff + wave2doppler(6731*u.AA, w0).value 
        pars['sii_6717_n_center'].value = ha_narrow_voff + wave2doppler(6717*u.AA, w0).value 
        pars['nii_6548_n_center'].value = ha_narrow_voff + wave2doppler(6548*u.AA, w0).value 
        pars['nii_6584_n_center'].value  = ha_narrow_voff + wave2doppler(6584*u.AA, w0).value 

        if ha_narrow_vary is False:

            pars['ha_n_center'].vary = False 
            pars['sii_6731_n_center'].vary = False 
            pars['sii_6717_n_center'].vary = False 
            pars['nii_6548_n_center'].vary = False 
            pars['nii_6584_n_center'].vary = False  

        else: 

            pars['ha_n_center'].min = -500.0
            pars['ha_n_center'].max = 500.0
    
            pars['sii_6731_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6731*u.AA, w0).value))
            pars['sii_6717_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6717*u.AA, w0).value))
            pars['nii_6548_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6548*u.AA, w0).value))
            pars['nii_6584_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6584*u.AA, w0).value))

        pars['nii_6548_n_amplitude'].set(expr='0.333*nii_6584_n_amplitude')

        # if si unconstrained to stop it giving something silly. 
        wmax = doppler2wave(np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value/xscale).max()*(u.km/u.s), w0)
        if wmax < 6717*u.AA:
            pars['sii_6717_n_amplitude'].value = 0.0
            pars['sii_6717_n_amplitude'].vary = False 
            pars['sii_6731_n_amplitude'].value = 0.0
            pars['sii_6731_n_amplitude'].vary = False 
            

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

        xscale = 1.0 

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

        pars['oiii_4959_n_amplitude'].min = 0.0
        pars['oiii_5007_n_amplitude'].min = 0.0
        pars['oiii_4959_b_amplitude'].min = 0.0 
        pars['oiii_5007_b_amplitude'].min = 0.0 
        if hb_narrow is True:      
            pars['hb_n_amplitude'].min = 0.0 
        for i in range(nGaussians): 
            pars['hb_b_{}_amplitude'.format(i)].min = 0.0  

        # Make sure we don't have all broad and no narrow component 
        # pars.add('oiii_5007_b_height_delta')
        # pars['oiii_5007_b_height_delta'].value = 0.5 
        # pars['oiii_5007_b_height_delta'].max = 2.0
        # pars['oiii_5007_b_height_delta'].min = 0.0
        # pars['oiii_5007_b_amplitude'].set(expr='oiii_5007_n_amplitude * oiii_5007_b_height_delta')
        

        # pars['oiii_5007_b_amplitude'].set(expr='((oiii_5007_n_amplitude/oiii_5007_n_sigma)*oiii_5007_b_height_delta)*oiii_5007_b_sigma')

        # Set 3:1 [OIII] peak ratio  
        pars['oiii_4959_n_amplitude'].set(expr='0.3333*oiii_5007_n_amplitude')

        # Also set 3:1 [OIII] peak ratio for broad components 
        pars['oiii_4959_b_amplitude'].set(expr='0.3333*oiii_5007_b_amplitude')


        # ---------------------------------------------------------------

        pars['oiii_4959_n_center'].value = wave2doppler(4960.295*u.AA, w0).value 
        pars['oiii_5007_n_center'].value = wave2doppler(5008.239*u.AA, w0).value 
        pars['oiii_4959_b_center'].value = wave2doppler(4960.295*u.AA, w0).value 
        pars['oiii_5007_b_center'].value = wave2doppler(5008.239*u.AA, w0).value 
        if hb_narrow is True: 
            pars['hb_n_center'].value = 0.0    
        
        for i in range(nGaussians): 
            pars['hb_b_{}_center'.format(i)].value = 0.0  
            pars['hb_b_{}_center'.format(i)].min = -2000.0  
            pars['hb_b_{}_center'.format(i)].max = 2000.0  

        if fix_broad_peaks:

            if nGaussians == 2:
                pars['hb_b_1_center'].set(expr = 'hb_b_0_center')
            elif nGaussians == 3:
                pars['hb_b_2_center'].set(expr = 'hb_b_0_center')


        pars['oiii_5007_n_center'].min = wave2doppler(5008.239*u.AA, w0).value - 3000.0
        pars['oiii_5007_n_center'].max = wave2doppler(5008.239*u.AA, w0).value + 3000.0
        
        if hb_narrow is True: 
            pars['hb_n_center'].set(expr = 'oiii_5007_n_center-{}'.format(wave2doppler(5008.239*u.AA, w0).value)) 
        
        pars['oiii_4959_n_center'].set(expr = 'oiii_5007_n_center+{}'.format(wave2doppler(4960.295*u.AA, w0).value - wave2doppler(5008.239*u.AA, w0).value))


        pars.add('oiii_5007_b_center_delta') 
        pars['oiii_5007_b_center_delta'].value = 500.0 
        pars['oiii_5007_b_center_delta'].min = 0.0
        pars['oiii_5007_b_center_delta'].max = 1500.0
        pars['oiii_5007_b_center'].set(expr='oiii_5007_n_center-oiii_5007_b_center_delta')
        pars['oiii_4959_b_center'].set(expr='oiii_4959_n_center-oiii_5007_b_center_delta')

        # Set broad components of OIII to have fixed relative velocity
        # pars['oiii_4959_b_center'].set(expr = 'oiii_5007_b_center+{}'.format(wave2doppler(4960.295*u.AA, w0).value - wave2doppler(5008.239*u.AA, w0).value))


        #----------------------------------------------------------------------

        pars['oiii_4959_n_sigma'].value = 500.0 / 2.35
        pars['oiii_5007_n_sigma'].value = 500.0 / 2.35
        pars['oiii_4959_b_sigma'].value = 1000.0 / 2.35
        pars['oiii_5007_b_sigma'].value = 1000.0 / 2.35
        if hb_narrow is True: 
            pars['hb_n_sigma'].value = 500.0 / 2.35
        for i in range(nGaussians): 
            pars['hb_b_{}_sigma'.format(i)].value = 1200.0 

        for i in range(nGaussians): 
            pars['hb_b_{}_sigma'.format(i)].min = 1000.0 / 2.35
    
        pars['oiii_5007_b_sigma'].max = 1200.0 / 2.35 
        pars['oiii_5007_b_sigma'].min = 100.0 / 2.35 
        pars['oiii_4959_b_sigma'].set(expr='oiii_5007_b_sigma')

        pars['oiii_5007_n_sigma'].min = 100.0 / 2.35 
        pars['oiii_5007_n_sigma'].max = 1200.0 / 2.35 

        # make sure broad component of oiii is broader than narrow component 
        # pars.add('oiii_5007_n_sigma_delta')
        # pars['oiii_5007_n_sigma_delta'].value = 0.5
        # pars['oiii_5007_n_sigma_delta'].max = 0.9
        # pars['oiii_5007_n_sigma'].set(expr='oiii_5007_b_sigma*oiii_5007_n_sigma_delta', min=100.0/2.35, max=1200.0/2.35)


        pars['oiii_4959_n_sigma'].set(expr='oiii_5007_n_sigma')

        if hb_narrow is True: 
            pars['hb_n_sigma'].set(expr='oiii_5007_n_sigma')


        if oiii_broad_off:
            pars['oiii_4959_b_amplitude'].set(value=0.0, vary=False)
            pars['oiii_5007_b_amplitude'].set(value=0.0, vary=False)
            pars['oiii_4959_b_sigma'].set(value=1200.0/2.35, vary=False)
            pars['oiii_5007_b_sigma'].set(value=1200.0/2.35, vary=False)
            pars['oiii_4959_b_center'].set(value=wave2doppler(4960.295*u.AA, w0).value, vary=False)
            pars['oiii_5007_b_center'].set(value=wave2doppler(5008.239*u.AA, w0).value, vary=False)




    elif fit_model == 'siiv':

        xscale = 1.0 

        mod = GaussianModel(prefix='g0_') + GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_') + GaussianModel(prefix='g3_') 
        
        pars = mod.make_params() 

        pars['g0_amplitude'].value = 1.0
        pars['g1_amplitude'].value = 1.0
        pars['g2_amplitude'].value = 1.0
        pars['g3_amplitude'].value = 1.0

        pars['g0_amplitude'].min = 0.0
        pars['g1_amplitude'].min = 0.0
        
        pars['g2_amplitude'].set(expr='g0_amplitude')
        pars['g3_amplitude'].set(expr='g1_amplitude')

        pars['g0_sigma'].value = 1.0
        pars['g1_sigma'].value = 1.0
        pars['g2_sigma'].value = 1.0
        pars['g3_sigma'].value = 1.0

        pars['g0_sigma'].min = 1000.0
        pars['g1_sigma'].min = 1000.0

        pars['g2_sigma'].set(expr='g0_sigma')
        pars['g3_sigma'].set(expr='g1_sigma')

        pars['g0_center'].value = 0.0
        pars['g1_center'].value = 0.0
        pars['g2_center'].value = 0.0
        pars['g3_center'].value = 0.0

        pars['g1_center'].set(expr='g0_center')        
        pars['g2_center'].set(expr = 'g0_center+{}'.format(wave2doppler(1402*u.AA, w0).value - wave2doppler(1397*u.AA, w0).value))
        pars['g3_center'].set(expr='g2_center')

    if fitting_method == 'leastsq':

        out = minimize(resid,
                       pars,
                       kws={'x':np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value/xscale), 
                            'model':mod, 
                            'data':ma.getdata(y[~ma.getmaskarray(y)]),
                            'sigma':ma.getdata(er[~ma.getmaskarray(er)])},
                       method=fitting_method
                       )         


    else:

        out = minimize(resid,
                       pars,
                       kws={'x':np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value/xscale), 
                            'model':mod, 
                            'data':ma.getdata(y[~ma.getmaskarray(y)]),
                            'sigma':ma.getdata(er[~ma.getmaskarray(er)])},
                       method=fitting_method,
                       options={'maxiter':1e6, 'maxfev':1e6} 
                       )                   

    
    if n_samples == 1: 
    
        if verbose:
        
            print colored(out.message, 'red'), colored('Number of function evaluations: {}'.format(out.nfev), 'red') 
            
            if fit_model == 'GaussHermite':
    
                for key, value in out.params.valuesdict().items():
                    if 'cen' in key:
                        print key, value * xscale
                    else:
                        print key, value
    
            else:
    
                print fit_report(out.params)

    if (plot) & (n_samples > 1):

        xdat_plot = ma.getdata(wav_array_plot_k[~ma.getmaskarray(wav_array_plot_k)]).value 
        vdat_plot = ma.getdata(vdat_array_plot_k[~ma.getmaskarray(vdat_array_plot_k)]).value 
        ydat_plot = ma.getdata(flux_array_plot_k[~ma.getmaskarray(flux_array_plot_k)]) 
        yerr_plot = ma.getdata(err_array_plot_k[~ma.getmaskarray(err_array_plot_k)])

        if 'pts1' in locals():
            pts1.set_data((vdat_plot, ydat_plot))
        else:
            pts1, = fit.plot(vdat_plot, ydat_plot, markerfacecolor='grey', markeredgecolor='None', linestyle='', marker='o', markersize=2)

        if 'pts2' in locals():
            pts2.set_data((vdat_plot, (ydat_plot - resid(params=out.params, x=vdat_plot/xscale, model=mod)) / yerr_plot))
        else:
            pts2, = residuals.plot(vdat_plot, (ydat_plot - resid(params=out.params, x=vdat_plot/xscale, model=mod)) / yerr_plot, markerfacecolor='grey', markeredgecolor='None', linestyle='', marker='o', markersize=2)

        vs = np.linspace(vdat_plot.min(), vdat_plot.max(), 1000)

        if 'line' in locals():
            line.set_data((vs, resid(params=out.params, x=vs/xscale, model=mod)))
        else:
            line, = fit.plot(vs, resid(params=out.params, x=vs/xscale, model=mod), color='black')

        fig.set_tight_layout(True)


        plt.pause(0.1)


    if save_dir is not None:
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        if n_samples == 1: 
    
            param_file = os.path.join(save_dir, 'my_params.txt')
            parfile = open(param_file, 'w')
            out.params.dump(parfile)
            parfile.close()
    
            wav_file = os.path.join(save_dir, 'wav.txt')
            parfile = open(wav_file, 'wb')
            pickle.dump(wav, parfile, -1)
            parfile.close()    
    
            if subtract_fe is True:
                flux_dump = flux - resid(params=bkgdpars_k, 
                                         x=wav.value, 
                                         model=bkgdmod,
                                         sp_fe=sp_fe)
            
            if subtract_fe is False:
                flux_dump = flux - resid(params=bkgdpars_k, 
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
    
            sd_file = os.path.join(save_dir, 'sd.txt')
            parfile = open(sd_file, 'wb')
            pickle.dump(xscale, parfile, -1)
            parfile.close()
    
            fittxt = ''    
            fittxt += strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n \n'
            fittxt += plot_title + '\n \n'
            fittxt += r'Converged with chi-squared = ' + str(out.chisqr) + ', DOF = ' + str(out.nfree) + '\n \n'
            fittxt += fit_report(out.params)
    
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
       
        for key, value in out.params.valuesdict().iteritems():
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
       
        for key, value in out.params.valuesdict().iteritems():
            if key.startswith('ha_b_'):
                pars_broad_ha[key].value = value 
                
    
        integrand = lambda x: mod_broad_ha.eval(params=pars_broad_ha, x=np.array(x))    
    
    else:
    
        integrand = lambda x: mod.eval(params=out.params, x=np.array(x))
    
    # Calculate FWHM of distribution
    if line_region.unit == u.AA:
        line_region = wave2doppler(line_region, w0)
    
    dv = 1.0 # i think if this was anything else might need to change intergration.
    xs = np.arange(line_region.value[0], line_region[1].value, dv) / xscale 
    
    norm = np.sum(integrand(xs) * dv)
    pdf = integrand(xs) / norm
    cdf = np.cumsum(pdf)
    cdf_r = np.cumsum(pdf[::-1])[::-1] # reverse cumsum

    # max of line
    peak_flux = np.max(integrand(xs)) / spec_norm
    
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
    xs_wav = doppler2wave(xs*xscale*(u.km/u.s), w0)
    
    if subtract_fe is True:
        flux_bkgd = resid(params=bkgdpars_k, 
                          x=xs_wav.value, 
                          model=bkgdmod,
                          sp_fe=sp_fe)
    
    if subtract_fe is False:

        flux_bkgd = resid(params=bkgdpars_k, 
                          x=xs_wav.value, 
                          model=bkgdmod)
    
    f = (flux_line + flux_bkgd) / flux_bkgd
    eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
    eqw = np.nansum(eqw)
    
    # Broad luminosity
    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)        
    broad_lum = np.sum(integrand(xs)[:-1] * np.diff(xs_wav.value)) * (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm 
    
    if fit_model == 'Ha':       
        
        # I know that the Gaussians are normalised, so the area is just the amplitude, so surely
        # I shouldn't have to do this function evaluation. 

        broad_fwhms = np.zeros(nGaussians)
        broad_cens = np.zeros(nGaussians)
        broad_amps = np.zeros(nGaussians)

        for i in range(nGaussians):   
            broad_fwhms[i] = np.array(out.params['ha_b_{}_fwhm'.format(i)].value)
            broad_cens[i] = np.array(out.params['ha_b_{}_center'.format(i)].value)
            broad_amps[i] = np.array(out.params['ha_b_{}_amplitude'.format(i)].value)

        inds = np.argsort(broad_fwhms)[::-1]

        broad_fwhms = broad_fwhms[inds]
        broad_cens = broad_cens[inds]
        broad_amps = broad_amps[inds]

        if len(broad_fwhms) == 1:
            broad_fwhms = np.append(broad_fwhms, np.nan)
            broad_cens = np.append(broad_cens, np.nan)
            broad_amps = np.append(broad_amps, np.nan)

        if len(broad_fwhms) == 2: 
            broad_fwhms = np.append(broad_fwhms, np.nan)
            broad_cens = np.append(broad_cens, np.nan)
            broad_amps = np.append(broad_amps, np.nan)


        narrow_mod = GaussianModel()
        narrow_pars = narrow_mod.make_params()
        narrow_pars['amplitude'].value = out.params['ha_n_amplitude'].value
        narrow_pars['sigma'].value = out.params['ha_n_sigma'].value 
        narrow_pars['center'].value = out.params['ha_n_center'].value 
    
        narrow_lum = np.sum(narrow_mod.eval(params=narrow_pars, x=xs)[:-1] * np.diff(xs_wav.value)) * \
                    (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
        narrow_fwhm = out.params['ha_n_fwhm'].value 
        narrow_voff = out.params['ha_n_center'].value 
    
        oiii_5007_eqw = np.nan 
        oiii_5007_lum = np.nan * (u.erg / u.s)
        oiii_5007_n_lum = np.nan * (u.erg / u.s)
        oiii_5007_b_lum = np.nan * (u.erg / u.s)
        oiii_5007_fwhm = np.nan
        oiii_5007_n_fwhm = np.nan 
        oiii_5007_b_fwhm = np.nan 
        oiii_5007_b_voff = np.nan 
        oiii_5007_blueshift = np.nan 
    
        #######################################################################
        """
        Nicely print out important fitting info
        """     
        
        if verbose: 

            if n_samples == 1: 
                if out.params['ha_n_sigma'].vary is True:
                    print 'Narrow FWHM = {0:.1f}, Initial = {1:.1f}, Vary = {2}, Min = {3:.1f}, Max = {4:.1f}'.format(out.params['ha_n_sigma'].value * 2.35, 
                                                                                                                      out.params['ha_n_sigma'].init_value * 2.35, 
                                                                                                                      out.params['ha_n_sigma'].vary, 
                                                                                                                      out.params['ha_n_sigma'].min * 2.35, 
                                                                                                                      out.params['ha_n_sigma'].max * 2.35) 
                else:
                    print 'Narrow FWHM = {0:.1f}, Vary = {1}'.format(out.params['ha_n_sigma'].value * 2.35, 
                                                                     out.params['ha_n_sigma'].vary) 
        
                if out.params['ha_n_center'].vary is True:
                    print 'Narrow Center = {0:.1f}, Initial = {1:.1f}, Vary = {2}, Min = {3:.1f}, Max = {4:.1f}'.format(out.params['ha_n_center'].value, 
                                                                                                                        out.params['ha_n_center'].init_value, 
                                                                                                                        out.params['ha_n_center'].vary, 
                                                                                                                        out.params['ha_n_center'].min, 
                                                                                                                        out.params['ha_n_center'].max) 
                else:
                    print 'Narrow Center = {0:.1f}, Vary = {1}'.format(out.params['ha_n_center'].value, 
                                                                       out.params['ha_n_center'].vary)     
                                                                 
                                                                 
                                                        
    
    
    
    elif fit_model == 'Hb':    

        broad_fwhms = np.zeros(nGaussians)
        broad_cens = np.zeros(nGaussians)
        broad_amps = np.zeros(nGaussians)

        for i in range(nGaussians):   
            broad_fwhms[i] = np.array(out.params['hb_b_{}_fwhm'.format(i)].value)
            broad_cens[i] = np.array(out.params['hb_b_{}_center'.format(i)].value)
            broad_amps[i] = np.array(out.params['hb_b_{}_amplitude'.format(i)].value)

        inds = np.argsort(broad_fwhms)[::-1]

        broad_fwhms = broad_fwhms[inds]
        broad_cens = broad_cens[inds]
        broad_amps = broad_amps[inds]

        if len(broad_fwhms) == 1:

            broad_fwhms = np.append(broad_fwhms, np.nan)
            broad_cens = np.append(broad_cens, np.nan)
            broad_amps = np.append(broad_amps, np.nan)

        if len(broad_fwhms) == 2:

            broad_fwhms = np.append(broad_fwhms, np.nan)
            broad_cens = np.append(broad_cens, np.nan)
            broad_amps = np.append(broad_amps, np.nan)

    
        oiii_5007_b_mod = GaussianModel()
        oiii_5007_b_pars = oiii_5007_b_mod.make_params() 
    
        oiii_5007_b_pars['amplitude'].value = out.params['oiii_5007_b_amplitude'].value
        oiii_5007_b_pars['sigma'].value = out.params['oiii_5007_b_sigma'].value 
        oiii_5007_b_pars['center'].value = out.params['oiii_5007_b_center'].value 
    
        oiii_5007_b_xs = np.arange(oiii_5007_b_pars['center'].value - 10000.0, 
                                   oiii_5007_b_pars['center'].value + 10000.0, 
                                   dv)
    
        oiii_5007_b_xs_wav = doppler2wave(oiii_5007_b_xs*(u.km/u.s), w0) 
    
    
        oiii_5007_b_lum = np.sum(oiii_5007_b_mod.eval(params=oiii_5007_b_pars, x=oiii_5007_b_xs)[:-1] * np.diff(oiii_5007_b_xs_wav.value)) * \
                    (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
        oiii_5007_b_fwhm = oiii_5007_b_pars['sigma'].value * 2.35 
    
        # Velocity of broad component of OIII relative to narrow component
        oiii_5007_b_voff = out.params['oiii_5007_b_center_delta'].value  

        oiii_5007_n_mod = GaussianModel()
        oiii_5007_n_pars = oiii_5007_n_mod.make_params() 
    
        oiii_5007_n_pars['amplitude'].value = out.params['oiii_5007_n_amplitude'].value
        oiii_5007_n_pars['sigma'].value = out.params['oiii_5007_n_sigma'].value 
        oiii_5007_n_pars['center'].value = out.params['oiii_5007_n_center'].value 
    
        oiii_5007_n_xs = np.arange(oiii_5007_n_pars['center'].value - 10000.0, 
                                   oiii_5007_n_pars['center'].value + 10000.0, 
                                   dv)
    
        oiii_5007_n_xs_wav = doppler2wave(oiii_5007_n_xs*(u.km/u.s), w0) 
    
    
        oiii_5007_n_lum = np.sum(oiii_5007_n_mod.eval(params=oiii_5007_n_pars, x=oiii_5007_n_xs)[:-1] * np.diff(oiii_5007_n_xs_wav.value)) * \
                    (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
        oiii_5007_n_fwhm = oiii_5007_n_pars['sigma'].value * 2.35 
           
        oiii_5007_mod = GaussianModel(prefix='oiii_5007_n_') + GaussianModel(prefix='oiii_5007_b_')
        oiii_5007_pars = oiii_5007_mod.make_params() 
    
        oiii_5007_pars['oiii_5007_n_amplitude'].value = out.params['oiii_5007_n_amplitude'].value
        oiii_5007_pars['oiii_5007_n_sigma'].value = out.params['oiii_5007_n_sigma'].value 
        oiii_5007_pars['oiii_5007_n_center'].value = out.params['oiii_5007_n_center'].value 
        oiii_5007_pars['oiii_5007_b_amplitude'].value = out.params['oiii_5007_b_amplitude'].value
        oiii_5007_pars['oiii_5007_b_sigma'].value = out.params['oiii_5007_b_sigma'].value 
        oiii_5007_pars['oiii_5007_b_center'].value = out.params['oiii_5007_b_center'].value 
    
    
        flux_line = oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs)
    
    
        if subtract_fe is True:

            flux_bkgd = resid(params=bkgdpars_k, 
                              x=oiii_5007_n_xs_wav.value, 
                              model=bkgdmod,
                              sp_fe=sp_fe)
    
        if subtract_fe is False:

            flux_bkgd = resid(params=bkgdpars_k, 
                              x=oiii_5007_n_xs_wav.value, 
                              model=bkgdmod)
    
        f = (flux_line + flux_bkgd) / flux_bkgd 
        oiii_5007_eqw = (f[:-1] - 1.0) * np.diff(oiii_5007_n_xs_wav.value)
        oiii_5007_eqw = np.nansum(oiii_5007_eqw)
    
        oiii_5007_lum = np.sum(oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs)[:-1] * np.diff(oiii_5007_n_xs_wav.value)) * \
                    (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
        oiii_5007_pdf = oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs)
        half_max = np.max(oiii_5007_pdf) / 2.0
    
        i = 0
        while oiii_5007_pdf[i] < half_max:
            i+=1
    
        oiii_5007_root1 = oiii_5007_n_xs[i]
    
        i = 0
        while oiii_5007_pdf[-i] < half_max:
            i+=1
    
        oiii_5007_root2 = oiii_5007_n_xs[-i]
        
        oiii_5007_fwhm = oiii_5007_root2 - oiii_5007_root1

        # median for blueshift 
        oiii_5007_norm = np.sum(oiii_5007_pdf * dv)
        oiii_5007_cdf = np.cumsum(oiii_5007_pdf / oiii_5007_norm)       
        oiii_5007_blueshift = out.params['oiii_5007_n_center'].value - oiii_5007_n_xs[np.argmin( np.abs(oiii_5007_cdf - 0.5))]

        


        #-------------------------------------------------------------------------------------------
    
        narrow_fwhm = out.params['oiii_5007_n_sigma'].value * 2.35 
        narrow_voff = out.params['oiii_5007_n_center'].value  - wave2doppler(5008.239*u.AA, w0).value  
         
        if hb_narrow is True:
    
            narrow_mod = GaussianModel()
            narrow_pars = narrow_mod.make_params()
            narrow_pars['amplitude'].value = out.params['hb_n_amplitude'].value
            narrow_pars['sigma'].value = out.params['hb_n_sigma'].value 
            narrow_pars['center'].value = out.params['hb_n_center'].value 
    
            narrow_lum = np.sum(narrow_mod.eval(params=narrow_pars, x=xs)[:-1] * np.diff(xs_wav.value)) * \
                        (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm
    
        
            #######################################################################
            """
            Nicely print out important fitting info
            """     
            
            if verbose: 

                if n_samples == 1:
    
                    if out.params['oiii_5007_n_sigma'].vary is True:
    
    
                        print 'Narrow FWHM = {0:.1f}, Initial = {1:.1f}, Min = {2:.1f}, Max = {3:.1f}'.format(out.params['oiii_5007_n_sigma'].value * 2.35, 
                                                                                                              out.params['oiii_5007_n_sigma'].init_value * 2.35, 
                                                                                                              out.params['oiii_5007_n_sigma'].min * 2.35, 
                                                                                                              out.params['oiii_5007_n_sigma'].max * 2.35) 
                    else:
                        print 'Narrow FWHM = {0:.1f}, Vary = {1}'.format(out.params['oiii_5007_n_sigma'].value * 2.35, 
                                                                         out.params['oiii_5007_n_sigma'].vary) 
            
                    if out.params['oiii_5007_n_center'].vary is True:
                        print 'Narrow Center = {0:.1f}, Initial = {1:.1f}, Min = {2:.1f}, Max = {3:.1f}'.format(out.params['oiii_5007_n_center'].value, 
                                                                                                                out.params['oiii_5007_n_center'].init_value, 
                                                                                                                out.params['oiii_5007_n_center'].min, 
                                                                                                                out.params['oiii_5007_n_center'].max) 
                    else:
                        print 'Narrow Center = {0:.1f}, Vary = {1}'.format(out.params['oiii_5007_n_center'].value, 
                                                                           out.params['oiii_5007_n_center'].vary)     
    
        else:
    
            narrow_lum = np.nan * (u.erg / u.s)          
    
    elif fit_model == 'GaussHermite':
      
        narrow_lum = np.nan * (u.erg / u.s)
        narrow_fwhm = np.nan 
        narrow_voff = np.nan 
        oiii_5007_eqw = np.nan 
        oiii_5007_lum = np.nan * (u.erg / u.s)
        oiii_5007_n_lum = np.nan * (u.erg / u.s)
        oiii_5007_b_lum = np.nan * (u.erg / u.s)
        oiii_5007_fwhm = np.nan 
        oiii_5007_n_fwhm = np.nan 
        oiii_5007_b_fwhm = np.nan 
        oiii_5007_b_voff = np.nan
        oiii_5007_blueshift = np.nan 

        broad_fwhms = np.repeat(np.nan, 3)
        broad_cens = np.repeat(np.nan, 3)
        broad_amps = np.repeat(np.nan, 3)

    elif fit_model == 'MultiGauss':

        broad_fwhms = np.zeros(nGaussians)
        broad_cens = np.zeros(nGaussians)
        broad_amps = np.zeros(nGaussians)

        for i in range(nGaussians):   
            broad_fwhms[i] = np.array(out.params['g{}_fwhm'.format(i)].value)
            broad_cens[i] = np.array(out.params['g{}_center'.format(i)].value)
            broad_amps[i] = np.array(out.params['g{}_amplitude'.format(i)].value)

        inds = np.argsort(broad_fwhms)[::-1]

        broad_fwhms = broad_fwhms[inds]
        broad_cens = broad_cens[inds]
        broad_amps = broad_amps[inds]

        if len(broad_fwhms) == 1:
            broad_fwhms = np.append(broad_fwhms, np.nan)
            broad_cens = np.append(broad_cens, np.nan)
            broad_amps = np.append(broad_amps, np.nan)

        if len(broad_fwhms) == 2: 
            broad_fwhms = np.append(broad_fwhms, np.nan)
            broad_cens = np.append(broad_cens, np.nan)
            broad_amps = np.append(broad_amps, np.nan)

        narrow_lum = np.nan * (u.erg / u.s)
        narrow_fwhm = np.nan 
        narrow_voff = np.nan 
        oiii_5007_eqw = np.nan 
        oiii_5007_lum = np.nan * (u.erg / u.s)
        oiii_5007_n_lum = np.nan * (u.erg / u.s)
        oiii_5007_b_lum = np.nan * (u.erg / u.s)
        oiii_5007_fwhm = np.nan 
        oiii_5007_n_fwhm = np.nan 
        oiii_5007_b_fwhm = np.nan 
        oiii_5007_b_voff = np.nan
        oiii_5007_blueshift = np.nan 

    if fit_model == 'siiv':


        broad_fwhm = [np.nan] * 3 
        narrow_lum = np.nan * (u.erg / u.s)
        narrow_fwhm = np.nan 
        narrow_voff = np.nan 
        oiii_5007_eqw = np.nan 
        oiii_5007_lum = np.nan * (u.erg / u.s)
        oiii_5007_n_lum = np.nan * (u.erg / u.s)
        oiii_5007_b_lum = np.nan * (u.erg / u.s)
        oiii_5007_fwhm = np.nan 
        oiii_5007_n_fwhm = np.nan 
        oiii_5007_b_fwhm = np.nan 
        oiii_5007_b_voff = np.nan
        oiii_5007_blueshift = np.nan 




    fit_out = {'name':plot_title, 
               'fwhm':(root2 - root1)*xscale,
               'fwhm_1':broad_fwhms[0],
               'fwhm_2':broad_fwhms[1],
               'fwhm_3':broad_fwhms[2],
               'sigma': sd*xscale,
               'median': md*xscale,
               'cen': func_center*xscale,
               'cen_1':broad_cens[0],
               'cen_2':broad_cens[1],
               'cen_3':broad_cens[2],
               'eqw': eqw,
               'broad_lum':np.log10(broad_lum.value),
               'peak':peak_flux, 
               'amplitude_1':broad_amps[0],
               'amplitude_2':broad_amps[1],
               'amplitude_3':broad_amps[2],
               'narrow_fwhm':narrow_fwhm,
               'narrow_lum':np.log10(narrow_lum.value) if ~np.isnan(narrow_lum.value) else np.nan,
               'narrow_voff':narrow_voff, 
               'oiii_5007_eqw':oiii_5007_eqw,
               'oiii_5007_lum':np.log10(oiii_5007_lum.value) if ~np.isnan(oiii_5007_lum.value) else np.nan,
               'oiii_5007_n_lum':np.log10(oiii_5007_n_lum.value) if ~np.isnan(oiii_5007_n_lum.value) else np.nan,
               'oiii_5007_b_lum':np.log10(oiii_5007_b_lum.value) if ~np.isnan(oiii_5007_b_lum.value) else np.nan,
               'oiii_fwhm':oiii_5007_fwhm,
               'oiii_n_fwhm':oiii_5007_n_fwhm,
               'oiii_b_fwhm':oiii_5007_b_fwhm,
               'oiii_5007_b_voff':oiii_5007_b_voff,
               'oiii_5007_blueshift':oiii_5007_blueshift, 
               'redchi':out.redchi,
               'snr':snr_k,
               'dv':spec_dv, 
               'monolum':np.log10(mono_lum_k),
               'fe_ew':eqw_fe_k}
    
    if n_samples == 1: 

        if verbose:

            print '\n'
            print 'Monochomatic luminosity at {0} = {1:.3f}'.format(mono_lum_wav, np.log10(mono_lum_k)) 
            print '\n'
        
    
            print  fit_out['name'] + '\n'\
                  'Broad FWHM: {0:.2f} km/s \n'.format(fit_out['fwhm']), \
                  'Narrow Broad FWHM: {0:.2f} km/s \n'.format(fit_out['fwhm_1']), \
                  'Very Broad FWHM: {0:.2f} km/s \n'.format(fit_out['fwhm_2']), \
                  'Broad sigma: {0:.2f} km/s \n'.format(fit_out['sigma']), \
                  'Broad median: {0:.2f} km/s \n'.format(fit_out['median']), \
                  'Broad centroid: {0:.2f} km/s \n'.format(fit_out['cen']), \
                  'Broad EQW: {0:.2f} A \n'.format(fit_out['eqw']), \
                  'Broad luminosity {0:.2f} erg/s \n'.format(fit_out['broad_lum']), \
                  'Broad peak {0:.2e} erg/s \n'.format(fit_out['peak']), \
                  'Narrow FWHM: {0:.2f} km/s \n'.format(fit_out['narrow_fwhm']), \
                  'Narrow luminosity: {0:.2f} km/s \n'.format(fit_out['narrow_lum']), \
                  'Narrow velocity: {0:.2f} km/s \n'.format(fit_out['narrow_voff']), \
                  'OIII5007 EQW: {0:.2f} A \n'.format(fit_out['oiii_5007_eqw']),\
                  'OIII5007 luminosity {0:.2f} erg/s \n'.format(fit_out['oiii_5007_lum']),\
                  'OIII5007 narrow luminosity: {0:.2f} erg/s \n'.format(fit_out['oiii_5007_n_lum']),\
                  'OIII5007 broad luminosity: {0:.2f} erg/s \n'.format(fit_out['oiii_5007_b_lum']),\
                  'OIII5007 FWHM: {0:.2f} km/s \n'.format(fit_out['oiii_fwhm']),\
                  'OIII5007 narrow FWHM: {0:.2f} km/s \n'.format(fit_out['oiii_n_fwhm']),\
                  'OIII5007 broad FWHM: {0:.2f} km/s \n'.format(fit_out['oiii_b_fwhm']),\
                  'OIII5007 broad velocity: {0:.2f} km/s \n'.format(fit_out['oiii_5007_b_voff']),\
                  'OIII5007 blueshift: {0:.2f} km/s \n'.format(fit_out['oiii_5007_blueshift']),\
                  'Reduced chi-squared: {0:.2f} \n'.format(fit_out['redchi']),\
                  'S/N: {0:.2f} \n'.format(fit_out['snr']), \
                  'dv: {0:.1f} km/s \n'.format(fit_out['dv'].value), \
                  'Monochomatic luminosity: {0:.2f} erg/s \n'.format(fit_out['monolum'])
    
            if emission_line == 'Hb':

                print fit_out['name'] + ',',\
                format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
                format(fit_out['fwhm_1'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_1']) else ',',\
                format(fit_out['fwhm_2'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_2']) else ',',\
                format(fit_out['fwhm_3'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_3']) else ',',\
                format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
                format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
                format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
                format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
                format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
                format(fit_out['narrow_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['narrow_fwhm']) else ',',\
                format(fit_out['narrow_lum'], '.2f') + ',' if ~np.isnan(fit_out['narrow_lum']) else ',',\
                format(fit_out['narrow_voff'], '.2f') + ',' if ~np.isnan(fit_out['narrow_voff']) else ',',\
                format(fit_out['oiii_5007_eqw'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_eqw']) else ',',\
                format(fit_out['oiii_5007_lum'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_lum']) else ',',\
                format(fit_out['oiii_5007_n_lum'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_n_lum']) else ',',\
                format(fit_out['oiii_5007_b_lum'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_b_lum']) else ',',\
                format(fit_out['oiii_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['oiii_fwhm']) else ',',\
                format(fit_out['oiii_n_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['oiii_n_fwhm']) else ',',\
                format(fit_out['oiii_b_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['oiii_b_fwhm']) else ',',\
                format(fit_out['oiii_5007_b_voff'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_b_voff']) else ',',\
                format(fit_out['oiii_5007_blueshift'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_blueshift']) else ',',\
                format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''


                # print  fit_out['name'] + ','\
                #       '{0:.2f},'.format(fit_out['fwhm']), \
                #       '{0:.2f},'.format(fit_out['sigma']), \
                #       '{0:.2f},'.format(fit_out['median']), \
                #       '{0:.2f},'.format(fit_out['cen']), \
                #       '{0:.2f},'.format(fit_out['eqw']), \
                #       '{0:.2f},'.format(fit_out['broad_lum']), \
                #       '{0:.2f},'.format(fit_out['narrow_fwhm']), \
                #       '{0:.2f},'.format(fit_out['narrow_lum']), \
                #       '{0:.2f},'.format(fit_out['narrow_voff']), \
                #       '{0:.2f},'.format(fit_out['oiii_5007_eqw']),\
                #       '{0:.2f},'.format(fit_out['oiii_5007_lum']),\
                #       '{0:.2f},'.format(fit_out['oiii_5007_n_lum']),\
                #       '{0:.2f},'.format(fit_out['oiii_5007_b_lum']),\
                #       '{0:.2f},'.format(fit_out['oiii_fwhm']),\
                #       '{0:.2f},'.format(fit_out['oiii_n_fwhm']),\
                #       '{0:.2f},'.format(fit_out['oiii_b_fwhm']),\
                #       '{0:.2f},'.format(fit_out['oiii_5007_b_voff']),\
                #       '{0:.2f}'.format(fit_out['redchi'])          
    

            elif emission_line == 'Ha':

                print fit_out['name'] + ',',\
                format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
                format(fit_out['fwhm_1'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_1']) else ',',\
                format(fit_out['fwhm_2'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_2']) else ',',\
                format(fit_out['fwhm_3'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_3']) else ',',\
                format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
                format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
                format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
                format(fit_out['cen_1'], '.2f') + ',' if ~np.isnan(fit_out['cen_1']) else ',',\
                format(fit_out['cen_2'], '.2f') + ',' if ~np.isnan(fit_out['cen_2']) else ',',\
                format(fit_out['cen_3'], '.2f') + ',' if ~np.isnan(fit_out['cen_3']) else ',',\
                format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
                format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
                format(fit_out['amplitude_1'], '.2f') + ',' if ~np.isnan(fit_out['amplitude_1']) else ',',\
                format(fit_out['amplitude_2'], '.2f') + ',' if ~np.isnan(fit_out['amplitude_2']) else ',',\
                format(fit_out['amplitude_3'], '.2f') + ',' if ~np.isnan(fit_out['amplitude_3']) else ',',\
                format(fit_out['narrow_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['narrow_fwhm']) else ',',\
                format(fit_out['narrow_lum'], '.2f') + ',' if ~np.isnan(fit_out['narrow_lum']) else ',',\
                format(fit_out['narrow_voff'], '.2f') + ',' if ~np.isnan(fit_out['narrow_voff']) else ',',\
                format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''

            else: 

                print fit_out['name'] + ',',\
                format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
                format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
                format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
                format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
                format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
                format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
                format(fit_out['peak'], '.2e') + ',' if ~np.isnan(fit_out['peak']) else ',',\
                format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''

    
    
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
                flux_plot = flux - resid(params=bkgdpars_k, 
                                         x=wav.value, 
                                         model=bkgdmod,
                                         sp_fe=sp_fe)
            
            if subtract_fe is False:
                flux_plot = flux - resid(params=bkgdpars_k, 
                                         x=wav.value, 
                                         model=bkgdmod)
    
            plot_fit(wav=wav,
                     flux = flux_plot,
                     err=err,
                     pars=out.params,
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
                     verbose=verbose,
                     reject_outliers=reject_outliers,  
                     reject_width=reject_width,
                     reject_sigma=reject_sigma,
                     gh_order = gh_order,
                     xscale=xscale)
    
            plot_fit_d3(wav=wav,
                        flux = flux_plot,
                        err=err,
                        pars=out.params,
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
                        verbose=verbose,
                        xscale=xscale,
                        gh_order=gh_order)
        

    else:

        print plot_title, k
    
    return fit_out 

        


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
             gh_order=6,
             maskout=None,
             verbose=True,
             plot=True,
             plot_savefig=None,
             plot_title='',
             save_dir=None,
             bkgd_median=True,
             fitting_method='leastsq',
             mask_negflux = True,
             mono_lum_wav = 5100 * u.AA,
             fit_model='MultiGauss',
             subtract_fe=False,
             fe_FWHM=4000.0*(u.km/u.s),
             fe_FWHM_vary=False,
             hb_narrow=True,
             n_rebin=1,
             ha_narrow_fwhm=600.0,
             ha_narrow_voff=0.0,
             ha_narrow_vary=True,  
             reject_outliers = False,  
             reject_width = 20,
             reject_sigma = 3.0,
             n_samples=1,
             emission_line='Ha',
             pseudo_continuum_fit=False,
             parallel=False,
             cores=8,
             fix_broad_peaks=False,
             oiii_broad_off=False):


    """
    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    In previos version I did not include the errors in the background fit, which is why
    the results are slightly different 

    """

    # n_samples = 1

    if n_samples > 1: 

        if save_dir is not None:
    
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if emission_line == 'Ha':
                columns = ['fwhm', 'fwhm_1', 'fwhm_2', 'fwhm_3', 'sigma', 'median', 'cen', 'eqw', 'broad_lum', 'peak', 'narrow_fwhm', 'narrow_lum', 'narrow_voff', 'redchi', 'snr', 'monolum', 'fe_ew'] 
                
            if emission_line == 'Hb':
                columns = ['fwhm', 'fwhm_1', 'fwhm_2', 'fwhm_3', 'sigma', 'median', 'cen', 'eqw', 'broad_lum', 'peak', 'narrow_fwhm', 'narrow_lum', 'narrow_voff', 'oiii_5007_eqw', 'oiii_5007_lum', 'oiii_5007_n_lum', 'oiii_5007_b_lum', 'oiii_fwhm', 'oiii_n_fwhm', 'oiii_b_fwhm', 'oiii_5007_b_voff', 'oiii_5007_blueshift', 'redchi', 'snr', 'monolum', 'fe_ew']
                
            if emission_line == 'CIV':
                columns = ['fwhm', 'sigma', 'median', 'cen', 'eqw', 'broad_lum', 'peak', 'redchi', 'snr', 'monolum']

            if pseudo_continuum_fit:
                columns = ['monolum', 'fe_ew']

            if emission_line == 'siiv':
                columns = ['fwhm', 'sigma', 'median', 'cen', 'eqw', 'broad_lum', 'peak', 'redchi', 'snr']


            index = np.arange(n_samples)

            df_out = pd.DataFrame(index=index, columns=columns)     


    home_dir = expanduser("~")

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA 
    dw = dw / (1.0 + z) # don't actually use this but if I did be careful if spectrum is rebinned

    # Normalise spectrum

    spec_norm = 1.0 / np.median(flux[(flux != 0.0) & ~np.isnan(flux)])
    flux = flux * spec_norm 
    err = err * spec_norm 

    # Rebin spectrum 
    wav_norebin = wav * 1.0
    flux_norebin = flux * 1.0
    err_norebin = err * 1.0 

    wav, flux, err = rebin(wav, flux, err, n_rebin)

    n_elements = len(wav)
    n_elements_norebin = len(wav_norebin)

    # index of the region we want to fit
    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    if len(wav[(wav > fitting_region[0]) & (wav < fitting_region[1])]) == 0:
       
        if verbose:
            print 'No flux in fitting region'

        if n_samples == 1:
       
            fit_out = {'name':plot_title, 
                       'fwhm':np.nan,
                       'fwhm_1':np.nan,
                       'fwhm_2':np.nan,
                       'fwhm_3':np.nan,
                       'sigma':np.nan,
                       'median':np.nan,
                       'cen':np.nan,
                       'cen_1':np.nan,
                       'cen_2':np.nan,
                       'cen_3':np.nan,
                       'eqw':np.nan,
                       'broad_lum':np.nan,
                       'peak':np.nan, 
                       'amplitude_1':np.nan,
                       'amplitude_2':np.nan,
                       'amplitude_3':np.nan,
                       'narrow_fwhm':np.nan,
                       'narrow_lum':np.nan,
                       'narrow_voff':np.nan, 
                       'oiii_5007_eqw':np.nan,
                       'oiii_5007_lum':np.nan,
                       'oiii_5007_n_lum':np.nan,
                       'oiii_5007_b_lum':np.nan,
                       'oiii_fwhm':np.nan,
                       'oiii_n_fwhm':np.nan,
                       'oiii_b_fwhm':np.nan,
                       'oiii_5007_b_voff':np.nan,
                       'oiii_5007_blueshift':np.nan,
                       'redchi':np.nan,
                       'snr':np.nan,
                       'dv':np.nan, 
                       'monolum':np.nan,
                       'fe_ew':np.nan}
            
            if save_dir is not None:
    
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
    
                with open(os.path.join(save_dir, 'fit.txt'), 'w') as f:
                    f.write('No fit')               
        
            if plot: 
                plot_spec(wav=wav,
                          flux=flux,
                          plot_savefig = plot_savefig,
                          save_dir = save_dir,
                          z=z,
                          w0=w0,
                          continuum_region=continuum_region,
                          fitting_region=fitting_region,
                          plot_region=plot_region,
                          plot_title=plot_title,
                          verbose=verbose)
    
            
            return fit_out 

        else:

            if save_dir is not None:
                df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 

            return None 


    # might have trouble if err is negative, but do whatever and mask out before fit. 

    if n_samples > 1: 

        # print 'Warning loading flux array!'
        flux_array = np.asarray(np.random.normal(flux,
                                                 err,
                                                 size=(n_samples, n_elements)),
                                dtype=np.float32)

        # f = os.path.join(save_dir, 'flux_array.pickle')
        # p = open(f, 'wb')
        # pickle.dump(flux_array, p, -1)
        # p.close()

        # f = os.path.join(save_dir, 'flux_array.pickle')
        # p = open(f, 'rb')
        # flux_array = pickle.load(p)
        # p.close()


        # Can't draw from normal distribution with sigma = 0.0
        # These pixels should be masked in the fit so won't matter. 
        err_norebin_tmp = err_norebin * 1.0
        err_norebin_tmp[err_norebin_tmp <= 0.0] = 1e3 

        flux_array_norebin = np.asarray(np.random.normal(flux_norebin,
                                                         err_norebin_tmp,
                                                         size=(n_samples, n_elements_norebin)),
                                dtype=np.float32)


    else: 

        flux_array = flux.reshape(1, n_elements) 
        flux_array_norebin = flux_norebin.reshape(1, n_elements_norebin) 
    
    wav_array = np.repeat(wav.reshape(1, n_elements), n_samples, axis=0) 
    err_array = np.repeat(err.reshape(1, n_elements), n_samples, axis=0) 
    vdat_array = wave2doppler(wav_array, w0)

    wav_array_norebin = np.repeat(wav_norebin.reshape(1, n_elements_norebin), n_samples, axis=0) 
    err_array_norebin = np.repeat(err_norebin.reshape(1, n_elements_norebin), n_samples, axis=0) 
    vdat_array_norebin = wave2doppler(wav_array_norebin, w0)

    # don't do any of the outlier masking etc. to this array. 

    if plot_region.unit == (u.km/u.s):
        plot_region = doppler2wave(plot_region, w0)

    mask = (ma.getdata(wav_array).value < plot_region[0].value) | (ma.getdata(wav_array).value > plot_region[1].value) 

    flux_array_plot = ma.masked_where(mask, flux_array)
    wav_array_plot = ma.masked_where(mask, wav_array)
    err_array_plot = ma.masked_where(mask, err_array)
    vdat_array_plot = ma.masked_where(mask, vdat_array)   
    
    # get rid of nans / infs / negative errors 


    mask = (np.isnan(flux_array)) | (err_array < 0.0) 
    mask_norebin = (np.isnan(flux_array_norebin)) | (err_array_norebin < 0.0) 

    flux_array = ma.masked_where(mask, flux_array)
    wav_array = ma.masked_where(mask, wav_array)
    err_array = ma.masked_where(mask, err_array)
    vdat_array = ma.masked_where(mask, vdat_array)      

    flux_array_norebin = ma.masked_where(mask_norebin, flux_array_norebin)
    wav_array_norebin = ma.masked_where(mask_norebin, wav_array_norebin)
    err_array_norebin = ma.masked_where(mask_norebin, err_array_norebin)
    vdat_array_norebin = ma.masked_where(mask_norebin, vdat_array_norebin)      

    if reject_outliers:
               
        flux_array_smooth = medfilt2d(flux_array, kernel_size=(1, reject_width))  
        
        mask = (flux_array_smooth - flux_array) / err_array > reject_sigma 
    
        xy = np.argwhere(mask)
    
        bad_pix = np.concatenate(([(i[0], i[1] - 2) for i in xy],
                                  [(i[0], i[1] - 1) for i in xy],
                                  [(i[0], i[1]) for i in xy],
                                  [(i[0], i[1] + 1) for i in xy],
                                  [(i[0], i[1] + 2) for i in xy]))

   
        # if we go over the edge then remove these 
        bad_pix = bad_pix[(bad_pix[:,1] > 0) & (bad_pix[:,1] < wav_array.shape[1] - 1)]
        
        bad_pix = zip(*bad_pix)
    
        mask = np.zeros(flux_array.shape, dtype=bool)
        
        mask[bad_pix] = True 
    
        flux_array = ma.masked_where(mask, flux_array)
        wav_array = ma.masked_where(mask, wav_array)
        err_array = ma.masked_where(mask, err_array)
        vdat_array = ma.masked_where(mask, vdat_array)

        
        #------------------------------------------------------------------------

        flux_array_smooth_norebin = medfilt2d(flux_array_norebin, kernel_size=(1, reject_width))  
        
        mask_norebin = (flux_array_smooth_norebin - flux_array_norebin) / err_array_norebin > reject_sigma 
    
        xy_norebin = np.argwhere(mask_norebin)
    
        bad_pix_norebin = np.concatenate(([(i[0], i[1] - 2) for i in xy_norebin],
                                          [(i[0], i[1] - 1) for i in xy_norebin],
                                          [(i[0], i[1]) for i in xy_norebin],
                                          [(i[0], i[1] + 1) for i in xy_norebin],
                                          [(i[0], i[1] + 2) for i in xy_norebin]))

   
        # if we go over the edge then remove these 
        bad_pix_norebin = bad_pix_norebin[(bad_pix_norebin[:,1] > 0) & (bad_pix_norebin[:,1] < wav_array_norebin.shape[1] - 1)]
        
        bad_pix_norebin = zip(*bad_pix_norebin)
    
        mask_norebin = np.zeros(flux_array_norebin.shape, dtype=bool)
        
        mask_norebin[bad_pix_norebin] = True 
    
        flux_array_norebin = ma.masked_where(mask_norebin, flux_array_norebin)
        wav_array_norebin = ma.masked_where(mask_norebin, wav_array_norebin)
        err_array_norebin = ma.masked_where(mask_norebin, err_array_norebin)
        vdat_array_norebin = ma.masked_where(mask_norebin, vdat_array_norebin)

    if maskout is not None:

        for item in maskout:
                             
            mask = (ma.getdata(vdat_array).value > item[0].value) & (ma.getdata(vdat_array).value < item[1].value) 
    
            flux_array = ma.masked_where(mask, flux_array)
            wav_array = ma.masked_where(mask, wav_array)
            err_array = ma.masked_where(mask, err_array)
            vdat_array = ma.masked_where(mask, vdat_array)   

        for item in maskout:
                             
            mask_norebin = (ma.getdata(vdat_array_norebin).value > item[0].value) & (ma.getdata(vdat_array_norebin).value < item[1].value) 
    
            flux_array_norebin = ma.masked_where(mask_norebin, flux_array_norebin)
            wav_array_norebin = ma.masked_where(mask_norebin, wav_array_norebin)
            err_array_norebin = ma.masked_where(mask_norebin, err_array_norebin)
            vdat_array_norebin = ma.masked_where(mask_norebin, vdat_array_norebin)   

    if mask_negflux:

        mask = flux_array < 0.0  

        flux_array = ma.masked_where(mask, flux_array)
        wav_array = ma.masked_where(mask, wav_array)
        err_array = ma.masked_where(mask, err_array)
        vdat_array = ma.masked_where(mask, vdat_array)      

        mask_norebin = flux_array_norebin < 0.0  

        flux_array_norebin = ma.masked_where(mask_norebin, flux_array_norebin)
        wav_array_norebin = ma.masked_where(mask_norebin, wav_array_norebin)
        err_array_norebin = ma.masked_where(mask_norebin, err_array_norebin)
        vdat_array_norebin = ma.masked_where(mask_norebin, vdat_array_norebin)      

    # index of region for continuum fit 
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    # if fitting gauss hermite model for CIV, we also fit in the continuum regions to extrapolate through red shelf. 

    if fit_model == 'GaussHermite':

        mask = (ma.getdata(wav_array).value < continuum_region[0][0].value) |\
               ((ma.getdata(wav_array).value > continuum_region[0][1].value) & (ma.getdata(wav_array).value < fitting_region[0].value))|\
               ((ma.getdata(wav_array).value > fitting_region[1].value) & (ma.getdata(wav_array).value < continuum_region[1][0].value))|\
               (ma.getdata(wav_array).value > continuum_region[1][1].value) 

    else: 

        mask = (ma.getdata(wav_array).value < fitting_region[0].value) | (ma.getdata(wav_array).value > fitting_region[1].value)

    flux_array_fit = ma.masked_where(mask, flux_array)
    wav_array_fit = ma.masked_where(mask, wav_array)
    err_array_fit = ma.masked_where(mask, err_array)
    vdat_array_fit = ma.masked_where(mask, vdat_array)

    
    blue_mask = (ma.getdata(wav_array).value < continuum_region[0][0].value) | (ma.getdata(wav_array).value > continuum_region[0][1].value)
    red_mask = (ma.getdata(wav_array).value < continuum_region[1][0].value) | (ma.getdata(wav_array).value > continuum_region[1][1].value)   

    flux_array_blue = ma.masked_where(blue_mask, flux_array)
    wav_array_blue = ma.masked_where(blue_mask, wav_array)
    err_array_blue = ma.masked_where(blue_mask, err_array)
    vdat_array_blue = ma.masked_where(blue_mask, vdat_array)

    flux_array_red = ma.masked_where(red_mask, flux_array)
    wav_array_red = ma.masked_where(red_mask, wav_array)
    err_array_red = ma.masked_where(red_mask, err_array)
    vdat_array_red = ma.masked_where(red_mask, vdat_array)

    #------------------------------------------------------------------------------------------------------------------------------

    blue_mask_norebin = (ma.getdata(wav_array_norebin).value < continuum_region[0][0].value) | (ma.getdata(wav_array_norebin).value > continuum_region[0][1].value)
    red_mask_norebin = (ma.getdata(wav_array_norebin).value < continuum_region[1][0].value) | (ma.getdata(wav_array_norebin).value > continuum_region[1][1].value)   

    flux_array_blue_norebin = ma.masked_where(blue_mask_norebin, flux_array_norebin)
    wav_array_blue_norebin = ma.masked_where(blue_mask_norebin, wav_array_norebin)
    err_array_blue_norebin = ma.masked_where(blue_mask_norebin, err_array_norebin)
    vdat_array_blue_norebin = ma.masked_where(blue_mask_norebin, vdat_array_norebin)

    flux_array_red_norebin = ma.masked_where(red_mask_norebin, flux_array_norebin)
    wav_array_red_norebin = ma.masked_where(red_mask_norebin, wav_array_norebin)
    err_array_red_norebin = ma.masked_where(red_mask_norebin, err_array_norebin)
    vdat_array_red_norebin = ma.masked_where(red_mask_norebin, vdat_array_norebin)


    #############################################################################################

    spec_dv = np.around(np.mean(const.c.to('km/s') * (1.0 - 10.0**-np.diff(np.log10(wav[(wav > fitting_region[0]) & (wav < fitting_region[1])].value)))), decimals=1)
    spec_dv_norebin = np.around(np.mean(const.c.to('km/s') * (1.0 - 10.0**-np.diff(np.log10(wav_norebin[(wav_norebin > fitting_region[0]) & (wav_norebin < fitting_region[1])].value)))), decimals=1)

    #############################################################################################

    # if no flux in continuum region 
    if len(ma.getdata(wav_array_blue[0, ~ma.getmaskarray(wav_array_blue[0, :])]).value) + len(ma.getdata(wav_array_red[0, ~ma.getmaskarray(wav_array_red[0, :])]).value) == 0:

        if verbose:
            print 'No flux in continuum region'
    
        if n_samples == 1:
        
            fit_out = {'name':plot_title, 
                       'fwhm':np.nan,
                       'fwhm_1':np.nan,
                       'fwhm_2':np.nan,
                       'fwhm_3':np.nan,
                       'sigma':np.nan,
                       'median':np.nan,
                       'cen':np.nan,
                       'cen_1':np.nan,
                       'cen_2':np.nan,
                       'cen_3':np.nan,
                       'eqw':np.nan,
                       'broad_lum':np.nan,
                       'peak':np.nan, 
                       'amplitude_1':np.nan,
                       'amplitude_2':np.nan,
                       'amplitude_3':np.nan,
                       'narrow_fwhm':np.nan,
                       'narrow_lum':np.nan,
                       'narrow_voff':np.nan, 
                       'oiii_5007_eqw':np.nan,
                       'oiii_5007_lum':np.nan,
                       'oiii_5007_n_lum':np.nan,
                       'oiii_5007_b_lum':np.nan,
                       'oiii_fwhm':np.nan,
                       'oiii_n_fwhm':np.nan,
                       'oiii_b_fwhm':np.nan,
                       'oiii_5007_b_voff':np.nan,
                       'oiii_5007_blueshift':np.nan,
                       'redchi':np.nan,
                       'snr':np.nan,
                       'dv':np.nan, 
                       'monolum':np.nan,
                       'fe_ew':np.nan}
            
            if save_dir is not None:
        
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
        
                with open(os.path.join(save_dir, 'fit.txt'), 'w') as f:
                    f.write('No fit')               
        
            if plot: 
                plot_spec(wav=wav,
                          flux=flux,
                          plot_savefig = plot_savefig,
                          save_dir = save_dir,
                          z=z,
                          w0=w0,
                          continuum_region=continuum_region,
                          fitting_region=fitting_region,
                          plot_region=plot_region,
                          plot_title=plot_title,
                          verbose=verbose)
        
            
            return fit_out 
    
        else:

            if save_dir is not None:
                df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 
    
            return None 

    """
    Calculate S/N ratio per pixel 
    Do this for the un-rebinned spectra  
    """
   
    wa = ma.concatenate((wav_array_blue_norebin, wav_array_red_norebin), axis=1)
    fl = ma.concatenate((flux_array_blue_norebin, flux_array_red_norebin), axis=1)
    er = ma.concatenate((err_array_blue_norebin, err_array_red_norebin), axis=1)

    mask = (er < 0.0) | np.isnan(fl)
    
    fl = ma.masked_where(mask, fl)
    er = ma.masked_where(mask, er)
    wa = ma.masked_where(mask, wa)
    
    snr = ma.median(fl / er, axis=1)

    # 33.02 is the A per resolution element I measured from the Arc spectrum for Liris 
    if verbose:
        if n_samples == 1:
        # print 'S/N per resolution element in continuum: {0:.2f}'.format(np.median( np.sqrt(33.02 / np.diff(wa) ) * fl[:-1] / er[:-1] )) 
            print 'S/N per pixel in continuum: {0:.2f}'.format(snr[0]) 
        
    ##############################################################################################  

    if bkgd_median is True:

        xdat_cont = np.array([ma.mean(wav_array_blue, axis=1), ma.mean(wav_array_red, axis=1)]).T
        ydat_cont = np.array([ma.median(flux_array_blue, axis=1), ma.median(flux_array_red, axis=1)]).T

        if subtract_fe is True:
            print 'Cant fit iron if bkgd_median is True' 

        elif subtract_fe is False:

            #-------------------------------------------------------------------------------------------------
            
            fitobj = []
            for k in range(n_samples):
                fitobj.append([k, xdat_cont[k], ydat_cont[k], flux_array_fit[k, :], flux_array_plot[k, :], wav_array_fit[k, :], wav_array_plot[k, :]])

            fit1_p = partial(fit1, 
                             n_samples = n_samples, 
                             plot_title = plot_title, 
                             verbose = verbose, 
                             mono_lum_wav = mono_lum_wav, 
                             spec_norm = spec_norm, 
                             z = z)

            mono_lum, eqw_fe = np.zeros(n_samples), np.zeros(n_samples) 
            bkgdpars = []

            if parallel:
                
                p = Pool(cores)
                out = p.map(fit1_p, fitobj)
                p.close() 
                p.join()

            else:
                
                out = map(fit1_p, fitobj)

            for k, o in enumerate(out):

                flux_array_fit[k, :] = o[0]
                flux_array_plot[k, :] = o[1]
                mono_lum[k] = o[2].value
                eqw_fe[k] = o[3]
                bkgdpars.append(o[4])

            
            # mono_lum, eqw_fe = np.zeros(n_samples), np.zeros(n_samples)
            # for k, (x, y) in enumerate(zip(xdat_cont, ydat_cont)):

            #     if n_samples > 1: 
            #         print plot_title, k 

            #     bkgdpars['exponent'].value = 1.0
            #     bkgdpars['amplitude'].value = 1.0 
                
            #     out = minimize(resid,
            #                    bkgdpars,
            #                    kws={'x':x, 
            #                         'model':bkgdmod, 
            #                         'data':y},
            #                    method='leastsq')

            #     flux_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])] = flux_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])] - resid(params=bkgdpars, x=ma.getdata(wav_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])]).value, model=bkgdmod)
            #     flux_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])] = flux_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])] - resid(params=bkgdpars, x=ma.getdata(wav_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])]).value, model=bkgdmod)                  

            #     if verbose:
            #         if n_samples == 1:
            #             print out.message  
            #             print fit_report(bkgdpars)

            #     ####################################################################################################################
            #     """
            #     Calculate flux at wavelength mono_lum_wav
            #     """
            #     # Calculate just power-law continuum (no Fe)
            #     cont_mod = Model(PLModel, 
            #                      param_names=['amplitude','exponent'], 
            #                      independent_vars=['x']) 

            #     cont_pars = cont_mod.make_params()
            #     cont_pars['exponent'].value = bkgdpars['exponent'].value
            #     cont_pars['amplitude'].value = bkgdpars['amplitude'].value  

            #     mono_flux = resid(params=cont_pars, 
            #                       x=[mono_lum_wav.value], 
            #                       model=cont_mod)[0]
                
            #     mono_flux = mono_flux / spec_norm
            #     mono_flux = mono_flux * (u.erg / u.cm / u.cm / u.s / u.AA)
            #     lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)

            #     mono_lum[k] =  (mono_flux * (1.0 + z) * 4.0 * math.pi * lumdist**2 * mono_lum_wav).value

            #     eqw_fe[k] = 0.0 

            # #     ######################################################################################################################

            #     #----------------------------------------------------------------------------------------------------------------------
    
    if bkgd_median is False:

        # concatentate drops the units 
        xdat_cont = ma.concatenate((wav_array_blue, wav_array_red), axis=1)
        ydat_cont = ma.concatenate((flux_array_blue, flux_array_red), axis=1)
        yerr_cont = ma.concatenate((err_array_blue, err_array_red), axis=1)

        fitobj = []

        for k in range(n_samples):
            fitobj.append([k, xdat_cont[k], ydat_cont[k], yerr_cont[k], flux_array_fit[k, :], flux_array_plot[k, :], wav_array_fit[k, :], wav_array_plot[k, :]])

        fit2_p = partial(fit2, 
                         n_samples = n_samples, 
                         subtract_fe = subtract_fe, 
                         home_dir = home_dir, 
                         fe_FWHM = fe_FWHM, 
                         fe_FWHM_vary = fe_FWHM_vary, 
                         spec_norm = spec_norm, 
                         mono_lum_wav = mono_lum_wav, 
                         z = z,
                         wav = wav, 
                         flux = flux, 
                         err = err, 
                         verbose = verbose, 
                         wav_array_blue = wav_array_blue, 
                         flux_array_blue = flux_array_blue, 
                         wav_array_red = wav_array_red, 
                         flux_array_red = flux_array_red, 
                         reject_outliers = reject_outliers, 
                         w0 = w0, 
                         maskout = maskout, 
                         continuum_region = continuum_region,
                         plot_title = plot_title,
                         plot = plot) 



        mono_lum, eqw_fe = np.zeros(n_samples), np.zeros(n_samples) 
        bkgdpars = []

        if parallel:
            
            p = Pool(cores)
            out = p.map(fit2_p, fitobj)
            p.close() 
            p.join()

        else:
            
            out = map(fit2_p, fitobj)

        for k, o in enumerate(out):

            flux_array_fit[k, :] = o[0]
            flux_array_plot[k, :] = o[1]
            mono_lum[k] = o[2].value
            eqw_fe[k] = o[3]
            bkgdpars.append(o[4])



    if (n_samples > 1) & (plot is True):

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

        for item in ma.extras.flatnotmasked_contiguous(vdat_array_fit[0, :].flatten()):
            fit.axvspan(vdat_array_fit[0, item].min(), vdat_array_fit[0, item].max(), color='moccasin', alpha=0.4)
            residuals.axvspan(vdat_array_fit[0, item].min(), vdat_array_fit[0, item].max(), color='moccasin', alpha=0.4)

        fit.set_ylabel(r'F$_\lambda$', fontsize=12)
        residuals.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=12)
        residuals.set_ylabel("Residual")

        if plot_region.unit == (u.km/u.s):
            plot_region = doppler2wave(plot_region, w0)

        plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])
        plotting_limits = wave2doppler(plot_region, w0) 
        fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)
        residuals.set_xlim(fit.get_xlim())    

        fit.set_ylim(ma.median(flux_array_fit[k, :]) - 2.0*ma.std(flux_array_fit[k, :]), ma.median(flux_array_fit[k, :]) + 4.0*ma.std(flux_array_fit[k, :]))
        residuals.set_ylim(-6, 6)

    else:

        fig, fit, residuals = None, None, None 


    fitobj = []

    for k in range(n_samples):
        fitobj.append([k, 
                       vdat_array_fit[k, :], 
                       flux_array_fit[k, :], 
                       err_array_fit[k, :], 
                       wav_array_plot[k, :], 
                       vdat_array_plot[k, :], 
                       flux_array_plot[k, :], 
                       err_array_plot[k, :], 
                       bkgdpars[k],  
                       mono_lum[k], 
                       eqw_fe[k],
                       snr[k]])

    fit3_p = partial(fit3,
                     fit_model = fit_model, 
                     nGaussians = nGaussians, 
                     gh_order = gh_order, 
                     fitting_region = fitting_region, 
                     ha_narrow_fwhm = ha_narrow_fwhm, 
                     ha_narrow_vary = ha_narrow_vary, 
                     ha_narrow_voff = ha_narrow_voff, 
                     w0 = w0, 
                     hb_narrow = hb_narrow, 
                     n_samples = n_samples, 
                     plot = plot, 
                     continuum_region = continuum_region, 
                     fitting_method = fitting_method, 
                     save_dir = save_dir, 
                     subtract_fe = subtract_fe, 
                     home_dir = home_dir,
                     z = z, 
                     line_region = line_region, 
                     spec_norm = spec_norm, 
                     plot_title = plot_title, 
                     verbose = verbose,
                     spec_dv = spec_dv,
                     wav = wav, 
                     flux = flux,
                     err = err,
                     mono_lum_wav = mono_lum_wav,
                     emission_line = emission_line,
                     plot_savefig = plot_savefig,
                     maskout = maskout,
                     plot_region = plot_region,
                     mask_negflux = mask_negflux,
                     reject_outliers = reject_outliers,
                     reject_width = reject_width,
                     reject_sigma = reject_sigma,
                     fig = fig, 
                     fit = fit,
                     residuals = residuals,
                     fix_broad_peaks = fix_broad_peaks,
                     oiii_broad_off = oiii_broad_off)

    if parallel:
        
        p = Pool(cores)
        out = p.map(fit3_p, fitobj)
        p.close() 
        p.join()

    else:
        
        out = map(fit3_p, fitobj)

    
    if (save_dir is not None) & (n_samples > 1):
        
        for k in range(n_samples):
            for col in df_out:
                df_out.loc[k, col] = out[k][col]

      
        df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 
         
    else:

        return out[0]
         
         
         
         
         

        # if subtract_fe is True:

        #     ################################################################################
        #     """
        #     Optical FeII template
        
        #     Broaden by convolution with Gaussian using astropy 
        #     Add to power-law with normalisation
        #     Allow shifting as well?
        
        #     Shen does all fits in logarithmic wavelength space. 
        #     Does this make a difference? 
        
        #     Convolve with width of Balmer line - might have to do iteratively 
        
        #     Either use FWHM from fit to Ha or leave convolution as a free paramter 
        #     See how this compares. 
        #     """

        #     fname = os.path.join(home_dir,'SpectraTools/irontemplate.dat')
        #     fe_wav, fe_flux = np.genfromtxt(fname, unpack=True)
        
        #     fe_flux = fe_flux / np.median(fe_flux)
    
        #     sp_fe = spec.Spectrum(wa=10**fe_wav, fl=fe_flux)

        #     bkgdmod = Model(PseudoContinuum, 
        #                     param_names=['amplitude',
        #                                  'exponent',
        #                                  'fe_norm',
        #                                  'fe_sd',
        #                                  'fe_shift'], 
        #                     independent_vars=['x']) 
            
        #     bkgdpars = bkgdmod.make_params() 
            
        #     fe_sigma = fe_FWHM / 2.35 
        #     fe_pix = fe_sigma / (sp_fe.dv * (u.km/u.s))

        #     bkgdpars['fe_sd'].vary = fe_FWHM_vary

        #     bkgdpars['fe_norm'].min = 0.0
        #     bkgdpars['fe_sd'].min = 2000.0 / 2.35 / sp_fe.dv
        #     bkgdpars['fe_sd'].max = 12000.0 / 2.35 / sp_fe.dv

        #     bkgdpars['fe_shift'].min = -20.0
        #     bkgdpars['fe_shift'].max = 20.0

        #     # I think that constraining the power-law slope to be negative
        #     # helps break some of the degeneracy between the fe template
        #     # and the continuum. I don't want to change any of my fits
        #     # so for now just do this for the pseudo-continuum fit. 

        #     # Seems to break if max is set to 0.0

        #     # if pseudo_continuum_fit:

        #     #     bkgdpars['exponent'].max = 0.5 


        # if subtract_fe is False:

        #     bkgdmod = Model(PLModel, 
        #                     param_names=['amplitude','exponent'], 
        #                     independent_vars=['x']) 

        #     bkgdpars = bkgdmod.make_params()
            
        #-------------------------------------------------------------------------------------
    #     mono_lum, eqw_fe = np.zeros(n_samples), np.zeros(n_samples)
    #     for k, (x, y, er) in enumerate(zip(xdat_cont, ydat_cont, yerr_cont)):  

    #         if len(ma.getdata(x[~ma.getmaskarray(x)]).value) == 0: 
    #             if verbose:
    #                 print 'No flux in continuum region'

    #         if subtract_fe is True:

    #             bkgdpars['fe_sd'].value = fe_pix.value  
    #             bkgdpars['exponent'].value = -1.0
    #             bkgdpars['amplitude'].value = 1.0
    #             bkgdpars['fe_norm'].value = 0.05 
    #             bkgdpars['fe_shift'].value = 0.0

    #             # print bkgdpars['fe_sd'].value * 2.35 * sp_fe.dv 

    #             out = minimize(resid,
    #                            bkgdpars,
    #                            kws={'x':ma.getdata(x[~ma.getmaskarray(x)]).value, 
    #                                 'model':bkgdmod, 
    #                                 'data':ma.getdata(y[~ma.getmaskarray(y)]), 
    #                                 'sigma':ma.getdata(er[~ma.getmaskarray(er)]), 
    #                                 'sp_fe':sp_fe},
    #                            method='slsqp')  

    #             flux_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])] = flux_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])] - resid(params=bkgdpars, x=ma.getdata(wav_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])]).value, model=bkgdmod, sp_fe=sp_fe)
    #             flux_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])] = flux_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])] - resid(params=bkgdpars, x=ma.getdata(wav_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])]).value, model=bkgdmod, sp_fe=sp_fe)                  

            
    #             ##############################################################################
    #             """
    #             Calculate EQW of Fe between 4435 and 4684A - same as Shen (2016)
    #             """
    #             fe_fl = PseudoContinuum(np.arange(4434,4684,1), 
    #                                     0.0,
    #                                     0.0,
    #                                     bkgdpars['fe_norm'].value,
    #                                     bkgdpars['fe_sd'].value,
    #                                     bkgdpars['fe_shift'].value,
    #                                     sp_fe)

    #             fe_fl = fe_fl / spec_norm 
    #             fe_fl = fe_fl * (u.erg / u.cm / u.cm / u.s / u.AA) 

    #             bg_fl = PseudoContinuum(np.arange(4434,4684,1), 
    #                                     bkgdpars['amplitude'].value,
    #                                     bkgdpars['exponent'].value,
    #                                     0.0,
    #                                     bkgdpars['fe_sd'].value,
    #                                     bkgdpars['fe_shift'].value,
    #                                     sp_fe)      

    #             bg_fl = bg_fl / spec_norm 
    #             bg_fl = bg_fl * (u.erg / u.cm / u.cm / u.s / u.AA) 
    
    #             f = (fe_fl + bg_fl) / bg_fl
    #             eqw_fe[k] = np.nansum((f[:-1] - 1.0) * 1.0) 

    #             if verbose:
    #                 print 'Fe EW: {}'.format(eqw_fe[k])  



    #             ###############################################################################



    #         if subtract_fe is False:


    #             bkgdpars['exponent'].value = 1.0
    #             bkgdpars['amplitude'].value = 1.0 

    #             out = minimize(resid,
    #                            bkgdpars,
    #                            kws={'x':ma.getdata(x[~ma.getmaskarray(x)]).value, 
    #                                 'model':bkgdmod, 
    #                                 'data':ma.getdata(y[~ma.getmaskarray(y)]),
    #                                 'sigma':ma.getdata(er[~ma.getmaskarray(er)])},  
    #                            method='leastsq') 


    #             flux_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])] = flux_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])] - resid(params=bkgdpars, x=ma.getdata(wav_array_fit[k, ~ma.getmaskarray(wav_array_fit[k ,:])]).value, model=bkgdmod)
    #             flux_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])] = flux_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])] - resid(params=bkgdpars, x=ma.getdata(wav_array_plot[k, ~ma.getmaskarray(wav_array_plot[k ,:])]).value, model=bkgdmod)                  
            
    #             eqw_fe[k] = 0.0

    #         ####################################################################################################################
    #         """
    #         Calculate flux at wavelength mono_lum_wav
    #         """

    #         # Calculate just power-law continuum (no Fe)
    #         cont_mod = Model(PLModel, 
    #                          param_names=['amplitude','exponent'], 
    #                          independent_vars=['x']) 

    #         cont_pars = cont_mod.make_params()
    #         cont_pars['exponent'].value = bkgdpars['exponent'].value
    #         cont_pars['amplitude'].value = bkgdpars['amplitude'].value  

    #         mono_flux = resid(params=cont_pars, 
    #                           x=[mono_lum_wav.value], 
    #                           model=cont_mod)[0]

            
    #         mono_flux = mono_flux / spec_norm

    #         mono_flux = mono_flux * (u.erg / u.cm / u.cm / u.s / u.AA)

    #         lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)

    #         mono_lum[k] = (mono_flux * (1.0 + z) * 4.0 * math.pi * lumdist**2 * mono_lum_wav).value

    #         if n_samples > 1: 
    #             print plot_title, k, mono_lum[k] 


    #         ######################################################################################################################

    #         if pseudo_continuum_fit:
    
    #             fit_out = {'name':plot_title,
    #                        'monolum':np.log10(mono_lum.value),
    #                        'fe_ew':eqw_fe} 
        
    #         if n_samples == 1:  

    #             if verbose:

    #                 print out.message, 'chi-squared = {}'.format(out.redchi)
    #                 print fit_report(bkgdpars)
      
    #                 if subtract_fe is True:
    #                     print 'Fe FWHM = {0:.1f} km/s (initial = {1:.1f} km/s)'.format(bkgdpars['fe_sd'].value * 2.35 * sp_fe.dv, bkgdpars['fe_sd'].init_value * 2.35 * sp_fe.dv)            
                

    #             if plot:
           
    #                 """
    #                 Plotting continuum / iron fit
    #                 """

    #                 xdat_cont_plot = ma.getdata(xdat_cont[~ma.getmaskarray(xdat_cont)]).value
    #                 ydat_cont_plot = ma.getdata(ydat_cont[~ma.getmaskarray(ydat_cont)])
    #                 yerr_cont_plot = ma.getdata(yerr_cont[~ma.getmaskarray(yerr_cont)])
            
    #                 fig, axs = plt.subplots(4, 1, figsize=(20,20))
        
    #                 axs[0].errorbar(wav.value, 
    #                                 flux, 
    #                                 yerr=err, 
    #                                 linestyle='', 
    #                                 alpha=0.5,
    #                                 color='grey')
        
    #                 axs[1].errorbar(wav.value, 
    #                                 median_filter(flux,51), 
    #                                 color='grey')
            
    #                 xdat_plotting = np.arange(xdat_cont_plot.min(), xdat_cont_plot.max(), 1)

    #                 if subtract_fe is True: 
            
    #                     axs[0].plot(xdat_plotting, 
    #                                 resid(params=bkgdpars, 
    #                                       x=xdat_plotting, 
    #                                       model=bkgdmod, 
    #                                       sp_fe=sp_fe), 
    #                                 color='black', 
    #                                 lw=2)
                
    #                     axs[1].plot(xdat_plotting, 
    #                                 resid(params=bkgdpars, 
    #                                       x=xdat_plotting, 
    #                                       model=bkgdmod, 
    #                                       sp_fe=sp_fe), 
    #                                 color='black', 
    #                                 lw=2)
                
    #                     gauss = Gaussian1DKernel(stddev=bkgdpars['fe_sd'].value) 
            
    #                     fe_flux = convolve(sp_fe.fl, gauss)
    #                     fe_flux = np.roll(fe_flux, int(bkgdpars['fe_shift'].value))
    #                     f = interp1d(sp_fe.wa, fe_flux) 
    #                     fe_flux = f(xdat_cont_plot)   
    
    #                     axs[0].plot(xdat_cont_plot, 
    #                                 bkgdpars['fe_norm'].value * fe_flux, 
    #                                 color='red')   

                        
    #                     amp = bkgdpars['amplitude'].value * 5000.0**(-bkgdpars['exponent'].value)    
                        
    #                     axs[0].plot(xdat_cont_plot, 
    #                                 amp*xdat_cont_plot**bkgdpars['exponent'].value, 
    #                                 color='blue')   


    #                     fe_flux = np.roll(sp_fe.fl, int(bkgdpars['fe_shift'].value))
                       
    #                     axs[1].plot(sp_fe.wa, 
    #                                 bkgdpars['fe_norm'].value * fe_flux, 
    #                                 color='red')  


    #                 if subtract_fe is False: 

    #                     axs[0].plot(xdat_plotting, 
    #                                 resid(params=bkgdpars, 
    #                                       x=xdat_plotting, 
    #                                       model=bkgdmod), 
    #                                 color='black', 
    #                                 lw=2)
                
    #                     axs[1].plot(xdat_plotting, 
    #                                 resid(params=bkgdpars, 
    #                                       x=xdat_plotting, 
    #                                       model=bkgdmod),
    #                                 color='black', 
    #                                 lw=2)
 
    #                     bkgdpars_tmp = bkgdpars.copy() 


    #                     axs[1].plot(xdat_plotting, 
    #                                 resid(params=bkgdpars, 
    #                                       x=xdat_plotting, 
    #                                       model=bkgdmod),
    #                                 color='black', 
    #                                 lw=2)
                    

            
    #                 axs[0].set_xlim(xdat_cont_plot.min() - 50.0, xdat_cont_plot.max() + 50.0)
    #                 axs[1].set_xlim(axs[0].get_xlim())

    #                 if subtract_fe is True: 
                    
    #                     func_vals = resid(params=bkgdpars, 
    #                                       x=xdat_cont_plot, 
    #                                       model=bkgdmod, 
    #                                       sp_fe=sp_fe)

    #                 if subtract_fe is False: 

    #                     func_vals = resid(params=bkgdpars, 
    #                                       x=xdat_cont_plot, 
    #                                       model=bkgdmod)
            
    #                 axs[0].set_ylim(np.median(ydat_cont_plot) - 3.0*np.std(ydat_cont_plot), np.median(ydat_cont_plot) + 3.0*np.std(ydat_cont_plot))
    #                 axs[1].set_ylim(axs[0].get_ylim())
                    
    #                 xdat_masking = np.arange(wav.min().value, wav.max().value, 0.05)*(u.AA)
    #                 vdat_masking = wave2doppler(xdat_masking, w0)
                    
    #                 mask = (xdat_masking.value < continuum_region[0][0].value) | \
    #                        ((xdat_masking.value > continuum_region[0][1].value) &  (xdat_masking.value < continuum_region[1][0].value))  | \
    #                        (xdat_masking.value > continuum_region[1][1].value)
                    
    #                 if maskout is not None:
    #                     for item in maskout:
    #                         mask = mask | ((vdat_masking.value > item.value[0]) & (vdat_masking.value < item.value[1]))
                    
    #                 vdat1_masking = ma.array(vdat_masking.value)
    #                 vdat1_masking[mask] = ma.masked 
            
    #                 for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
                    
    #                     axs[0].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
    #                                doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
    #                                alpha=0.4, 
    #                                color='powderblue')    
                  
    #                     axs[1].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
    #                                doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
    #                                alpha=0.4, 
    #                                color='powderblue')    
                   
    #                     axs[2].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
    #                                doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
    #                                alpha=0.4, 
    #                                color='powderblue')    
                    
    #                     axs[3].axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, 
    #                                doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, 
    #                                alpha=0.4, 
    #                                color='powderblue')    
        
    #                 if subtract_fe is True: 
                    
    #                     axs[2].scatter(xdat_cont_plot, 
    #                                   (ydat_cont_plot - resid(params=bkgdpars, 
    #                                                           x=xdat_cont_plot, 
    #                                                           model=bkgdmod, 
    #                                                           sp_fe=sp_fe)) / yerr_cont_plot, 
    #                                   edgecolor='None', 
    #                                   s=15, 
    #                                   facecolor='black')

    #                 if subtract_fe is False: 

    #                     axs[2].scatter(xdat_cont_plot, 
    #                                   (ydat_cont_plot - resid(params=bkgdpars, 
    #                                                           x=xdat_cont_plot, 
    #                                                           model=bkgdmod)) / yerr_cont_plot, 
    #                                   edgecolor='None', 
    #                                   s=15, 
    #                                   facecolor='black')

                    
    #                 axs[2].axhline(0.0, color='black', linestyle='--')
    #                 axs[2].set_xlim(axs[0].get_xlim())
            
    #                 axs[3].plot(wav.value, flux, color='grey')

    #                 if subtract_fe is True:
                    
    #                     axs[3].plot(xdat_cont_plot, 
    #                                 resid(params=bkgdpars, 
    #                                       x=xdat_cont_plot, 
    #                                       model=bkgdmod, 
    #                                       sp_fe=sp_fe), 
    #                                 color='black', 
    #                                 lw=2)

    #                 if subtract_fe is False: 

    #                     axs[3].plot(xdat_cont_plot, 
    #                                 resid(params=bkgdpars, 
    #                                       x=xdat_cont_plot, 
    #                                       model=bkgdmod), 
    #                                 color='black', 
    #                                 lw=2)

    #                 if reject_outliers: 

    #                     mask = ma.getmaskarray(wav_array_blue).flatten() 
            
    #                     mask[ma.getdata(wav_array_blue.flatten()).value < continuum_region[0][0].value] = False
    #                     mask[ma.getdata(wav_array_blue.flatten()).value > continuum_region[0][1].value] = False
            
    #                     xdat_cont_outliers = ma.masked_where(~mask, ma.getdata(wav_array_blue).flatten())
    #                     ydat_cont_outliers = ma.masked_where(~mask, ma.getdata(flux_array_blue).flatten())
            
    #                     axs[0].scatter(xdat_cont_outliers,
    #                                    ydat_cont_outliers,
    #                                    color='firebrick', 
    #                                    s=40, 
    #                                    marker='x') 
            
    #                     mask = ma.getmaskarray(wav_array_red).flatten() 
            
    #                     mask[ma.getdata(wav_array_red.flatten()).value < continuum_region[1][0].value] = False
    #                     mask[ma.getdata(wav_array_red.flatten()).value > continuum_region[1][1].value] = False
            
    #                     xdat_cont_outliers = ma.masked_where(~mask, ma.getdata(wav_array_red).flatten())
    #                     ydat_cont_outliers = ma.masked_where(~mask, ma.getdata(flux_array_red).flatten())
            
    #                     axs[0].scatter(xdat_cont_outliers,
    #                                    ydat_cont_outliers,
    #                                    color='firebrick', 
    #                                    s=40, 
    #                                    marker='x') 

    #                 if verbose:
            
    #                     global coords
    #                     coords = [] 
    #                     cid = fig.canvas.mpl_connect('button_press_event', lambda x: onclick2(x, w0=w0))
            
    #                     plt.show(1)
            
    #                 plt.close() 

    # if pseudo_continuum_fit:

    #     """
    #     Only fitting PseudoContinuum
    #     """

    #     if n_samples == 1:       

    #         return fit_out

    #     else: 
       
    #         if save_dir is not None:
    #             df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 

    #         return None 

       

    # Rescale x axis so everything is of order unity.
    # y axis has already been scaled  

    # if fit_model == 'MultiGauss':

    #     xscale = 1.0 

    #     # Make model 
    #     bkgd = ConstantModel()
    #     mod = bkgd
    #     pars = bkgd.make_params()
        
    #     # A bit unnessesary, but I need a way to do += in the loop below
    #     pars['c'].value = 0.0
    #     pars['c'].vary = False


    #     for i in range(nGaussians):
    #         gmod = GaussianModel(prefix='g{}_'.format(i))
    #         mod += gmod
    #         pars += gmod.guess(flux_array_fit[0,~ma.getmaskarray(flux_array_fit[0,:])], x=ma.getdata(vdat_array_fit[0,~ma.getmaskarray(vdat_array_fit[0,:])]).value)


               
    #     for i in range(nGaussians):
    #         pars['g{}_center'.format(i)].value = 0.0
    #         pars['g{}_center'.format(i)].min = -2000.0
    #         pars['g{}_center'.format(i)].max = 2000.0 
    #         pars['g{}_amplitude'.format(i)].min = 0.0
    #         # pars['g{}_sigma'.format(i)].min = 1000.0 # sometimes might be better if this is relaxed 
    #         # pars['g{}_sigma'.format(i)].max = 10000.0 

    #     # pars['g0_center'].set(expr='g1_center') 

    # elif fit_model == 'GaussHermite':

    #     # Calculate mean and variance

    #     """
    #     Because p isn't a real probability distribution we sometimes get negative 
    #     variances if a lot of the flux is negative. 
    #     Therefore confine this to only work on the bit of the 
    #     spectrum that is positive
    #     """

    #     ydat = ma.getdata(flux_array_fit[0,~ma.getmaskarray(flux_array_fit[0,:])])
    #     vdat = ma.getdata(vdat_array_fit[0,~ma.getmaskarray(vdat_array_fit[0,:])])


    #     p = ydat[ydat > 0.0] / np.sum(ydat[ydat > 0.0])
    #     m = np.sum(vdat[ydat > 0.0] * p)
    #     v = np.sum(p * (vdat[ydat > 0.0]-m)**2)
    #     xscale = np.sqrt(v).value 

    #     param_names = []

    #     for i in range(gh_order + 1):
            
    #         param_names.append('amp{}'.format(i))
    #         param_names.append('sig{}'.format(i))
    #         param_names.append('cen{}'.format(i))

    #     if gh_order == 0: 
 
    #         mod = Model(gausshermite_0, independent_vars=['x'], param_names=param_names) 
    
    #     if gh_order == 1: 
 
    #         mod = Model(gausshermite_1, independent_vars=['x'], param_names=param_names) 
     
    #     if gh_order == 2: 
 
    #         mod = Model(gausshermite_2, independent_vars=['x'], param_names=param_names) 
     
    #     if gh_order == 3: 
 
    #         mod = Model(gausshermite_3, independent_vars=['x'], param_names=param_names) 
     
    #     if gh_order == 4: 
 
    #         mod = Model(gausshermite_4, independent_vars=['x'], param_names=param_names) 
     
    #     if gh_order == 5: 
 
    #         mod = Model(gausshermite_5, independent_vars=['x'], param_names=param_names) 

    #     if gh_order == 6: 
 
    #         mod = Model(gausshermite_6, independent_vars=['x'], param_names=param_names) 

    #     pars = mod.make_params()

    #     for i in range(gh_order + 1):

    #         pars['amp{}'.format(i)].value = 1.0
    #         pars['sig{}'.format(i)].value = 1.0
    #         pars['cen{}'.format(i)].value = 0.0
 
    #         # remember that wav_fitting_array mask includes the background windows, 
    #         # so don't use this 
    #         # already converted fitting_region in to units of wavelength. 

    #         pars['cen{}'.format(i)].min = wave2doppler(fitting_region[0], w0).value / xscale
    #         pars['cen{}'.format(i)].max = wave2doppler(fitting_region[1], w0).value / xscale

    #         pars['sig{}'.format(i)].min = 0.1

    #         pars['amp{}'.format(i)].min = 0.0

        
        
    # elif fit_model == 'Ha':

    #     """
    #     Implement the Shen+15/11 fitting procedure
    #     The narrow components of Hα, [NII]λλ6548,6584, [SII]λλ6717,6731 are each fit with a single Gaussian. 
    #     Their velocity offsets from the systemic redshift and line widths are constrained to be the same
    #     The relative flux ratio of the two [NII] components is fixed to 2.96 - which way round is this? 
    #     We impose an upper limit on the narrow line FWHM < 1200 km/s 
    #     The broad Hα component is modelled in two different ways: 
    #     a) a single Gaussian with a FWHM > 1200 km/s; 
    #     b) multiple Gaussians with up to three Gaussians, each with a FWHM >1200 km/s. 
    #     The second method yields similar results to the fits with a truncated Gaussian-Hermite function. 
    #     During the fitting, all lines are restricted to be emission lines (i.e., positive flux)

    #     Also fit Boroson and Green iron template 

    #     """

    #     xscale = 1.0 

    #     mod = GaussianModel(prefix='ha_n_')  

    #     mod += GaussianModel(prefix='nii_6548_n_')

    #     mod += GaussianModel(prefix='nii_6584_n_')

    #     mod += GaussianModel(prefix='sii_6717_n_')

    #     mod += GaussianModel(prefix='sii_6731_n_')


    #     for i in range(nGaussians):

    #         mod += GaussianModel(prefix='ha_b_{}_'.format(i))  

    #     pars = mod.make_params() 

    #     pars['nii_6548_n_amplitude'].value = 1000.0
    #     pars['nii_6584_n_amplitude'].value = 1000.0
    #     pars['sii_6717_n_amplitude'].value = 1000.0 
    #     pars['sii_6731_n_amplitude'].value = 1000.0 
    #     pars['ha_n_amplitude'].value = 1000.0  
    #     for i in range(nGaussians):
    #         pars['ha_b_{}_amplitude'.format(i)].value = 1000.0          

    #     for i in range(nGaussians): 
    #         pars['ha_b_{}_center'.format(i)].value = (-1)**i * 100.0  
    #         pars['ha_b_{}_center'.format(i)].min = -2000.0  
    #         pars['ha_b_{}_center'.format(i)].max = 2000.0  

    #     pars['nii_6548_n_sigma'].value = ha_narrow_fwhm / 2.35 
    #     pars['nii_6584_n_sigma'].value = ha_narrow_fwhm / 2.35
    #     pars['sii_6717_n_sigma'].value = ha_narrow_fwhm / 2.35
    #     pars['sii_6731_n_sigma'].value = ha_narrow_fwhm / 2.35
    #     pars['ha_n_sigma'].value = ha_narrow_fwhm / 2.35 

    #     if ha_narrow_vary is False:

    #         pars['nii_6548_n_sigma'].vary = False
    #         pars['nii_6584_n_sigma'].vary = False
    #         pars['sii_6717_n_sigma'].vary = False
    #         pars['sii_6731_n_sigma'].vary = False
    #         pars['ha_n_sigma'].vary = False

    #     else:

    #         pars['ha_n_sigma'].max = 1200.0 / 2.35 
    #         pars['ha_n_sigma'].min = 400.0 / 2.35 
    #         pars['nii_6548_n_sigma'].set(expr='ha_n_sigma')
    #         pars['nii_6584_n_sigma'].set(expr='ha_n_sigma')
    #         pars['sii_6717_n_sigma'].set(expr='ha_n_sigma')
    #         pars['sii_6731_n_sigma'].set(expr='ha_n_sigma')
   
    #     for i in range(nGaussians): 
    #         pars['ha_b_{}_sigma'.format(i)].value = 1200.0

    #     for i in range(nGaussians): 
    #         pars['ha_b_{}_sigma'.format(i)].min = 1200.0 / 2.35

    #     pars['nii_6548_n_amplitude'].min = 0.0
    #     pars['nii_6584_n_amplitude'].min = 0.0
    #     pars['sii_6717_n_amplitude'].min = 0.0 
    #     pars['sii_6731_n_amplitude'].min = 0.0 
    #     pars['ha_n_amplitude'].min = 0.0 
    #     for i in range(nGaussians): 
    #         pars['ha_b_{}_amplitude'.format(i)].min = 0.0  
        
    #     pars['ha_n_center'].value = ha_narrow_voff 
    #     pars['sii_6731_n_center'].value = ha_narrow_voff + wave2doppler(6731*u.AA, w0).value 
    #     pars['sii_6717_n_center'].value = ha_narrow_voff + wave2doppler(6717*u.AA, w0).value 
    #     pars['nii_6548_n_center'].value = ha_narrow_voff + wave2doppler(6548*u.AA, w0).value 
    #     pars['nii_6584_n_center'].value  = ha_narrow_voff + wave2doppler(6584*u.AA, w0).value 

    #     if ha_narrow_vary is False:

    #         pars['ha_n_center'].vary = False 
    #         pars['sii_6731_n_center'].vary = False 
    #         pars['sii_6717_n_center'].vary = False 
    #         pars['nii_6548_n_center'].vary = False 
    #         pars['nii_6584_n_center'].vary = False  

    #     else: 

    #         pars['ha_n_center'].min = -500.0
    #         pars['ha_n_center'].max = 500.0
    
    #         pars['sii_6731_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6731*u.AA, w0).value))
    #         pars['sii_6717_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6717*u.AA, w0).value))
    #         pars['nii_6548_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6548*u.AA, w0).value))
    #         pars['nii_6584_n_center'].set(expr = 'ha_n_center+{}'.format(wave2doppler(6584*u.AA, w0).value))

    #     pars['nii_6548_n_amplitude'].set(expr='0.333*nii_6584_n_amplitude')

        
    # elif fit_model == 'Hb':

    #     """
    #     From Shen+15/11

    #     Need to get Boroson and Green (1992) optical iron template, 
    #     which people seem to convolve with a Gaussian
    
    #     Model each [OIII] line with Gaussian, one for the core and the other for the blue wing.
    #     Decide whether to fix flux ratio 3:1
    #     Velocity offset and FWHM of narrow Hb tied to the core [OIII] components
    #     Upper limit of 1200 km/s on the narrow line FWHM
    #     Broad Hb modelled by single gaussian, or up to 3 Gaussians each with FWHM > 1200 km/s

    #     It seams like a = b not the same as b = a, which is annoying. 
    #     Need to be careful

    #     """

    #     xscale = 1.0 

    #     mod = GaussianModel(prefix='oiii_4959_n_')

    #     mod += GaussianModel(prefix='oiii_5007_n_')

    #     mod += GaussianModel(prefix='oiii_4959_b_')

    #     mod += GaussianModel(prefix='oiii_5007_b_')

    #     if hb_narrow is True: 
    #         mod += GaussianModel(prefix='hb_n_')  

    #     for i in range(nGaussians):

    #         mod += GaussianModel(prefix='hb_b_{}_'.format(i))  

    #     pars = mod.make_params() 

    #     pars['oiii_4959_n_amplitude'].value = 1000.0
    #     pars['oiii_5007_n_amplitude'].value = 1000.0
    #     pars['oiii_4959_b_amplitude'].value = 1000.0 
    #     pars['oiii_5007_b_amplitude'].value = 1000.0 
    #     if hb_narrow is True: 
    #         pars['hb_n_amplitude'].value = 1000.0  
    #     for i in range(nGaussians):
    #         pars['hb_b_{}_amplitude'.format(i)].value = 1000.0  

    #     pars['oiii_4959_n_center'].value = wave2doppler(4960.295*u.AA, w0).value 
    #     pars['oiii_5007_n_center'].value = wave2doppler(5008.239*u.AA, w0).value 
    #     pars['oiii_4959_b_center'].value = wave2doppler(4960.295*u.AA, w0).value 
    #     pars['oiii_5007_b_center'].value = wave2doppler(5008.239*u.AA, w0).value 
    #     if hb_narrow is True: 
    #         pars['hb_n_center'].value = 0.0    
    #     for i in range(nGaussians): 
    #         pars['hb_b_{}_center'.format(i)].value = -100.0  
    #         pars['hb_b_{}_center'.format(i)].min = -2000.0  
    #         pars['hb_b_{}_center'.format(i)].max = 2000.0  

    #     pars['oiii_4959_n_sigma'].value = 800 / 2.35
    #     pars['oiii_5007_n_sigma'].value = 800 / 2.35
    #     pars['oiii_4959_b_sigma'].value = 1200.0 
    #     pars['oiii_5007_b_sigma'].value = 1200.0 
    #     if hb_narrow is True: 
    #         pars['hb_n_sigma'].value = 800 / 2.35
    #     for i in range(nGaussians): 
    #         pars['hb_b_{}_sigma'.format(i)].value = 1200.0

    #     pars['oiii_4959_n_amplitude'].min = 0.0
    #     pars['oiii_5007_n_amplitude'].min = 0.0
    #     pars['oiii_4959_b_amplitude'].min = 0.0 
    #     pars['oiii_5007_b_amplitude'].min = 0.0 
    #     if hb_narrow is True:      
    #         pars['hb_n_amplitude'].min = 0.0 
    #     for i in range(nGaussians): 
    #         pars['hb_b_{}_amplitude'.format(i)].min = 0.0  
        

    #     pars['oiii_5007_n_center'].min = wave2doppler(5008.239*u.AA, w0).value - 500.0
    #     pars['oiii_5007_n_center'].max = wave2doppler(5008.239*u.AA, w0).value + 500.0
    #     if hb_narrow is True: 
    #         pars['hb_n_center'].set(expr = 'oiii_5007_n_center-{}'.format(wave2doppler(5008.239*u.AA, w0).value)) 
    #     pars['oiii_4959_n_center'].set(expr = 'oiii_5007_n_center+{}'.format(wave2doppler(4960.295*u.AA, w0).value - wave2doppler(5008.239*u.AA, w0).value))

    #     pars['oiii_5007_n_sigma'].max = 900.0 / 2.35 
    #     pars['oiii_5007_n_sigma'].min = 400.0 / 2.35 
    #     pars['oiii_4959_n_sigma'].set(expr='oiii_5007_n_sigma')
    #     if hb_narrow is True: 
    #         pars['hb_n_sigma'].set(expr='oiii_5007_n_sigma')

    #     pars.add('oiii_5007_b_center_delta') 
    #     pars['oiii_5007_b_center_delta'].value = 500.0 
    #     pars['oiii_5007_b_center_delta'].min = -1500.0 
    #     pars['oiii_5007_b_center_delta'].max = 1500.0

    #     pars['oiii_5007_b_center'].set(expr='oiii_5007_n_center-oiii_5007_b_center_delta')

    #     pars.add('oiii_4959_b_center_delta') 
    #     pars['oiii_4959_b_center_delta'].value = 500.0 
    #     pars['oiii_4959_b_center_delta'].min = -1500.0 
    #     pars['oiii_4959_b_center_delta'].max = 1000.0

    #     pars['oiii_4959_b_center'].set(expr='oiii_4959_n_center-oiii_4959_b_center_delta')

    #     for i in range(nGaussians): 
    #         pars['hb_b_{}_sigma'.format(i)].min = 1000.0 / 2.35
    
    #     pars['oiii_5007_b_sigma'].min = 1200.0 / 2.35 
    #     pars['oiii_5007_b_sigma'].max = 6000.0 / 2.35
    #     pars['oiii_4959_b_sigma'].set(expr='oiii_5007_b_sigma')

    #     # Set amplitude of [oiii] broad component less than narrow component 
    #     pars.add('oiii_5007_b_amp_delta')
    #     pars['oiii_5007_b_amp_delta'].value = 0.1
    #     pars['oiii_5007_b_amp_delta'].min = 0.0 

    #     pars['oiii_5007_b_amplitude'].set(expr='((oiii_5007_n_amplitude/oiii_5007_n_sigma)-oiii_5007_b_amp_delta)*oiii_5007_b_sigma')

    #     # Set 3:1 [OIII] peak ratio  
    #     pars['oiii_4959_n_amplitude'].set(expr='0.3333*oiii_5007_n_amplitude')

    #     # Also set 3:1 [OIII] peak ratio for broad components 
    #     pars.add('oiii_4959_b_amp_delta')   
    #     pars['oiii_4959_b_amp_delta'].value = 0.1 
    #     pars['oiii_4959_b_amp_delta'].min = 0.0

    #     pars['oiii_4959_b_amplitude'].set(expr='oiii_4959_b_sigma*((oiii_5007_b_amplitude/oiii_5007_b_sigma)-oiii_4959_b_amp_delta)/3')  


    # elif fit_model == 'siiv':

    #     xscale = 1.0 

    #     mod = GaussianModel(prefix='g0_') + GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_') + GaussianModel(prefix='g3_') 
        
    #     pars = mod.make_params() 

    #     pars['g0_amplitude'].value = 1.0
    #     pars['g1_amplitude'].value = 1.0
    #     pars['g2_amplitude'].value = 1.0
    #     pars['g3_amplitude'].value = 1.0

    #     pars['g0_amplitude'].min = 0.0
    #     pars['g1_amplitude'].min = 0.0
        
    #     pars['g2_amplitude'].set(expr='g0_amplitude')
    #     pars['g3_amplitude'].set(expr='g1_amplitude')

    #     pars['g0_sigma'].value = 1.0
    #     pars['g1_sigma'].value = 1.0
    #     pars['g2_sigma'].value = 1.0
    #     pars['g3_sigma'].value = 1.0

    #     pars['g0_sigma'].min = 1000.0
    #     pars['g1_sigma'].min = 1000.0

    #     pars['g2_sigma'].set(expr='g0_sigma')
    #     pars['g3_sigma'].set(expr='g1_sigma')

    #     pars['g0_center'].value = 0.0
    #     pars['g1_center'].value = 0.0
    #     pars['g2_center'].value = 0.0
    #     pars['g3_center'].value = 0.0

    #     pars['g1_center'].set(expr='g0_center')        
    #     pars['g2_center'].set(expr = 'g0_center+{}'.format(wave2doppler(1402*u.AA, w0).value - wave2doppler(1397*u.AA, w0).value))
    #     pars['g3_center'].set(expr='g2_center')


    # if (n_samples > 1) & (plot is True):

    #     fig = plt.figure(figsize=(6,10))

    #     fit = fig.add_subplot(3,1,1)
    #     fit.set_xticklabels( () )
    #     residuals = fig.add_subplot(3,1,2)

    #     blue_cont = wave2doppler(continuum_region[0], w0)
    #     red_cont = wave2doppler(continuum_region[1], w0) 

    #     fit.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    #     fit.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

    #     residuals.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    #     residuals.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

    #     for item in ma.extras.flatnotmasked_contiguous(vdat_array_fit[0, :].flatten()):
    #         fit.axvspan(vdat_array_fit[0, item].min(), vdat_array_fit[0, item].max(), color='moccasin', alpha=0.4)
    #         residuals.axvspan(vdat_array_fit[0, item].min(), vdat_array_fit[0, item].max(), color='moccasin', alpha=0.4)

    #     fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    #     residuals.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=12)
    #     residuals.set_ylabel("Residual")

    #     if plot_region.unit == (u.km/u.s):
    #         plot_region = doppler2wave(plot_region, w0)

    #     plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])
    #     plotting_limits = wave2doppler(plot_region, w0) 
    #     fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)
    #     residuals.set_xlim(fit.get_xlim())    

    #     fit.set_ylim(ma.median(flux_array_fit[k, :]) - 2.0*ma.std(flux_array_fit[k, :]), ma.median(flux_array_fit[k, :]) + 4.0*ma.std(flux_array_fit[k, :]))
    #     residuals.set_ylim(-6, 6)

    # for k, (x, y, er) in enumerate(zip(vdat_array_fit, flux_array_fit, err_array_fit)):

    #     if fit_model == 'MultiGauss':

    #         for i in range(nGaussians):
                
    #             pars['g{}_center'.format(i)].value = 0.0
    #             pars['g{}_amplitude'.format(i)].value =  1.0 
    #             pars['g{}_sigma'.format(i)].value = 1200.0 
          
    #     elif fit_model == 'GaussHermite':

    #         for i in range(gh_order + 1):

    #             pars['amp{}'.format(i)].value = 1.0
    #             pars['sig{}'.format(i)].value = 1.0
    #             pars['cen{}'.format(i)].value = 0.0        
        
    #     elif fit_model == 'Ha':

    #         pars['nii_6548_n_amplitude'].value = 1000.0
    #         pars['nii_6584_n_amplitude'].value = 1000.0
    #         pars['sii_6717_n_amplitude'].value = 1000.0 
    #         pars['sii_6731_n_amplitude'].value = 1000.0 
    #         pars['ha_n_amplitude'].value = 1000.0  
           
    #         for i in range(nGaussians):
    #             pars['ha_b_{}_amplitude'.format(i)].value = 1000.0          
    
    #         for i in range(nGaussians): 
    #             pars['ha_b_{}_center'.format(i)].value = (-1)**i * 100.0  
           
    #         pars['nii_6548_n_sigma'].value = ha_narrow_fwhm / 2.35 
    #         pars['nii_6584_n_sigma'].value = ha_narrow_fwhm / 2.35
    #         pars['sii_6717_n_sigma'].value = ha_narrow_fwhm / 2.35
    #         pars['sii_6731_n_sigma'].value = ha_narrow_fwhm / 2.35
    #         pars['ha_n_sigma'].value = ha_narrow_fwhm / 2.35 
    
    #         for i in range(nGaussians): 
    #             pars['ha_b_{}_sigma'.format(i)].value = 1200.0
    
    #         pars['ha_n_center'].value = ha_narrow_voff 
    #         pars['sii_6731_n_center'].value = ha_narrow_voff + wave2doppler(6731*u.AA, w0).value 
    #         pars['sii_6717_n_center'].value = ha_narrow_voff + wave2doppler(6717*u.AA, w0).value 
    #         pars['nii_6548_n_center'].value = ha_narrow_voff + wave2doppler(6548*u.AA, w0).value 
    #         pars['nii_6584_n_center'].value  = ha_narrow_voff + wave2doppler(6584*u.AA, w0).value 
            
    #     elif fit_model == 'Hb':
    
    #         pars['oiii_4959_n_amplitude'].value = 1000.0
    #         pars['oiii_5007_n_amplitude'].value = 1000.0
    #         pars['oiii_4959_b_amplitude'].value = 1000.0 
    #         pars['oiii_5007_b_amplitude'].value = 1000.0 
    #         if hb_narrow is True: 
    #             pars['hb_n_amplitude'].value = 1000.0  
    #         for i in range(nGaussians):
    #             pars['hb_b_{}_amplitude'.format(i)].value = 1000.0  
    
    #         pars['oiii_4959_n_center'].value = wave2doppler(4960.295*u.AA, w0).value 
    #         pars['oiii_5007_n_center'].value = wave2doppler(5008.239*u.AA, w0).value 
    #         pars['oiii_4959_b_center'].value = wave2doppler(4960.295*u.AA, w0).value 
    #         pars['oiii_5007_b_center'].value = wave2doppler(5008.239*u.AA, w0).value 
    #         if hb_narrow is True: 
    #             pars['hb_n_center'].value = 0.0    
    #         for i in range(nGaussians): 
    #             pars['hb_b_{}_center'.format(i)].value = -100.0  
    
    #         pars['oiii_4959_n_sigma'].value = 800 / 2.35
    #         pars['oiii_5007_n_sigma'].value = 800 / 2.35
    #         pars['oiii_4959_b_sigma'].value = 1200.0 
    #         pars['oiii_5007_b_sigma'].value = 1200.0 
    #         if hb_narrow is True: 
    #             pars['hb_n_sigma'].value = 800 / 2.35
    #         for i in range(nGaussians): 
    #             pars['hb_b_{}_sigma'.format(i)].value = 1200.0
            
    #         pars['oiii_4959_b_center_delta'].value = 500.0 
    #         pars['oiii_5007_b_amp_delta'].value = 0.1
    #         pars['oiii_4959_b_amp_delta'].value = 0.1 

        
    #     elif fit_model == 'siiv':

    #         pars['g0_amplitude'].value = 1.0
    #         pars['g1_amplitude'].value = 1.0
            
    #         pars['g0_sigma'].value = 1.0
    #         pars['g1_sigma'].value = 1.0
            
    #         pars['g0_center'].value = 0.0

    #     out = minimize(resid,
    #                    pars,
    #                    kws={'x':np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value/xscale), 
    #                         'model':mod, 
    #                         'data':ma.getdata(y[~ma.getmaskarray(y)]),
    #                         'sigma':ma.getdata(er[~ma.getmaskarray(er)])},
    #                    method=fitting_method)

    #     if n_samples == 1: 
        
    #         if verbose:
            
    #             print out.message 
                
    #             if fit_model == 'GaussHermite':
        
    #                 for key, value in pars.valuesdict().items():
    #                     if 'cen' in key:
    #                         print key, value * xscale
    #                     else:
    #                         print key, value
        
    #             else:
        
    #                 print fit_report(pars)

    #     if (plot) & (n_samples > 1):

    #         xdat_plot = ma.getdata(wav_array_plot[k, ~ma.getmaskarray(wav_array_plot[k, :])]).value 
    #         vdat_plot = ma.getdata(vdat_array_plot[k, ~ma.getmaskarray(vdat_array_plot[k, :])]).value 
    #         ydat_plot = ma.getdata(flux_array_plot[k, ~ma.getmaskarray(flux_array_plot[k, :])]) 
    #         yerr_plot = ma.getdata(err_array_plot[k, ~ma.getmaskarray(err_array_plot[k, :])])

    #         if 'pts1' in locals():
    #             pts1.set_data((vdat_plot, ydat_plot))
    #         else:
    #             pts1, = fit.plot(vdat_plot, ydat_plot, markerfacecolor='grey', markeredgecolor='None', linestyle='', marker='o', markersize=2)

    #         if 'pts2' in locals():
    #             pts2.set_data((vdat_plot, (ydat_plot - resid(params=pars, x=vdat_plot/xscale, model=mod)) / yerr_plot))
    #         else:
    #             pts2, = residuals.plot(vdat_plot, (ydat_plot - resid(params=pars, x=vdat_plot/xscale, model=mod)) / yerr_plot, markerfacecolor='grey', markeredgecolor='None', linestyle='', marker='o', markersize=2)

    #         vs = np.linspace(vdat_plot.min(), vdat_plot.max(), 1000)

    #         if 'line' in locals():
    #             line.set_data((vs, resid(params=pars, x=vs/xscale, model=mod)))
    #         else:
    #             line, = fit.plot(vs, resid(params=pars, x=vs/xscale, model=mod), color='black')

    #         fig.set_tight_layout(True)


    #         plt.pause(0.1)


    #     if save_dir is not None:
            
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    
    #         if n_samples == 1: 
    
    #             param_file = os.path.join(save_dir, 'my_params.txt')
    #             parfile = open(param_file, 'w')
    #             pars.dump(parfile)
    #             parfile.close()
        
    #             wav_file = os.path.join(save_dir, 'wav.txt')
    #             parfile = open(wav_file, 'wb')
    #             pickle.dump(wav, parfile, -1)
    #             parfile.close()
        
        
    #             if subtract_fe is True:
    #                 flux_dump = flux - resid(params=bkgdpars[k], 
    #                                          x=wav.value, 
    #                                          model=bkgdmod,
    #                                          sp_fe=sp_fe)
                
    #             if subtract_fe is False:
    #                 flux_dump = flux - resid(params=bkgdpars[k], 
    #                                          x=wav.value, 
    #                                          model=bkgdmod)
        
    #             flx_file = os.path.join(save_dir, 'flx.txt')
    #             parfile = open(flx_file, 'wb')
    #             pickle.dump(flux_dump, parfile, -1)
    #             parfile.close()
        
    #             err_file = os.path.join(save_dir, 'err.txt')
    #             parfile = open(err_file, 'wb')
    #             pickle.dump(err, parfile, -1)
    #             parfile.close()
        
    #             sd_file = os.path.join(save_dir, 'sd.txt')
    #             parfile = open(sd_file, 'wb')
    #             pickle.dump(xscale, parfile, -1)
    #             parfile.close()
        
    #             fittxt = ''    
    #             fittxt += strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n \n'
    #             fittxt += plot_title + '\n \n'
    #             fittxt += r'Converged with chi-squared = ' + str(out.chisqr) + ', DOF = ' + str(out.nfree) + '\n \n'
    #             fittxt += fit_report(pars)
        
    #             with open(os.path.join(save_dir, 'fit.txt'), 'w') as f:
    #                 f.write(fittxt) 

    #     # Calculate stats 
    
    #     if fit_model == 'Hb':
    
    #         """
    #         Only use broad Hb components to calculate stats 
    #         """
    
    #         mod_broad_hb = ConstantModel()
    
    #         for i in range(nGaussians):
    #             mod_broad_hb += GaussianModel(prefix='hb_b_{}_'.format(i))  
    
    #         pars_broad_hb = mod_broad_hb.make_params()
           
    #         pars_broad_hb['c'].value = 0.0
           
    #         for key, value in pars.valuesdict().iteritems():
    #             if key.startswith('hb_b_'):
    #                 pars_broad_hb[key].value = value 
                    
    
    #         integrand = lambda x: mod_broad_hb.eval(params=pars_broad_hb, x=np.array(x))
    
    #     elif fit_model == 'Ha':
    
    #         """
    #         Only use broad Hb components to calculate stats 
    #         """
    
    #         mod_broad_ha = ConstantModel()
    
    #         for i in range(nGaussians):
    #             mod_broad_ha += GaussianModel(prefix='ha_b_{}_'.format(i))  
    
    #         pars_broad_ha = mod_broad_ha.make_params()
           
    #         pars_broad_ha['c'].value = 0.0
           
    #         for key, value in pars.valuesdict().iteritems():
    #             if key.startswith('ha_b_'):
    #                 pars_broad_ha[key].value = value 
                    
    
    #         integrand = lambda x: mod_broad_ha.eval(params=pars_broad_ha, x=np.array(x))    
    
    #     else:
    
    #         integrand = lambda x: mod.eval(params=pars, x=np.array(x))
    
    #     # Calculate FWHM of distribution
    #     if line_region.unit == u.AA:
    #         line_region = wave2doppler(line_region, w0)
    
    #     dv = 1.0 # i think if this was anything else might need to change intergration.
    #     xs = np.arange(line_region.value[0], line_region[1].value, dv) / xscale 
    
    #     norm = np.sum(integrand(xs) * dv)
    #     pdf = integrand(xs) / norm
    #     cdf = np.cumsum(pdf)
    #     cdf_r = np.cumsum(pdf[::-1])[::-1] # reverse cumsum

    #     # max of line
    #     peak_flux = np.max(integrand(xs)) / spec_norm
    
    #     func_center = xs[np.argmax(pdf)] 
       
    #     half_max = np.max(pdf) / 2.0
    
    #     i = 0
    #     while pdf[i] < half_max:
    #         i+=1
    
    #     root1 = xs[i]
    
    #     i = 0
    #     while pdf[-i] < half_max:
    #         i+=1
    
    #     root2 = xs[-i]
    
    #     md = xs[np.argmin( np.abs( cdf - 0.5))]
     
    #     m = np.sum(xs * pdf * dv)
    
    #     v = np.sum( (xs-m)**2 * pdf * dv )
    #     sd = np.sqrt(v)
    
    #     # Equivalent width
    #     flux_line = integrand(xs)
    #     xs_wav = doppler2wave(xs*xscale*(u.km/u.s), w0)
     
    #     if subtract_fe is True:
    #         flux_bkgd = resid(params=bkgdpars[k], 
    #                           x=xs_wav.value, 
    #                           model=bkgdmod,
    #                           sp_fe=sp_fe)
        
    #     if subtract_fe is False:

    #         bkgdmod = Model(PLModel, 
    #                         param_names=['amplitude','exponent'], 
    #                         independent_vars=['x']) 

    #         flux_bkgd = resid(params=bkgdpars[k], 
    #                           x=xs_wav.value, 
    #                           model=bkgdmod)
        
    
    
    #     f = (flux_line + flux_bkgd) / flux_bkgd
    #     eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
    #     eqw = np.nansum(eqw)
    
    #     # Broad luminosity
    #     lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)        
    #     broad_lum = np.sum(integrand(xs)[:-1] * np.diff(xs_wav.value)) * (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm 
    
    #     if fit_model == 'Ha':       
            
    #         # I know that the Gaussians are normalised, so the area is just the amplitude, so surely
    #         # I shouldn't have to do this function evaluation. 
    

    #         # this bit won't work unless nGaussians = 2 
    #         broad_fwhm = np.zeros(nGaussians)

    #         for i in range(nGaussians):   
    #             broad_fwhm[i] = np.array(pars['ha_b_{}_fwhm'.format(i)].value)

    #         broad_fwhm = np.sort(broad_fwhm)[::-1]

    #         if len(broad_fwhm) == 1:
    #             broad_fwhm = np.append(broad_fwhm, np.nan)

    #         if len(broad_fwhm) == 2: 
    #             broad_fwhm = np.append(broad_fwhm, np.nan)

    #         narrow_mod = GaussianModel()
    #         narrow_pars = narrow_mod.make_params()
    #         narrow_pars['amplitude'].value = pars['ha_n_amplitude'].value
    #         narrow_pars['sigma'].value = pars['ha_n_sigma'].value 
    #         narrow_pars['center'].value = pars['ha_n_center'].value 
    
    #         narrow_lum = np.sum(narrow_mod.eval(params=narrow_pars, x=xs)[:-1] * np.diff(xs_wav.value)) * \
    #                     (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
    #         narrow_fwhm = pars['ha_n_fwhm'].value 
    #         narrow_voff = pars['ha_n_center'].value 
    
    #         oiii_5007_eqw = np.nan 
    #         oiii_5007_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_n_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_b_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_fwhm = np.nan
    #         oiii_5007_n_fwhm = np.nan 
    #         oiii_5007_b_fwhm = np.nan 
    #         oiii_5007_b_voff = np.nan 
    
    #         #######################################################################
    #         """
    #         Nicely print out important fitting info
    #         """     
            
    #         if verbose: 

    #             if n_samples == 1: 
    #                 if pars['ha_n_sigma'].vary is True:
    #                     print 'Narrow FWHM = {0:.1f}, Initial = {1:.1f}, Vary = {2}, Min = {3:.1f}, Max = {4:.1f}'.format(pars['ha_n_sigma'].value * 2.35, 
    #                                                                                                                       pars['ha_n_sigma'].init_value * 2.35, 
    #                                                                                                                       pars['ha_n_sigma'].vary, 
    #                                                                                                                       pars['ha_n_sigma'].min * 2.35, 
    #                                                                                                                       pars['ha_n_sigma'].max * 2.35) 
    #                 else:
    #                     print 'Narrow FWHM = {0:.1f}, Vary = {1}'.format(pars['ha_n_sigma'].value * 2.35, 
    #                                                                      pars['ha_n_sigma'].vary) 
            
    #                 if pars['ha_n_center'].vary is True:
    #                     print 'Narrow Center = {0:.1f}, Initial = {1:.1f}, Vary = {2}, Min = {3:.1f}, Max = {4:.1f}'.format(pars['ha_n_center'].value, 
    #                                                                                                                         pars['ha_n_center'].init_value, 
    #                                                                                                                         pars['ha_n_center'].vary, 
    #                                                                                                                         pars['ha_n_center'].min, 
    #                                                                                                                         pars['ha_n_center'].max) 
    #                 else:
    #                     print 'Narrow Center = {0:.1f}, Vary = {1}'.format(pars['ha_n_center'].value, 
    #                                                                      pars['ha_n_center'].vary)     
                                                                     
                                                                     
                                                            
    
    
    
    #     elif fit_model == 'Hb':    

    #         # this bit won't work unless nGaussians = 2 
    #         broad_fwhm = np.zeros(nGaussians)

    #         for i in range(nGaussians):   
    #             broad_fwhm[i] = np.array(pars['hb_b_{}_fwhm'.format(i)].value)

    #         broad_fwhm = np.sort(broad_fwhm)[::-1]

    #         if len(broad_fwhm) == 1:
    #             broad_fwhm = np.append(broad_fwhm, np.nan)

    #         if len(broad_fwhm) == 2:
    #             broad_fwhm = np.append(broad_fwhm, np.nan)
     
    #         oiii_5007_b_mod = GaussianModel()
    #         oiii_5007_b_pars = oiii_5007_b_mod.make_params() 
    
    #         oiii_5007_b_pars['amplitude'].value = pars['oiii_5007_b_amplitude'].value
    #         oiii_5007_b_pars['sigma'].value = pars['oiii_5007_b_sigma'].value 
    #         oiii_5007_b_pars['center'].value = pars['oiii_5007_b_center'].value 
    
    #         oiii_5007_b_xs = np.arange(oiii_5007_b_pars['center'].value - 10000.0, 
    #                                    oiii_5007_b_pars['center'].value + 10000.0, 
    #                                    dv)
    
    #         oiii_5007_b_xs_wav = doppler2wave(oiii_5007_b_xs*(u.km/u.s), w0) 
    
    
    #         oiii_5007_b_lum = np.sum(oiii_5007_b_mod.eval(params=oiii_5007_b_pars, x=oiii_5007_b_xs)[:-1] * np.diff(oiii_5007_b_xs_wav.value)) * \
    #                     (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
    #         oiii_5007_b_fwhm = oiii_5007_b_pars['sigma'].value * 2.35 
    
    #         oiii_5007_b_voff = oiii_5007_b_pars['center'].value - wave2doppler(5008.239*u.AA, w0).value
    
    #         oiii_5007_n_mod = GaussianModel()
    #         oiii_5007_n_pars = oiii_5007_n_mod.make_params() 
    
    #         oiii_5007_n_pars['amplitude'].value = pars['oiii_5007_n_amplitude'].value
    #         oiii_5007_n_pars['sigma'].value = pars['oiii_5007_n_sigma'].value 
    #         oiii_5007_n_pars['center'].value = pars['oiii_5007_n_center'].value 
    
    #         oiii_5007_n_xs = np.arange(oiii_5007_n_pars['center'].value - 10000.0, 
    #                                    oiii_5007_n_pars['center'].value + 10000.0, 
    #                                    dv)
    
    #         oiii_5007_n_xs_wav = doppler2wave(oiii_5007_n_xs*(u.km/u.s), w0) 
    
    
    #         oiii_5007_n_lum = np.sum(oiii_5007_n_mod.eval(params=oiii_5007_n_pars, x=oiii_5007_n_xs)[:-1] * np.diff(oiii_5007_n_xs_wav.value)) * \
    #                     (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
    #         oiii_5007_n_fwhm = oiii_5007_n_pars['sigma'].value * 2.35 
               
    #         oiii_5007_mod = GaussianModel(prefix='oiii_5007_n_') + GaussianModel(prefix='oiii_5007_b_')
    #         oiii_5007_pars = oiii_5007_mod.make_params() 
    
    #         oiii_5007_pars['oiii_5007_n_amplitude'].value = pars['oiii_5007_n_amplitude'].value
    #         oiii_5007_pars['oiii_5007_n_sigma'].value = pars['oiii_5007_n_sigma'].value 
    #         oiii_5007_pars['oiii_5007_n_center'].value = pars['oiii_5007_n_center'].value 
    #         oiii_5007_pars['oiii_5007_b_amplitude'].value = pars['oiii_5007_b_amplitude'].value
    #         oiii_5007_pars['oiii_5007_b_sigma'].value = pars['oiii_5007_b_sigma'].value 
    #         oiii_5007_pars['oiii_5007_b_center'].value = pars['oiii_5007_b_center'].value 
    
    
    #         flux_line = oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs)
    
    
    #         if subtract_fe is True:
    #             flux_bkgd = resid(params=bkgdpars[k], 
    #                               x=oiii_5007_n_xs_wav.value, 
    #                               model=bkgdmod,
    #                               sp_fe=sp_fe)
        
    #         if subtract_fe is False:
    #             flux_bkgd = resid(params=bkgdpars[k], 
    #                               x=oiii_5007_n_xs_wav.value, 
    #                               model=bkgdmod)
    
    #         f = (flux_line + flux_bkgd) / flux_bkgd 
    #         oiii_5007_eqw = (f[:-1] - 1.0) * np.diff(oiii_5007_n_xs_wav.value)
    #         oiii_5007_eqw = np.nansum(oiii_5007_eqw)
    
    #         oiii_5007_lum = np.sum(oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs)[:-1] * np.diff(oiii_5007_n_xs_wav.value)) * \
    #                     (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm  
    
    #         oiii_5007_pdf = oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs)
    #         half_max = np.max(oiii_5007_pdf) / 2.0
        
    #         i = 0
    #         while oiii_5007_pdf[i] < half_max:
    #             i+=1
        
    #         oiii_5007_root1 = oiii_5007_n_xs[i]
        
    #         i = 0
    #         while oiii_5007_pdf[-i] < half_max:
    #             i+=1
        
    #         oiii_5007_root2 = oiii_5007_n_xs[-i]
            
    #         oiii_5007_fwhm = oiii_5007_root2 - oiii_5007_root1
    
    #         narrow_fwhm = pars['oiii_5007_n_sigma'].value * 2.35 
    #         narrow_voff = pars['oiii_5007_n_center'].value  - wave2doppler(5008.239*u.AA, w0).value  
             
    #         if hb_narrow is True:
    
    #             narrow_mod = GaussianModel()
    #             narrow_pars = narrow_mod.make_params()
    #             narrow_pars['amplitude'].value = pars['hb_n_amplitude'].value
    #             narrow_pars['sigma'].value = pars['hb_n_sigma'].value 
    #             narrow_pars['center'].value = pars['hb_n_center'].value 
    
    #             narrow_lum = np.sum(narrow_mod.eval(params=narrow_pars, x=xs)[:-1] * np.diff(xs_wav.value)) * \
    #                         (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm
    
            
    #             #######################################################################
    #             """
    #             Nicely print out important fitting info
    #             """     
                
    #             if verbose: 

    #                 if n_samples == 1:
    
    #                     if pars['oiii_5007_n_sigma'].vary is True:
        
        
    #                         print 'Narrow FWHM = {0:.1f}, Initial = {1:.1f}, Min = {2:.1f}, Max = {3:.1f}'.format(pars['hb_n_sigma'].value * 2.35, 
    #                                                                                                               pars['hb_n_sigma'].init_value * 2.35, 
    #                                                                                                               pars['hb_n_sigma'].min * 2.35, 
    #                                                                                                               pars['hb_n_sigma'].max * 2.35) 
    #                     else:
    #                         print 'Narrow FWHM = {0:.1f}, Vary = {1}'.format(pars['hb_n_sigma'].value * 2.35, 
    #                                                                          pars['oiii_5007_n_sigma'].vary) 
                
    #                     if pars['oiii_5007_n_center'].vary is True:
    #                         print 'Narrow Center = {0:.1f}, Initial = {1:.1f}, Min = {2:.1f}, Max = {3:.1f}'.format(pars['hb_n_center'].value, 
    #                                                                                                                             pars['hb_n_center'].init_value, 
    #                                                                                                                             pars['hb_n_center'].min, 
    #                                                                                                                             pars['hb_n_center'].max) 
    #                     else:
    #                         print 'Narrow Center = {0:.1f}, Vary = {1}'.format(pars['hb_n_center'].value, 
    #                                                                            pars['oiii_5007_n_center'].vary)     
    
    #         else:
    
    #             narrow_lum = np.nan * (u.erg / u.s)          
    
    #     elif fit_model == 'GaussHermite':
          
    #         narrow_lum = np.nan * (u.erg / u.s)
    #         narrow_fwhm = np.nan 
    #         narrow_voff = np.nan 
    #         oiii_5007_eqw = np.nan 
    #         oiii_5007_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_n_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_b_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_fwhm = np.nan 
    #         oiii_5007_n_fwhm = np.nan 
    #         oiii_5007_b_fwhm = np.nan 
    #         oiii_5007_b_voff = np.nan

    #         broad_fwhm = np.repeat(np.nan, 3)

    #     elif fit_model == 'MultiGauss':

    #         broad_fwhm = np.zeros(nGaussians)

    #         for i in range(nGaussians):   
    #             broad_fwhm[i] = np.array(pars['g{}_fwhm'.format(i)].value)

    #         broad_fwhm = np.sort(broad_fwhm)[::-1]

    #         if len(broad_fwhm) == 1:
    #             broad_fwhm = np.append(broad_fwhm, np.nan)

    #         if len(broad_fwhm) == 2:
    #             broad_fwhm = np.append(broad_fwhm, np.nan)

    #         narrow_lum = np.nan * (u.erg / u.s)
    #         narrow_fwhm = np.nan 
    #         narrow_voff = np.nan 
    #         oiii_5007_eqw = np.nan 
    #         oiii_5007_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_n_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_b_lum = np.nan * (u.erg / u.s)
    #         oiii_5007_fwhm = np.nan 
    #         oiii_5007_n_fwhm = np.nan 
    #         oiii_5007_b_fwhm = np.nan 
    #         oiii_5007_b_voff = np.nan

    #     fit_out = {'name':plot_title, 
    #                'fwhm':(root2 - root1)*xscale,
    #                'fwhm_1':broad_fwhm[0],
    #                'fwhm_2':broad_fwhm[1],
    #                'fwhm_3':broad_fwhm[2],
    #                'sigma': sd*xscale,
    #                'median': md*xscale,
    #                'cen': func_center*xscale,
    #                'eqw': eqw,
    #                'broad_lum':np.log10(broad_lum.value),
    #                'peak':peak_flux, 
    #                'narrow_fwhm':narrow_fwhm,
    #                'narrow_lum':np.log10(narrow_lum.value) if ~np.isnan(narrow_lum.value) else np.nan,
    #                'narrow_voff':narrow_voff, 
    #                'oiii_5007_eqw':oiii_5007_eqw,
    #                'oiii_5007_lum':np.log10(oiii_5007_lum.value) if ~np.isnan(oiii_5007_lum.value) else np.nan,
    #                'oiii_5007_n_lum':np.log10(oiii_5007_n_lum.value) if ~np.isnan(oiii_5007_n_lum.value) else np.nan,
    #                'oiii_5007_b_lum':np.log10(oiii_5007_b_lum.value) if ~np.isnan(oiii_5007_b_lum.value) else np.nan,
    #                'oiii_fwhm':oiii_5007_fwhm,
    #                'oiii_n_fwhm':oiii_5007_n_fwhm,
    #                'oiii_b_fwhm':oiii_5007_b_fwhm,
    #                'oiii_5007_b_voff':oiii_5007_b_voff,
    #                'redchi':out.redchi,
    #                'snr':snr[k],
    #                'dv':spec_dv, 
    #                'monolum':np.log10(mono_lum[k]),
    #                'fe_ew':eqw_fe[k]}
        
    #     if n_samples == 1: 

    #         if verbose:

    #             print 'Monochomatic luminosity at {0} = {1:.3f}'.format(mono_lum_wav, np.log10(mono_lum.value)) 
            
    
    #             print  fit_out['name'] + '\n'\
    #                   'Broad FWHM: {0:.2f} km/s \n'.format(fit_out['fwhm']), \
    #                   'Narrow Broad FWHM: {0:.2f} km/s \n'.format(fit_out['fwhm_1']), \
    #                   'Very Broad FWHM: {0:.2f} km/s \n'.format(fit_out['fwhm_2']), \
    #                   'Broad sigma: {0:.2f} km/s \n'.format(fit_out['sigma']), \
    #                   'Broad median: {0:.2f} km/s \n'.format(fit_out['median']), \
    #                   'Broad centroid: {0:.2f} km/s \n'.format(fit_out['cen']), \
    #                   'Broad EQW: {0:.2f} A \n'.format(fit_out['eqw']), \
    #                   'Broad luminosity {0:.2f} erg/s \n'.format(fit_out['broad_lum']), \
    #                   'Broad peak {0:.2e} erg/s \n'.format(fit_out['peak']), \
    #                   'Narrow FWHM: {0:.2f} km/s \n'.format(fit_out['narrow_fwhm']), \
    #                   'Narrow luminosity: {0:.2f} km/s \n'.format(fit_out['narrow_lum']), \
    #                   'Narrow velocity: {0:.2f} km/s \n'.format(fit_out['narrow_voff']), \
    #                   'OIII5007 EQW: {0:.2f} A \n'.format(fit_out['oiii_5007_eqw']),\
    #                   'OIII5007 luminosity {0:.2f} erg/s \n'.format(fit_out['oiii_5007_lum']),\
    #                   'OIII5007 narrow luminosity: {0:.2f} erg/s \n'.format(fit_out['oiii_5007_n_lum']),\
    #                   'OIII5007 broad luminosity: {0:.2f} erg/s \n'.format(fit_out['oiii_5007_b_lum']),\
    #                   'OIII5007 FWHM: {0:.2f} km/s \n'.format(fit_out['oiii_fwhm']),\
    #                   'OIII5007 narrow FWHM: {0:.2f} km/s \n'.format(fit_out['oiii_n_fwhm']),\
    #                   'OIII5007 broad FWHM: {0:.2f} km/s \n'.format(fit_out['oiii_b_fwhm']),\
    #                   'OIII5007 broad velocity: {0:.2f} km/s \n'.format(fit_out['oiii_5007_b_voff']),\
    #                   'Reduced chi-squared: {0:.2f} \n'.format(fit_out['redchi']),\
    #                   'S/N: {0:.2f} \n'.format(fit_out['snr']), \
    #                   'dv: {0:.1f} km/s \n'.format(fit_out['dv'].value), \
    #                   'Monochomatic luminosity: {0:.2f} erg/s \n'.format(fit_out['monolum'])
        
    #             if emission_line == 'Hb':

    #                 print fit_out['name'] + ',',\
    #                 format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
    #                 format(fit_out['fwhm_1'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_1']) else ',',\
    #                 format(fit_out['fwhm_2'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_2']) else ',',\
    #                 format(fit_out['fwhm_3'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_3']) else ',',\
    #                 format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
    #                 format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
    #                 format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
    #                 format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
    #                 format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
    #                 format(fit_out['narrow_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['narrow_fwhm']) else ',',\
    #                 format(fit_out['narrow_lum'], '.2f') + ',' if ~np.isnan(fit_out['narrow_lum']) else ',',\
    #                 format(fit_out['narrow_voff'], '.2f') + ',' if ~np.isnan(fit_out['narrow_voff']) else ',',\
    #                 format(fit_out['oiii_5007_eqw'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_eqw']) else ',',\
    #                 format(fit_out['oiii_5007_lum'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_lum']) else ',',\
    #                 format(fit_out['oiii_5007_n_lum'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_n_lum']) else ',',\
    #                 format(fit_out['oiii_5007_b_lum'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_b_lum']) else ',',\
    #                 format(fit_out['oiii_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['oiii_fwhm']) else ',',\
    #                 format(fit_out['oiii_n_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['oiii_n_fwhm']) else ',',\
    #                 format(fit_out['oiii_b_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['oiii_b_fwhm']) else ',',\
    #                 format(fit_out['oiii_5007_b_voff'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_b_voff']) else ',',\
    #                 format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''


    #                 # print  fit_out['name'] + ','\
    #                 #       '{0:.2f},'.format(fit_out['fwhm']), \
    #                 #       '{0:.2f},'.format(fit_out['sigma']), \
    #                 #       '{0:.2f},'.format(fit_out['median']), \
    #                 #       '{0:.2f},'.format(fit_out['cen']), \
    #                 #       '{0:.2f},'.format(fit_out['eqw']), \
    #                 #       '{0:.2f},'.format(fit_out['broad_lum']), \
    #                 #       '{0:.2f},'.format(fit_out['narrow_fwhm']), \
    #                 #       '{0:.2f},'.format(fit_out['narrow_lum']), \
    #                 #       '{0:.2f},'.format(fit_out['narrow_voff']), \
    #                 #       '{0:.2f},'.format(fit_out['oiii_5007_eqw']),\
    #                 #       '{0:.2f},'.format(fit_out['oiii_5007_lum']),\
    #                 #       '{0:.2f},'.format(fit_out['oiii_5007_n_lum']),\
    #                 #       '{0:.2f},'.format(fit_out['oiii_5007_b_lum']),\
    #                 #       '{0:.2f},'.format(fit_out['oiii_fwhm']),\
    #                 #       '{0:.2f},'.format(fit_out['oiii_n_fwhm']),\
    #                 #       '{0:.2f},'.format(fit_out['oiii_b_fwhm']),\
    #                 #       '{0:.2f},'.format(fit_out['oiii_5007_b_voff']),\
    #                 #       '{0:.2f}'.format(fit_out['redchi'])          
        
    #             elif emission_line == 'Ha':

    #                 print fit_out['name'] + ',',\
    #                 format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
    #                 format(fit_out['fwhm_1'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_1']) else ',',\
    #                 format(fit_out['fwhm_2'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_2']) else ',',\
    #                 format(fit_out['fwhm_3'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_3']) else ',',\
    #                 format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
    #                 format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
    #                 format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
    #                 format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
    #                 format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
    #                 format(fit_out['narrow_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['narrow_fwhm']) else ',',\
    #                 format(fit_out['narrow_lum'], '.2f') + ',' if ~np.isnan(fit_out['narrow_lum']) else ',',\
    #                 format(fit_out['narrow_voff'], '.2f') + ',' if ~np.isnan(fit_out['narrow_voff']) else ',',\
    #                 format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''

    #             else: 

    #                 print fit_out['name'] + ',',\
    #                 format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
    #                 format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
    #                 format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
    #                 format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
    #                 format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
    #                 format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
    #                 format(fit_out['peak'], '.2e') + ',' if ~np.isnan(fit_out['peak']) else ',',\
    #                 format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''

        
        
    #         if save_dir is not None:
        
    #             with open(os.path.join(save_dir, 'fit.txt'), 'a') as f:

    #                 f.write('\n \n')
    #                 f.write('Peak: {0:.2f} km/s \n'.format(func_center))
    #                 f.write('FWHM: {0:.2f} km/s \n'.format(root2 - root1))
    #                 f.write('Median: {0:.2f} km/s \n'.format(md))
    #                 f.write('Sigma: {0:.2f} km/s \n'.format(sd))
    #                 f.write('Reduced chi-squared: {0:.2f} \n'.format(out.redchi))
    #                 f.write('EQW: {0:.2f} A \n'.format(eqw))      
        
    #             my_line_props = line_props(plot_title, func_center, root2 - root1, md, sd, out.redchi, eqw)  
    #             param_file = os.path.join(save_dir, 'line_props.txt')
    #             parfile = open(param_file, 'w')
    #             pickle.dump(my_line_props, parfile, -1)
    #             parfile.close()
              
        
        
    #         if plot is True:
        
    #             if subtract_fe is True:
    #                 flux_plot = flux - resid(params=bkgdpars, 
    #                                          x=wav.value, 
    #                                          model=bkgdmod,
    #                                          sp_fe=sp_fe)
                
    #             if subtract_fe is False:
    #                 flux_plot = flux - resid(params=bkgdpars, 
    #                                          x=wav.value, 
    #                                          model=bkgdmod)
        
    #             plot_fit(wav=wav,
    #                      flux = flux_plot,
    #                      err=err,
    #                      pars=pars,
    #                      mod=mod,
    #                      out=out,
    #                      plot_savefig = plot_savefig,
    #                      save_dir = save_dir,
    #                      maskout = maskout,
    #                      z=z,
    #                      w0=w0,
    #                      continuum_region=continuum_region,
    #                      fitting_region=fitting_region,
    #                      plot_region=plot_region,
    #                      plot_title=plot_title,
    #                      line_region=line_region,
    #                      mask_negflux = mask_negflux,
    #                      fit_model=fit_model,
    #                      hb_narrow=hb_narrow,
    #                      verbose=verbose,
    #                      reject_outliers=reject_outliers,  
    #                      reject_width=reject_width,
    #                      reject_sigma=reject_sigma,
    #                      gh_order = gh_order,
    #                      xscale=xscale)
        
    #             plot_fit_d3(wav=wav,
    #                         flux = flux_plot,
    #                         err=err,
    #                         pars=pars,
    #                         mod=mod,
    #                         out=out,
    #                         plot_savefig = plot_savefig,
    #                         save_dir = save_dir,
    #                         maskout = maskout,
    #                         z=z,
    #                         w0=w0,
    #                         continuum_region=continuum_region,
    #                         fitting_region=fitting_region,
    #                         plot_region=plot_region,
    #                         plot_title=plot_title,
    #                         line_region=line_region,
    #                         mask_negflux = mask_negflux,
    #                         fit_model=fit_model,
    #                         hb_narrow=hb_narrow,
    #                         verbose=verbose,
    #                         xscale=xscale,
    #                         gh_order=gh_order)
    #         return fit_out 

    #     else:

    #         print plot_title, k, fit_out['fwhm']

    #         if save_dir is not None:

    #             for col in df_out:

    #                 df_out.loc[k, col] = fit_out[col]


    return None 

  

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
