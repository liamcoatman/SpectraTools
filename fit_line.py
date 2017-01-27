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
from lmfit.models import GaussianModel, LorentzianModel, PowerLawModel, ConstantModel, LinearModel
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
import sys 
from barak import spec 
from astropy import constants as const
import mpld3
from scipy.signal import medfilt2d, medfilt
from functools import partial 
import pandas as pd
from multiprocessing import Pool 
from copy import deepcopy
import warnings
import matplotlib
import palettable 
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


def rebin_simple(wa, fl, er, n):
    
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

def find_wa_edges(wa):

    """ Given wavelength bin centres, find the edges of wavelengh
    bins.

    Examples
    --------

    >>> print find_wa_edges([1, 2.1, 3.3, 4.6])
    [ 0.45  1.55  2.7   3.95  5.25]
    
    """
    wa = np.asarray(wa)
    edges = wa[:-1] + 0.5 * (wa[1:] - wa[:-1])
    edges = [2*wa[0] - edges[0]] + edges.tolist() + [2*wa[-1] - edges[-1]]

    return np.array(edges)


def rebin(wa0, fl0, er0, wa1, weighted=False):

    """ Rebins spectrum to a new wavelength scale generated using the
    keyword parameters.

    Returns the rebinned spectrum.

    Will probably get the flux and errors for the first and last pixel
    of the rebinned spectrum wrong.

    General pointers about rebinning if you care about errors in the
    rebinned values:

    1. Don't rebin to a smaller bin size.
    2. Be aware when you rebin you introduce correlations between
       neighbouring points and between their errors.
    3. Rebin as few times as possible.

    """

    # Note: 0 suffix indicates the old spectrum, 1 the rebinned spectrum.

    fl1 = np.zeros(len(wa1))
    er1 = np.zeros(len(wa1))
 
    # find pixel edges, used when rebinning
    edges0 = find_wa_edges(wa0)
    edges1 = find_wa_edges(wa1)

    widths0 = edges0[1:] - edges0[:-1]

    npts0 = len(wa0)
    npts1 = len(wa1)

    df = 0.
    de2 = 0.
    npix = 0    # number of old pixels contributing to rebinned pixel,
    
    j = 0                # index of rebinned array
    i = 0                # index of old array

    # sanity check
    if edges0[-1] < edges1[0] or edges1[-1] < edges0[0]:
        raise ValueError('Wavelength scales do not overlap!')
    
    # find the first contributing old pixel to the rebinned spectrum
    if edges0[i+1] < edges1[0]:
        # Old wa scale extends lower than the rebinned scale. Find the
        # first old pixel that overlaps with rebinned scale.
        while edges0[i+1] < edges1[0]:
            i += 1
        i -= 1
    elif edges0[0] > edges1[j+1]:
        # New rebinned wa scale extends lower than the old scale. Find
        # the first rebinned pixel that overlaps with the old spectrum
        while edges0[0] > edges1[j+1]:
            fl1[j] = np.nan
            er1[j] = np.nan
            j += 1
        j -= 1

    lo0 = edges0[i]      # low edge of contr. (sub-)pixel in old scale
    
    while True:
    
        hi0 = edges0[i+1]  # upper edge of contr. (sub-)pixel in old scale
        hi1 = edges1[j+1]  # upper edge of jth pixel in rebinned scale

        if hi0 < hi1:

            if er0[i] > 0:

                dpix = (hi0 - lo0) / widths0[i]
                
                if weighted:

                    # https://en.wikipedia.org/wiki/Inverse-variance_weighting

                    df += (fl0[i] / er0[i]**2) * dpix
                    de2 += dpix / er0[i]**2
                    npix += dpix / er0[i]**2
                
                else:

                    df += fl0[i] * dpix
                    de2 += er0[i]**2 * dpix
                    npix += dpix 

      
            lo0 = hi0
            i += 1
          
            if i == npts0:  break
        
        else:

            # We have all old pixel flux values that contribute to the
            # new pixel; append the new flux value and move to the
            # next new pixel.
            
            if er0[i] > 0:

                dpix = (hi1 - lo0) / widths0[i]

                if weighted:
           
                    df += (fl0[i] / er0[i]**2) * dpix
                    de2 += dpix / er0[i]**2
                    npix += dpix / er0[i]**2
    
                else:

                    df += fl0[i] * dpix
                    de2 += er0[i]**2 * dpix
                    npix += dpix 


            if npix > 0:
                
                # find total flux and error, then divide by number of
                # pixels (i.e. conserve flux density).
                
                fl1[j] = df / npix
               
                if weighted:

                    # Not 100% sure this is correct 
                    er1[j] = np.sqrt(1.0 / npix) 

                else:
                    # sum in quadrature and then divide by npix
                    # simply following the rules of propagation
                    # of uncertainty             
                    er1[j] = np.sqrt(de2) / npix  
            
            else:

                fl1[j] = np.nan
                er1[j] = np.nan
            
            df = 0.
            de2 = 0.
            npix = 0.
            lo0 = hi1
            j += 1
            
            if j == npts1:  break
        
    return wa1, fl1, er1  

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

# Simple mouse click function to store coordinates
def onclick3(event):

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
                o += '[{0:.1f}, {1:.1f}]'.format(coords[2*i], coords[2*i+1])  
            else:
                o += '[{0:.1f}, {1:.1f}],'.format(coords[2*i], coords[2*i+1])  
        o += ']*u.AA'
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

def oiii_reconstruction(best_weights, ax=None):

    """
    Reconstruct just OIII emission from ICA components
    """

    cs = palettable.colorbrewer.diverging.RdBu_7.mpl_colors 
    

    comps_wav, comps, w = make_model_mfica(mfica_n_weights=10)
    
    w['w1'].value = 0.0 
    w['w2'].value = 0.0
    w['w3'].value = 0.0
    w['w4'].value = best_weights['w4']
    w['w5'].value = best_weights['w5']
    w['w6'].value = best_weights['w6']
    w['w7'].value = best_weights['w7']
    w['w8'].value = best_weights['w8']
    w['w9'].value = best_weights['w9']
    w['w10'].value = best_weights['w10']
    w['shift'].value = best_weights['shift']

    flux = mfica_model(w, comps, comps_wav, comps_wav)

    """
    We use the blue wing of the 4960 peak to reconstruct the 
    5008 peak
    """

    peak_diff = 5008.239 - 4960.295

    # just made up these boundaries
    inds1 = (comps_wav > 4900.0) & (comps_wav < 4980 - peak_diff)
    inds2 = (comps_wav > 4980.0) & (comps_wav < 5050.0)

    wav_5008 = np.concatenate((comps_wav[inds1] + peak_diff, comps_wav[inds2]))
    flux_5008 = np.concatenate((flux[inds1], flux[inds2]))

    """
    Fit linear model and subtract background
    """

    xfit = np.concatenate((wav_5008[:10], wav_5008[-10:]))
    yfit = np.concatenate((flux_5008[:10], flux_5008[-10:]))

    mod = LinearModel()
    out = mod.fit(yfit, x=xfit, slope=0.0, intercept=0.0)

    flux_5008 = flux_5008 - mod.eval(params=out.params, x=wav_5008)

    """
    If flux is negative set to zero
    """

    flux_5008[flux_5008 < 0.0] = 0.0 


    xs = np.arange(wav_5008.min(), wav_5008.max(), 0.01)
    vs = wave2doppler(xs*u.AA, 5008.239*u.AA)

    f = interp1d(wav_5008, flux_5008)

    cdf = np.cumsum(f(xs) / np.sum(f(xs))) 

    if ax is not None: 

        ax.plot(comps_wav, flux, color='grey') 
        ax.axhline(0.0, color='black', linestyle='--')
        ax.set_xlim(4900, 5100)
    
        ax.plot(wav_5008, flux_5008, color='black', lw=2)
         
        ax.axvline(xs[np.argmin(np.abs(cdf - 0.05))], color=cs[0], linestyle='--') 
        ax.axvline(xs[np.argmin(np.abs(cdf - 0.10))], color=cs[1], linestyle='--') 
        ax.axvline(xs[np.argmin(np.abs(cdf - 0.25))], color=cs[2], linestyle='--') 
        ax.axvline(xs[np.argmin(np.abs(cdf - 0.50))], color='grey', linestyle='--') 
        ax.axvline(xs[np.argmin(np.abs(cdf - 0.75))], color=cs[4], linestyle='--') 
        ax.axvline(xs[np.argmin(np.abs(cdf - 0.90))], color=cs[5], linestyle='--') 
        ax.axvline(xs[np.argmin(np.abs(cdf - 0.95))], color=cs[6], linestyle='--') 

        return None 

    else:

        return {'mfica_oiii_v05': vs[np.argmin(np.abs(cdf - 0.05))].value,
                'mfica_oiii_v10': vs[np.argmin(np.abs(cdf - 0.1))].value,
                'mfica_oiii_v25': vs[np.argmin(np.abs(cdf - 0.25))].value,
                'mfica_oiii_v50': vs[np.argmin(np.abs(cdf - 0.50))].value,
                'mfica_oiii_v75': vs[np.argmin(np.abs(cdf - 0.75))].value,
                'mfica_oiii_v90': vs[np.argmin(np.abs(cdf - 0.90))].value,
                'mfica_oiii_v95': vs[np.argmin(np.abs(cdf - 0.95))].value}

          
 

def plot_mfica_fit(mfica_n_weights=10,
                   xi=None,
                   yi=None,
                   dyi=None,
                   out=None,
                   comps=None,
                   comps_wav=None,
                   plot_savefig=None,
                   verbose=False,
                   save_dir=None):
    


    fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    
    plt.subplots_adjust(hspace=0.0) 
    
    # color = matplotlib.cm.get_cmap('Set1')
    # color = color(np.linspace(0, 1, mfica_n_weights))
    set1 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 
    
    xi_noreject = np.asarray(ma.getdata(xi[~ma.getmaskarray(xi)]).value)
    yi_noreject = ma.getdata(yi[~ma.getmaskarray(yi)])
    dyi_noreject = ma.getdata(dyi[~ma.getmaskarray(dyi)])

    xs = np.arange(xi_noreject.min(), xi_noreject.max(), 1)

    params = Parameters() 

    params.add('w1', value=out['w1']) 
    params.add('w2', value=out['w2'])
    params.add('w3', value=out['w3'])
    params.add('w4', value=out['w4'])
    params.add('w5', value=out['w5'])
    params.add('w6', value=out['w6'])
    params.add('w7', value=out['w7'])
    params.add('w8', value=out['w8'])
    params.add('w9', value=out['w9'])
    params.add('w10', value=out['w10'])
    params.add('shift', value=out['shift'])


    axs[0].plot(xs, 
                mfica_model(params, 
                            comps, 
                            comps_wav, 
                            xs), 
                color=set1[0], 
                zorder=3, 
                lw=2)

    axs[0].errorbar(xi_noreject, 
                    yi_noreject, 
                    yerr=dyi_noreject, 
                    linestyle='', 
                    color=set1[8], 
                    alpha=0.2, 
                    zorder=1)

    axs[0].plot(xi_noreject, 
                yi_noreject, 
                marker='o', 
                linestyle='', 
                markerfacecolor='None', 
                markeredgecolor='black', 
                alpha=0.4, 
                markersize=1, 
                zorder=2)
    
    colors = [4, 1, 1, 2, 2, 2, 8, 8, 8, 8]

    for i in range(1, mfica_n_weights+1):

        axs[0].plot(xs, 
                    mfica_get_comp(i, 
                                   params, 
                                   comps, 
                                   comps_wav, 
                                   xs), 
                    c=set1[colors[i-1]])
        

    axs[0].axvline(5008.239, linestyle='--', c='red', zorder=0)
    axs[0].axvline(4960.295, linestyle='--', c='red', zorder=0)
    
    axs[0].set_ylim(-0.1 * np.max(mfica_model(params, comps, comps_wav, xi_noreject)), 
                    1.1 * np.max(mfica_model(params, comps, comps_wav, xi_noreject)))

    axs[0].set_xlim(xi_noreject.min(), xi_noreject.max())

    # plot rejected points   

    xi_reject = np.array(xi[ma.getmaskarray(xi)])
    yi_reject = np.array(yi[ma.getmaskarray(yi)])
    dyi_reject = np.array(dyi[ma.getmaskarray(dyi)])


    axs[0].errorbar(xi_reject, 
                    yi_reject, 
                    yerr=dyi_reject, 
                    linestyle='', 
                    color=set1[8], 
                    alpha=0.2, 
                    zorder=1)

    axs[0].plot(xi_reject, 
                yi_reject, 
                marker='o', 
                linestyle='', 
                markerfacecolor='None', 
                markeredgecolor=set1[0], 
                alpha=0.4, 
                markersize=1, 
                zorder=2)
    #---------------------------------------------------------------------------


    oiii_reconstruction(out, ax=axs[1])
    

    #---------------------------------------------------------------------------

    axs[2].plot(xi_noreject, 
                (yi_noreject - mfica_model(params, comps, comps_wav, xi_noreject)) / dyi_noreject, 
                linestyle='', 
                marker='o', 
                markersize=3,
                markerfacecolor=set1[8],
                markeredgecolor='None')
    
    axs[2].set_ylim(-5, 5)
    
    axs[2].axhline(0.0, color='black', linestyle='--')

    axs[0].set_yticks([])
    axs[2].set_yticks([])
    
    fig.tight_layout()

    if plot_savefig is not None:
        fig.savefig(os.path.join(save_dir, plot_savefig))
    

    if verbose: 
        global coords
        coords = [] 
        cid = fig.canvas.mpl_connect('button_press_event', onclick3)

    if verbose:
        plt.show(1)
    
    plt.close()

    return None 

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
              verbose=True,
              emission_line='MFICA'):    

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(wav.value, 
               flux, 
               edgecolor='None', 
               s=15, 
               alpha=0.9, 
               facecolor='black')

    if emission_line != 'MFICA':

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

    fit.scatter(vdat.value, ydat, edgecolor='None', s=5, facecolor='black')
    fit.errorbar(vdat.value, 
                 ydat, 
                 yerr=yerr, 
                 linestyle='', 
                 alpha=0.2, 
                 color='grey')




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

    # if not all masked (no continuum region in plot region)
    if not mask.all(): 

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
    

    if (fit_model == 'Hb') | (fit_model == 'OIII'): 

        fit.axvline(0, linestyle='--', c='red')
        fit.axvline(wave2doppler(5008.239*u.AA, w0).value, linestyle='--', c='red')
        fit.axvline(wave2doppler(4960.295*u.AA, w0).value, linestyle='--', c='red')
 
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



        #--------------------------------------------------------
        
        mod_ha = GaussianModel(prefix='ha_n_')

        for i in range(nGaussians):
            mod_ha += GaussianModel(prefix='ha_b_{}_'.format(i))  

        pars_ha = mod_ha.make_params()

        pars_ha['ha_n_center'].value = pars['ha_n_center'].value
        pars_ha['ha_n_sigma'].value = pars['ha_n_sigma'].value
        pars_ha['ha_n_amplitude'].value = pars['ha_n_amplitude'].value   

        for key, value in pars.valuesdict().iteritems():
            if key.startswith('ha_b_'):
                pars_ha[key].value = value   

        fit.plot(np.sort(vdat.value), 
                 mod_ha.eval(pars_ha, x=np.sort(vdat.value)),
                 c='blue',
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
                  flux_array_red):

  
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

    global coords
    coords = [] 
    cid = fig.canvas.mpl_connect('button_press_event', lambda x: onclick2(x, w0=w0))

    plt.show(1)

    plt.close() 

    return None 

def mfica_model(weights, comps, comps_wav, x):

    fl = weights['w1'].value * comps[:, 0]

    for i in range(comps.shape[1] - 1):
        fl += weights['w{}'.format(i+2)].value * comps[:, i+1]

    # because otherwise tiny shifts don't do anything to the chi-squared
    shift = weights['shift'].value * 10.0 

    f = interp1d(comps_wav + shift, 
                 fl, 
                 bounds_error=False, 
                 fill_value=0.0)


    return f(x)

def mfica_get_comp(i, weights, comps, comps_wav, x):

    f = interp1d(comps_wav + weights['shift'].value, 
                 weights['w{}'.format(i)].value * comps[:, i-1], 
                 bounds_error=False, 
                 fill_value=np.nan)

    return f(x)

  
def mfica_resid(weights=None, 
                comps=None,
                comps_wav=None,
                x=None, 
                data=None, 
                sigma=None):


    if data is not None:

        resids = mfica_model(weights, comps, comps_wav, x) - data 

        if sigma is not None:

            weighted = np.sqrt(resids ** 2 / sigma ** 2)     
            
            return weighted

        else:

            return resids
    
    else:

        return model(weights, comps)   

def fit1(obj,
         n_samples, 
         plot_title, 
         verbose, 
         mono_lum_wav, 
         spec_norm, 
         z,
         save_dir):


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
            print colored(out.message, 'red'), colored('Number of function evaluations: {}'.format(out.nfev), 'red') 
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
        print plot_title, k, out.nfev

    if save_dir is not None:

        """
        Pickle background+continuum model (not in real flux units) 
        """

        # remove expressions from Parameters instance - https://groups.google.com/forum/#!topic/lmfit-py/6tCcTNe307I
        params_dump = deepcopy(out.params)
        for v in params_dump:
            params_dump[v].expr = None
            
        param_file = os.path.join(save_dir, 'my_params_bkgd.txt')
        parfile = open(param_file, 'w')
        params_dump.dump(parfile)
        parfile.close()


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
         plot,
         save_dir,
         show_continuum_fit,
         pseudo_continuum_fit): 

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
                       options={'maxiter':1e4, 'maxfev':1e4})

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


        if pseudo_continuum_fit & (n_samples == 1): 

            """ 
            Temporary function to write out continuum (but not fe subtracted) spectrum 
            """

            mask_temp = (wav.value > 4000.0) & (wav.value < 5500.0) 
            wav_array_cont_sub = wav.value[mask_temp]

            fl_cont = PseudoContinuum(wav_array_cont_sub, 
                                      out.params['amplitude'].value,
                                      out.params['exponent'].value,
                                      0.0,
                                      out.params['fe_sd'].value,
                                      out.params['fe_shift'].value,
                                      sp_fe)

            fl_fe = PseudoContinuum(wav_array_cont_sub, 
                                    0.0,
                                    0.0,
                                    out.params['fe_norm'].value,
                                    out.params['fe_sd'].value,
                                    out.params['fe_shift'].value,
                                    sp_fe)
    

            flux_array_cont_sub = flux[mask_temp] - fl_cont   
            flux_array_sub = flux[mask_temp] - fl_cont - fl_fe         
          

            # with open(os.path.join(save_dir, 'spec_cont_subtracted.txt'), 'w') as f:
            #     for ww, ff in zip(wav_array_cont_sub, flux_array_cont_sub):
            #         f.write('{} {} \n'.format(ww, ff))

            
            fig, ax = plt.subplots()
            
            ax.plot(wav_array_cont_sub, flux_array_cont_sub, color='grey', alpha=0.4)
            # ax.plot(wav_array_cont_sub, flux_array_sub, color='black', alpha=0.4)

            ax.plot(wav_array_cont_sub, fl_fe, color='red')

            ax.axhline(0, color='black', linestyle='--')

            # ax.set_xlim(4656, 5400)
            # ax.set_ylim(0, 0.7)

            # fig.savefig('/home/lc585/OIIIPaper/fe_example.png')

            plt.show() 

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
        print plot_title, k, out.nfev 


    ######################################################################################################################

    if n_samples == 1:  

        if verbose:

            print colored(out.message, 'red'), colored('Number of function evaluations: {}'.format(out.nfev), 'red') 
            print fit_report(out.params)
    
            if subtract_fe is True:
                print 'Fe FWHM = {0:.1f} km/s (initial = {1:.1f} km/s)'.format(out.params['fe_sd'].value * 2.35 * sp_fe.dv, out.params['fe_sd'].init_value * 2.35 * sp_fe.dv)      
                print 'Fe template shift = {0:.1f} km/s'.format(wave2doppler((4862.0 + out.params['fe_shift'].value)*u.AA, 4862.0*u.AA).value)      
            
        if verbose & show_continuum_fit:

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
                          flux_array_red)


        if save_dir is not None:

            """
            Pickle background+continuum model (not in real flux units) 
            """

            # remove expressions from Parameters instance - https://groups.google.com/forum/#!topic/lmfit-py/6tCcTNe307I
            params_dump = deepcopy(out.params)
            for v in params_dump:
                params_dump[v].expr = None
                
            param_file = os.path.join(save_dir, 'my_params_bkgd.txt')
            parfile = open(param_file, 'w')
            params_dump.dump(parfile)
            parfile.close()





    return (flux_array_fit_k, flux_array_plot_k, mono_lum, eqw_fe, out.params)

def fit4(obj, 
         n_samples, 
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
         show_continuum_fit,
         save_dir): 

    """
    Fit power-law to background
    """

    k = obj[0]
    x = obj[1]
    y = obj[2]
    er = obj[3]
    flux_array_fit_k = obj[4]
    err_array_fit_k = obj[5]
    wav_array_fit_k = obj[6] 

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

    # Need to do like this because operations do nothing to masked elements 
    mask = ma.getmask(wav_array_fit_k)

    wav_array_fit_k = ma.getdata(wav_array_fit_k)
    flux_array_fit_k = ma.getdata(flux_array_fit_k)
    err_array_fit_k = ma.getdata(err_array_fit_k)

    # flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] = \
    #     flux_array_fit_k[~ma.getmaskarray(wav_array_fit_k)] / \
    #     resid(params=out.params, 
    #           x=ma.getdata(wav_array_fit_k[~ma.getmaskarray(wav_array_fit_k)]).value, 
    #           model=bkgdmod)

    # err_array_fit_k[~ma.getmaskarray(err_array_fit_k)] = \
    #     err_array_fit_k[~ma.getmaskarray(err_array_fit_k)] / \
    #     resid(params=out.params, 
    #           x=ma.getdata(wav_array_fit_k[~ma.getmaskarray(wav_array_fit_k)]).value, 
    #           model=bkgdmod)

    flux_array_fit_k = flux_array_fit_k / resid(params=out.params, 
                                                x=wav_array_fit_k.value, 
                                                model=bkgdmod)

    err_array_fit_k = err_array_fit_k / resid(params=out.params, 
                                              x=wav_array_fit_k.value, 
                                              model=bkgdmod)

    # save parameters 

    # remove expressions from Parameters instance
    # https://groups.google.com/forum/#!topic/lmfit-py/6tCcTNe307I
    
    if save_dir is not None: 

        params_dump = deepcopy(out.params)
        for v in params_dump:
            params_dump[v].expr = None
            
        param_file = os.path.join(save_dir, 'my_params_remove_slope.txt')
        parfile = open(param_file, 'w')
        params_dump.dump(parfile)
        parfile.close()



    flux_array_fit_k = ma.array(flux_array_fit_k, mask=mask)
    err_array_fit_k = ma.array(err_array_fit_k, mask=mask)


    if verbose & show_continuum_fit:

        plot_continum(x, 
                      y, 
                      er,
                      wav, 
                      flux, 
                      err, 
                      False, 
                      out.params, 
                      bkgdmod, 
                      None, 
                      continuum_region, 
                      maskout, 
                      w0, 
                      reject_outliers, 
                      wav_array_blue, 
                      flux_array_blue, 
                      wav_array_red, 
                      flux_array_red)


    return (flux_array_fit_k, err_array_fit_k)


def make_model_mfica(mfica_n_weights=10):

    # Read components 
    comps = np.genfromtxt('/data/vault/phewett/ICAtest/DR12exp/Spectra/hbeta_2154_c10.dat')
    
    comps_wav = comps[:, 0]
    comps = comps[:, 1:mfica_n_weights+1]

    #--------------------------------------------------------------------------------------

    """
    Take out slope from spectrum. Found by fitting power-law in emission line-free windows

    fname = '/home/phewett/projects/sdss_redshifts/qsomod_z250_ng.dat'
    wav, flux = np.genfromtxt(fname, unpack=True)
    
    windows free from strong emission (except fe)
    
    mask = ((wav > 4200) & (wav < 4230)) |\
           ((wav > 4435) & (wav < 4700)) |\
           ((wav > 5100) & (wav < 5535))

    mod = PowerLawModel()
    pars = mod.make_params() 

    result = mod.fit(flux[mask], x=wav[mask])

    print result.params['exponent'].value, result.params['amplitude'].value


    """ 
    
    mod = PowerLawModel()
    pars = mod.make_params() 
    pars['exponent'].value = -2.27531304895 
    pars['amplitude'].value = 112798878.981

    comps = comps / mod.eval(params=pars, x=comps_wav)[:, None]
    
    weights = Parameters() 

    weights.add('w1', value=0.1193, min=0.0)
    weights.add('w2', value=0.0971, min=0.0)
    weights.add('w3', value=0.0467, min=0.0)
    weights.add('w4', value=0.0102, min=0.0)
    weights.add('w5', value=0.0131, min=0.0)
    weights.add('w6', value=0.0205, min=0.0)
    
    # Correction terms --------------------------

    weights.add('w7', value=0.0)
    weights.add('w8', value=0.0)
    weights.add('w9', value=0.0)
    weights.add('w10', value=0.0)

    weights.add('shift', value=-0.1)

    if mfica_n_weights == 8:

        weights['w9'].vary = False
        weights['w10'].vary = False


    return comps_wav, comps, weights 


def make_model_hb(xscale=1.0, 
                  w0=4862.721*u.AA,
                  nGaussians=2, 
                  hb_narrow=True,
                  fix_broad_peaks=True,
                  oiii_broad_off=False,
                  oiii_template=False):

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

    mod = GaussianModel(prefix='oiii_5007_n_')
    mod += GaussianModel(prefix='oiii_5007_b_')
    mod += GaussianModel(prefix='oiii_4959_n_')
    mod += GaussianModel(prefix='oiii_4959_b_')

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
        # pars['hb_b_{}_sigma'.format(i)].max = 20000.0 / 2.35
    
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

    if oiii_template is True:

        """
        For low S/N use low-z SDSS composite 
        /data/lc585/nearIR_spectra/linefits/SDSSComposite/my_params.txt
        """


        pars['oiii_5007_n_sigma'].set(value=167.69, vary=False, min=None, max=None, expr=None)
        pars['oiii_5007_b_sigma'].set(value=510.53, vary=False, min=None, max=None, expr=None)
        pars['oiii_4959_n_sigma'].set(value=167.69, vary=False, min=None, max=None, expr=None)
        pars['oiii_4959_b_sigma'].set(value=510.53, vary=False, min=None, max=None, expr=None)
        pars['hb_n_sigma'].set(value=167.69, vary=False, min=None, max=None, expr=None)
        
        pars.add('oiii_scale') 
        pars['oiii_scale'].value = 1.0 
        pars['oiii_scale'].min = 0.0 

        pars['oiii_5007_n_amplitude'].set(value=233.24, vary=False, expr='oiii_scale*233.24', min=None, max=None)
        pars['oiii_5007_b_amplitude'].set(value=236.18, vary=False, expr='oiii_scale*236.18', min=None, max=None)
        pars['oiii_4959_n_amplitude'].set(value=77.74, vary=False, expr='oiii_scale*77.74', min=None, max=None)
        pars['oiii_4959_b_amplitude'].set(value=78.72, vary=False, expr='oiii_scale*78.72', min=None, max=None)
        pars['hb_n_amplitude'].set(value=53.24, vary=False, expr='oiii_scale*53.24', min=None, max=None)

 
        pars.add('oiii_n_center_delta')
        pars['oiii_n_center_delta'].min = -500.0
        pars['oiii_n_center_delta'].max = 500.0
        pars['oiii_n_center_delta'].value = 0.0  
        pars['oiii_n_center_delta'].vary = True  
        
        pars.add('oiii_5007_center_fixed')
        pars['oiii_5007_center_fixed'].value = wave2doppler(5008.239*u.AA, w0).value
        pars['oiii_5007_center_fixed'].vary = False 

        pars.add('oiii_4959_center_fixed')
        pars['oiii_4959_center_fixed'].value = wave2doppler(4960.295*u.AA, w0).value
        pars['oiii_4959_center_fixed'].vary = False 

        pars['oiii_5007_n_center'].set(value=wave2doppler(5008.239*u.AA, w0).value, vary=True, min=None, max=None, expr='oiii_5007_center_fixed+oiii_n_center_delta')
        pars['oiii_4959_n_center'].set(value=wave2doppler(4960.295*u.AA, w0).value, vary=True, min=None, max=None, expr='oiii_4959_center_fixed+oiii_n_center_delta')
        pars['hb_n_center'].set(value=0.0, vary=True, min=None, max=None, expr='oiii_n_center_delta')

        pars.add('oiii_5007_b_center_delta') 
        pars['oiii_5007_b_center_delta'].set(value=200.416080, vary=False, min=None, max=None, expr=None)
        pars['oiii_5007_b_center'].set(expr='oiii_5007_n_center-oiii_5007_b_center_delta')
        pars['oiii_4959_b_center'].set(expr='oiii_4959_n_center-oiii_5007_b_center_delta')

    return mod, pars



def make_model_ha(x,
                  xscale=1.0, 
                  w0=6564.89*u.AA,
                  nGaussians=2,
                  fix_broad_peaks=True,
                  ha_narrow_fwhm=1200.0,
                  ha_narrow_voff=0.0,
                  ha_narrow_vary=True):


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

    return mod, pars 

def make_model_gh(xscale=1.0,
                  gh_order=4,
                  fmin=1500.0,
                  fmax=1600.0):

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

        pars['cen{}'.format(i)].min = fmin / xscale
        pars['cen{}'.format(i)].max = fmax / xscale

        pars['sig{}'.format(i)].min = 0.1

        pars['amp{}'.format(i)].min = 0.0  



    return mod, pars 


def make_model_multigauss(x,
                          y,
                          nGaussians=2,
                          fix_broad_peaks=False):

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

    return mod, pars  
  


def make_model_oiii(xscale=1.0, 
                    w0=4862.721*u.AA,
                    nGaussians=2, 
                    hb_narrow=True,
                    fix_broad_peaks=True,
                    oiii_broad_off=False,
                    oiii_template=False,
                    fix_oiii_peak_ratio=True):


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

    mod = GaussianModel(prefix='oiii_5007_n_')
    mod += GaussianModel(prefix='oiii_5007_b_')
    mod += GaussianModel(prefix='oiii_4959_n_')
    mod += GaussianModel(prefix='oiii_4959_b_')

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
    pars.add('oiii_peak_ratio') 
    pars['oiii_peak_ratio'].value = 0.3333

    pars['oiii_4959_n_amplitude'].set(expr='oiii_peak_ratio*oiii_5007_n_amplitude')
    pars['oiii_4959_b_amplitude'].set(expr='oiii_peak_ratio*oiii_5007_b_amplitude')

    if fix_oiii_peak_ratio:
        pars['oiii_peak_ratio'].vary = False  

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

    pars['oiii_5007_n_center'].min = wave2doppler(5008.239*u.AA, w0).value - 3000.0
    pars['oiii_5007_n_center'].max = wave2doppler(5008.239*u.AA, w0).value + 3000.0
    
    altfit = True 

    if hb_narrow is True: 

        if altfit:
            pars['hb_n_center'].min = -1000.0
            pars['hb_n_center'].max = 1000.0
        else:
            pars['hb_n_center'].set(expr = 'oiii_5007_n_center-{}'.format(wave2doppler(5008.239*u.AA, w0).value)) 
    
    pars['oiii_4959_n_center'].set(expr = 'oiii_5007_n_center+{}'.format(wave2doppler(4960.295*u.AA, w0).value - wave2doppler(5008.239*u.AA, w0).value))

    pars.add('oiii_5007_b_center_delta') 
    pars['oiii_5007_b_center_delta'].value = 500.0 
    pars['oiii_5007_b_center_delta'].min = -500.0
    pars['oiii_5007_b_center_delta'].max = 2000.0
    pars['oiii_5007_b_center'].set(expr='oiii_5007_n_center-oiii_5007_b_center_delta')
    pars['oiii_4959_b_center'].set(expr='oiii_4959_n_center-oiii_5007_b_center_delta')

    # Set broad components of OIII to have fixed relative velocity
    # pars['oiii_4959_b_center'].set(expr = 'oiii_5007_b_center+{}'.format(\
    #     wave2doppler(4960.295*u.AA, w0).value - wave2doppler(5008.239*u.AA, w0).value))


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
        # pars['hb_b_{}_sigma'.format(i)].max = 20000.0 / 2.35
    
    pars['oiii_5007_n_sigma'].min = 100.0 / 2.35 
    pars['oiii_5007_n_sigma'].max = 2000.0 / 2.35 

    pars['oiii_4959_n_sigma'].set(expr='oiii_5007_n_sigma')

    if hb_narrow is True: 
        if altfit:
            pars['hb_n_sigma'].min = 100.0 / 2.35 
            pars['hb_n_sigma'].max = 2000.0 / 2.35 
        else:
            pars['hb_n_sigma'].set(expr='oiii_5007_n_sigma')

    # make sure broad component of oiii is broader than narrow component 
    # pars.add('oiii_5007_b_sigma_delta')
    # pars['oiii_5007_b_sigma_delta'].value = 0.5
    # pars['oiii_5007_b_sigma_delta'].max = 0.9
    # pars['oiii_5007_n_sigma'].set(expr='oiii_5007_b_sigma*oiii_5007_n_sigma_delta', min=100.0/2.35, max=1200.0/2.35)

    pars['oiii_5007_b_sigma'].max = 2000.0 / 2.35 
    pars['oiii_5007_b_sigma'].min = 100.0 / 2.35 
    pars['oiii_4959_b_sigma'].set(expr='oiii_5007_b_sigma')

    if oiii_broad_off:

        pars['oiii_4959_b_amplitude'].set(value=0.0, vary=False)
        pars['oiii_5007_b_amplitude'].set(value=0.0, vary=False)
        pars['oiii_4959_b_sigma'].set(value=1200.0/2.35, vary=False)
        pars['oiii_5007_b_sigma'].set(value=1200.0/2.35, vary=False)
        pars['oiii_4959_b_center'].set(value=wave2doppler(4960.295*u.AA, w0).value, vary=False)
        pars['oiii_5007_b_center'].set(value=wave2doppler(5008.239*u.AA, w0).value, vary=False)

    if oiii_template is True:

        """
        For low S/N use low-z SDSS composite 
        /data/lc585/nearIR_spectra/linefits/SDSSComposite/my_params.txt
        """


        pars['oiii_5007_n_sigma'].set(value=167.69, vary=False, min=None, max=None, expr=None)
        pars['oiii_5007_b_sigma'].set(value=510.53, vary=False, min=None, max=None, expr=None)
        pars['oiii_4959_n_sigma'].set(value=167.69, vary=False, min=None, max=None, expr=None)
        pars['oiii_4959_b_sigma'].set(value=510.53, vary=False, min=None, max=None, expr=None)
        
        if hb_narrow is True and altfit is not True:
            pars['hb_n_sigma'].set(value=167.69, vary=False, min=None, max=None, expr=None)
        
        pars.add('oiii_scale') 
        pars['oiii_scale'].value = 1.0 
        pars['oiii_scale'].min = 0.0 

        pars['oiii_5007_n_amplitude'].set(value=233.24, vary=False, expr='oiii_scale*233.24', min=None, max=None)
        pars['oiii_5007_b_amplitude'].set(value=236.18, vary=False, expr='oiii_scale*236.18', min=None, max=None)
        pars['oiii_4959_n_amplitude'].set(value=77.74, vary=False, expr='oiii_scale*77.74', min=None, max=None)
        pars['oiii_4959_b_amplitude'].set(value=78.72, vary=False, expr='oiii_scale*78.72', min=None, max=None)
        if hb_narrow is True and altfit is not True:
            pars['hb_n_amplitude'].set(value=53.24, vary=False, expr='oiii_scale*53.24', min=None, max=None)

 
        pars.add('oiii_n_center_delta')
        pars['oiii_n_center_delta'].min = -500.0
        pars['oiii_n_center_delta'].max = 500.0
        pars['oiii_n_center_delta'].value = 0.0  
        pars['oiii_n_center_delta'].vary = False # Change this to true if want offset to vary.   
        
        pars.add('oiii_5007_center_fixed')
        pars['oiii_5007_center_fixed'].value = wave2doppler(5008.239*u.AA, w0).value
        pars['oiii_5007_center_fixed'].vary = False 

        pars.add('oiii_4959_center_fixed')
        pars['oiii_4959_center_fixed'].value = wave2doppler(4960.295*u.AA, w0).value
        pars['oiii_4959_center_fixed'].vary = False 

        pars['oiii_5007_n_center'].set(value=wave2doppler(5008.239*u.AA, w0).value, vary=True, min=None, max=None, expr='oiii_5007_center_fixed+oiii_n_center_delta')
        pars['oiii_4959_n_center'].set(value=wave2doppler(4960.295*u.AA, w0).value, vary=True, min=None, max=None, expr='oiii_4959_center_fixed+oiii_n_center_delta')
        if hb_narrow is True and altfit is not True:
            pars['hb_n_center'].set(value=0.0, vary=True, min=None, max=None, expr='oiii_n_center_delta')

        pars.add('oiii_5007_b_center_delta') 
        pars['oiii_5007_b_center_delta'].set(value=200.416080, vary=False, min=None, max=None, expr=None)
        pars['oiii_5007_b_center'].set(expr='oiii_5007_n_center-oiii_5007_b_center_delta')
        pars['oiii_4959_b_center'].set(expr='oiii_4959_n_center-oiii_5007_b_center_delta')

    return mod, pars 


def make_model_siiv(w0=1400*u.AA):

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

    return mod, pars 



def get_sigma(vl, fl, dv):

    norm = np.sum(fl * dv) 
    pdf = fl / norm 

    m = np.sum(vl * pdf * dv)
    v = np.sum((vl-m)**2 * pdf * dv)

    return np.sqrt(v)

def get_median(vl, fl, dv):

    norm = np.sum(fl * dv) 
    pdf = fl / norm 
    cdf = np.cumsum(pdf)  
    return vl[np.argmin(np.abs(cdf - 0.5))]

def get_fwhm(vl, fl):

    """
    vl: input velocity array
    fl: input flux array
    """

    half_max = np.max(fl) / 2.0

    i = 0
    while fl[i] < half_max:
        i+=1
    
    root1 = vl[i]
    
    i = 0
    while fl[-i] < half_max:
        i+=1
    
    root2 = vl[-i]

    return root2 - root1


def get_shape(vl, fl):


    """
    Following Baskin & Loar 2005, calculate the shape of the line
    This could certainly be optimized. 
    """


    half_max = np.max(fl) / 2.0

    i = 0
    while fl[i] < half_max:
        i+=1
    
    root1 = vl[i]
    
    i = 0
    while fl[-i] < half_max:
        i+=1
    
    root2 = vl[-i]

    fwhm = root2 - root1

    quarter_max = np.max(fl) / 4.0 

    i = 0
    while fl[i] < quarter_max:
        i+=1
    
    root1 = vl[i]
    
    i = 0
    while fl[-i] < quarter_max:
        i+=1
    
    root2 = vl[-i]

    fwqm = root2 - root1

    three_quarter_max = 3.0 * np.max(fl) / 4.0

    i = 0
    while fl[i] < three_quarter_max:
        i+=1
    
    root1 = vl[i]
    
    i = 0
    while fl[-i] < three_quarter_max:
        i+=1
    
    root2 = vl[-i]

    fw3qm = root2 - root1

    return (fwqm + fw3qm) / 2.0 / fwhm  




def get_eqw(vl, 
            fl, 
            w0=6564.89*u.AA, 
            bkgdmod=None, 
            bkgdpars_k=None, 
            sp_fe=None,
            subtract_fe=False):

    xs_wav = doppler2wave(vl*(u.km/u.s), w0)
    
    if subtract_fe is True:

        flux_bkgd = resid(params=bkgdpars_k, 
                          x=xs_wav.value, 
                          model=bkgdmod,
                          sp_fe=sp_fe)
    
    if subtract_fe is False:

        flux_bkgd = resid(params=bkgdpars_k, 
                          x=xs_wav.value, 
                          model=bkgdmod)
    
    f = (fl + flux_bkgd) / flux_bkgd
    eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)

    return np.nansum(eqw)

def get_center(vl, fl, dv):

    norm = np.sum(fl * dv) 
    pdf = fl / norm 

    return vl[np.argmax(pdf)] 

def get_lum(vl, 
            fl, 
            z=0.0,
            w0=6564.89*u.AA,
            spec_norm=1.0):

    xs_wav = doppler2wave(vl*(u.km/u.s), w0)

    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)    

    return np.sum(fl[:-1] * np.diff(xs_wav.value)) * (u.erg / u.s / u.cm / u.cm) * (1.0 + z) * 4.0 * math.pi * lumdist**2  / spec_norm 


def get_broad_stats(vl,
                    fl,
                    dv=1.0, 
                    w0=6564.89*u.AA,
                    bkgdmod=None,
                    bkgdpars_k=None, 
                    sp_fe=None,
                    subtract_fe=None,
                    spec_norm=None,
                    z=0.0):                


    """
    Get stats for broad component
    """

    fwhm = get_fwhm(vl, fl)

    sd = get_sigma(vl, fl, dv)

    md = get_median(vl, fl, dv)

    eqw = get_eqw(vl,
                  fl,
                  w0=w0, 
                  bkgdmod=bkgdmod, 
                  bkgdpars_k=bkgdpars_k, 
                  sp_fe=sp_fe,
                  subtract_fe=subtract_fe)

    peak_flux = np.max(fl) / spec_norm

    func_center = get_center(vl, fl, dv)

    broad_lum = get_lum(vl, 
                        fl, 
                        z=z,
                        w0=w0,
                        spec_norm=spec_norm)



    shape = get_shape(vl, fl)
    


    return (fwhm, sd, md, eqw, peak_flux, func_center, broad_lum, shape)

def get_stats_oiii(out,
                   mod=None,
                   xscale=1.0, 
                   w0=4862.721*u.AA,
                   plot_title='',
                   bkgdmod=None,
                   bkgdpars_k=None,
                   sp_fe=None,
                   subtract_fe=True,
                   z=0.0,
                   spec_norm=1.0,
                   varray=None,
                   farray=None,
                   sarray=None,
                   nGaussians=1,
                   oiii_broad_off=False,
                   hb_narrow=False):

    mod_oiii_5007 = GaussianModel(prefix='oiii_5007_b_') + GaussianModel(prefix='oiii_5007_n_') 
    mod_oiii_5007_pars = mod_oiii_5007.make_params() 

    mod_oiii_5007_pars['oiii_5007_b_amplitude'].value = out.params['oiii_5007_b_amplitude'].value 
    mod_oiii_5007_pars['oiii_5007_b_sigma'].value = out.params['oiii_5007_b_sigma'].value
    mod_oiii_5007_pars['oiii_5007_b_center'].value = out.params['oiii_5007_b_center'].value

    mod_oiii_5007_pars['oiii_5007_n_amplitude'].value = out.params['oiii_5007_n_amplitude'].value 
    mod_oiii_5007_pars['oiii_5007_n_sigma'].value = out.params['oiii_5007_n_sigma'].value
    mod_oiii_5007_pars['oiii_5007_n_center'].value = out.params['oiii_5007_n_center'].value

    integrand = lambda x: mod_oiii_5007.eval(params=mod_oiii_5007_pars, x=np.array(x) + wave2doppler(5008.239*u.AA, w0).value) 
   

    dv = 1.0 
    xs = np.arange(-10000.0, 10000.0, dv) / xscale 
        
    norm = np.sum(integrand(xs) * dv)
    pdf = integrand(xs) / norm
    cdf = np.cumsum(pdf)

    oiii_5007_eqw = get_eqw(xs*xscale,
                            integrand(xs),
                            w0=w0, 
                            bkgdmod=bkgdmod, 
                            bkgdpars_k=bkgdpars_k, 
                            sp_fe=sp_fe,
                            subtract_fe=subtract_fe)
            
    oiii_5007_lum = get_lum(xs*xscale, 
                            integrand(xs), 
                            z=z,
                            w0=w0,
                            spec_norm=spec_norm)
            
    # absolute assymetry - not perfect because depends on somewhat arbitary
    # z_IR, but should do for now. 

    # xs_blue = np.arange(-10000.0, 0.0, dv) / xscale 
    # xs_red = np.arange(0.0, 10000.0, dv) / xscale 

    # A = (np.sum(integrand(xs_blue) * dv) - np.sum(integrand(xs_red) * dv)) / norm
    # print A, 1.0 - 2.0*cdf[xs == 0.0]

    #------------------------------------------------------------------------
   
    # Calculate S/N of 5008 within w90 
    varray_oiii = varray - wave2doppler(5008.239*u.AA, w0).value # center on OIII 

    vmin_oiii = np.argmin(np.abs(varray_oiii - xs[np.argmin(np.abs(cdf - 0.05))]))
    vmax_oiii = np.argmin(np.abs(varray_oiii - xs[np.argmin(np.abs(cdf - 0.95))]))
    
    varray_oiii = varray_oiii[vmin_oiii: vmax_oiii]
    farray_oiii = farray[vmin_oiii: vmax_oiii]
    sarray_oiii = sarray[vmin_oiii: vmax_oiii]

    # snr_oiii = np.median(integrand(varray_oiii/xscale) / sarray_oiii)

    #------------------------------------------------------------------------

    """Alternative calculation of S/N in OIII """


    try: 

        # We use peak to avoid assigning a high S/N to objects which are fit with 
        # low amplitude but very broad gaussians. 
        signal_oiii = np.max(integrand(varray_oiii/xscale))

        # rebin noise to 200 km/s - minimum in our sample 

        if varray_oiii.max() - varray_oiii.min() > 200.0: 
        
            varray_oiii_rebin, farray_oiii_rebin, sarray_oiii_rebin = rebin(varray_oiii, 
                                                                            farray_oiii, 
                                                                            sarray_oiii,
                                                                            np.arange(varray_oiii.min(),
                                                                                      varray_oiii.max(),
                                                                                      200.0),
                                                                            weighted=True)

        else:

            # Unanable to do binning 
            varray_oiii_rebin = varray_oiii
            farray_oiii_rebin = farray_oiii

        # subtract signal (from all components) and calculate noise  
        integrand_full =  lambda x: mod.eval(params=out.params, x=np.array(x) + wave2doppler(5008.239*u.AA, w0).value) 

        noise_oiii = np.nanstd(farray_oiii_rebin - integrand_full(varray_oiii_rebin/xscale)) 

        snr_oiii = signal_oiii / noise_oiii



    except ValueError as e:
        snr_oiii = np.nan 

    if np.isinf(snr_oiii):
        snr_oiii = np.nan

    #------------------------------------------------------------------------

    # Calculate S/N of Hb within w90 

    mod_broad_hb = ConstantModel()
    
    for i in range(nGaussians):
        mod_broad_hb += GaussianModel(prefix='hb_b_{}_'.format(i))  
    
    pars_broad_hb = mod_broad_hb.make_params()
    
    pars_broad_hb['c'].value = 0.0
    
    for key, value in out.params.valuesdict().iteritems():
        if key.startswith('hb_b_'):
            pars_broad_hb[key].value = value 

    integrand_hb = lambda x: mod_broad_hb.eval(params=pars_broad_hb, x=np.array(x))

    norm_hb = np.sum(integrand_hb(xs) * dv)
    pdf_hb = integrand_hb(xs) / norm_hb
    cdf_hb = np.cumsum(pdf_hb)

    vmin_hb = np.argmin(np.abs(varray - xs[np.argmin(np.abs(cdf_hb - 0.10))]))
    vmax_hb = np.argmin(np.abs(varray - xs[np.argmin(np.abs(cdf_hb - 0.90))]))
    
    varray_hb = varray[vmin_hb: vmax_hb]
    farray_hb = farray[vmin_hb: vmax_hb]
    sarray_hb = sarray[vmin_hb: vmax_hb]
    
    # We use the model flux rather than the data to calculate the signal-to-noise
    # snr_hb = np.median(integrand_hb(varray_hb/xscale) / sarray_hb)



    #-------------------------------------------------------------------------

    """Alternative calculation of S/N in Hb"""

    try: 
        signal_hb = np.max(integrand_hb(varray_hb/xscale))

        # rebin noise to 200 km/s - minimum in our sample 

        if varray_hb.max() - varray_hb.min() > 200.0: 
        
            varray_hb_rebin, farray_hb_rebin, sarray_hb_rebin = rebin(varray_hb, 
                                                                      farray_hb, 
                                                                      sarray_hb,
                                                                      np.arange(varray_hb.min(),
                                                                                varray_hb.max(),
                                                                                200.0),
                                                                      weighted=True)

        else:

            # Unanable to do binning 
            varray_hb_rebin = varray_hb
            farray_hb_rebin = farray_hb

        # subtract signal (from all components) and calculate noise  
        integrand_full =  lambda x: mod.eval(params=out.params, x=np.array(x)) 

        noise_hb = np.nanstd(farray_hb_rebin - integrand_full(varray_hb_rebin/xscale)) 

        snr_hb = signal_hb / noise_hb

    except ValueError as e:
        snr_hb = np.nan 

    if np.isinf(snr_hb):
        snr_hb = np.nan

    #------------------------------------------------------------------------


    # FWHM. Not really well defined if double-peaked. 
    # In this case need to use Peterson et al. (2004) perscription
    fwhm = get_fwhm(xs*xscale, integrand(xs)) 

    #------------------------------------------------------------------------

    
    # Get centroid of narrower OIII component 
    if oiii_broad_off is True:
        oiii_5007_n_cen = out.params['oiii_5007_n_center'].value 
    else:
        if out.params['oiii_5007_n_sigma'].value < out.params['oiii_5007_b_sigma'].value:
            oiii_5007_n_cen = out.params['oiii_5007_n_center'].value 
        else:
            oiii_5007_n_cen = out.params['oiii_5007_b_center'].value 

    woiii = doppler2wave(oiii_5007_n_cen*(u.km/u.s), w0)
    woiii = woiii * (1.0 + z) 
    oiii_narrow_z = (woiii / (5008.239*u.AA)) - 1.0

    # Following Shen (2016), calculate redshift from peak of whole OIII profile

    oiii_peak = xs[np.argmax(integrand(xs))] * xscale
    woiii = doppler2wave(oiii_peak*(u.km/u.s) + wave2doppler(5008.239*u.AA, w0), w0)
    woiii = woiii * (1.0 + z) 
    oiii_full_peak_z = (woiii / (5008.239*u.AA)) - 1.0    

    #------------------------------------------------------------------------
    
    """Calculate redshift from peak/median of broad+narrow components"""

    mod_hb = ConstantModel()

    if hb_narrow:
        mod_hb += GaussianModel(prefix='hb_n_')

    for i in range(nGaussians):
        mod_hb += GaussianModel(prefix='hb_b_{}_'.format(i))  
    
    pars_hb = mod_hb.make_params()

    for key, value in out.params.valuesdict().iteritems():
        if key.startswith('hb_'):
            pars_hb[key].value = value 

    pars_hb['c'].value = 0.0

    integrand_hb = lambda x: mod_hb.eval(params=pars_hb, x=np.array(x))

    peak = xs[np.argmax(integrand_hb(xs))]*xscale

    w = doppler2wave(peak*(u.km/u.s), w0) * (1.0 + z)
    hb_z = (w / w0) - 1.0    


    #------------------------------------------------------------------------


    
    fit_out = {'name':plot_title,
               'oiii_5007_v5':xs[np.argmin(np.abs(cdf - 0.05))],
               'oiii_5007_v10':xs[np.argmin(np.abs(cdf - 0.1))],
               'oiii_5007_v25':xs[np.argmin(np.abs(cdf - 0.25))],
               'oiii_5007_v50':xs[np.argmin(np.abs(cdf - 0.50))],
               'oiii_5007_v75':xs[np.argmin(np.abs(cdf - 0.75))],
               'oiii_5007_v90':xs[np.argmin(np.abs(cdf - 0.90))],
               'oiii_5007_v95':xs[np.argmin(np.abs(cdf - 0.95))],
               'oiii_5007_eqw':oiii_5007_eqw,
               'oiii_5007_lum':np.log10(oiii_5007_lum.value),
               'oiii_5007_snr':snr_oiii,
               'oiii_5007_fwhm':fwhm,
               'oiii_peak_ratio':out.params['oiii_peak_ratio'].value, 
               'oiii_5007_narrow_z':oiii_narrow_z,
               'oiii_5007_full_peak_z':oiii_full_peak_z,
               'oiii_5007_full_peak_vel':oiii_peak,
               'hb_snr':snr_hb, 
               'hb_z':hb_z,
               'redchi':out.redchi}

    return fit_out 

def get_stats_hb(out=None, 
                 line_region=[-10000.0, 10000.0]*(u.km/u.s), 
                 xscale=1.0,
                 w0=4862.721*u.AA,
                 bkgdmod=None,
                 bkgdpars_k=None,
                 sp_fe=None,
                 subtract_fe=False,
                 spec_norm=None,
                 z=0.0,
                 mod=None,
                 nGaussians=1,
                 plot_title='',
                 hb_narrow=True):


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

    dv = 1.0 # i think if this was anything else might need to change intergration.
    xs = np.arange(line_region.value[0], line_region[1].value, dv) / xscale 

    fwhm, sd, md, eqw, peak_flux, func_center, broad_lum, shape = get_broad_stats(xs*xscale, 
                                                                                  integrand(xs),
                                                                                  dv=dv, 
                                                                                  w0=w0,
                                                                                  bkgdmod=bkgdmod,
                                                                                  bkgdpars_k=bkgdpars_k,
                                                                                  subtract_fe=subtract_fe,
                                                                                  sp_fe=sp_fe,
                                                                                  spec_norm=spec_norm,
                                                                                  z=z)    

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

        very_broad_frac = np.nan 

    else: 

        very_broad_mod = GaussianModel()
        very_broad_pars = very_broad_mod.make_params()

        for key, value in out.params.valuesdict().iteritems():
            if key.startswith('hb_b_{}'.format(inds[0])):
                very_broad_pars[key.replace('hb_b_{}_'.format(inds[0]), '')].value = value 
     
        quite_broad_mod = GaussianModel()
        quite_broad_pars = quite_broad_mod.make_params()

        for key, value in out.params.valuesdict().iteritems():
            if key.startswith('hb_b_{}'.format(inds[1])):
                quite_broad_pars[key.replace('hb_b_{}_'.format(inds[1]), '')].value = value 

        very_broad_integrand = very_broad_mod.eval(params=very_broad_pars, x=np.array(xs))
        quite_broad_integrand = quite_broad_mod.eval(params=quite_broad_pars, x=np.array(xs))

        very_broad_frac = np.sum(very_broad_integrand) / (np.sum(quite_broad_integrand) + np.sum(very_broad_integrand))

   
    oiii_5007_b_mod = GaussianModel()
    oiii_5007_b_pars = oiii_5007_b_mod.make_params() 
    
    oiii_5007_b_pars['amplitude'].value = out.params['oiii_5007_b_amplitude'].value
    oiii_5007_b_pars['sigma'].value = out.params['oiii_5007_b_sigma'].value 
    oiii_5007_b_pars['center'].value = out.params['oiii_5007_b_center'].value 
    
    oiii_5007_b_xs = np.arange(oiii_5007_b_pars['center'].value - 10000.0, 
                               oiii_5007_b_pars['center'].value + 10000.0, 
                               dv)
    
    oiii_5007_b_lum = get_lum(oiii_5007_b_xs,
                              oiii_5007_b_mod.eval(params=oiii_5007_b_pars, x=oiii_5007_b_xs),
                              z=z,
                              w0=w0,
                              spec_norm=spec_norm)   

    
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
    
    oiii_5007_n_lum = get_lum(oiii_5007_n_xs,
                              oiii_5007_n_mod.eval(params=oiii_5007_n_pars, x=oiii_5007_n_xs),
                              z=z,
                              w0=w0,
                              spec_norm=spec_norm)
    
    oiii_5007_n_fwhm = oiii_5007_n_pars['sigma'].value * 2.35 
       
    oiii_5007_mod = GaussianModel(prefix='oiii_5007_n_') + GaussianModel(prefix='oiii_5007_b_')
    oiii_5007_pars = oiii_5007_mod.make_params() 
    
    oiii_5007_pars['oiii_5007_n_amplitude'].value = out.params['oiii_5007_n_amplitude'].value
    oiii_5007_pars['oiii_5007_n_sigma'].value = out.params['oiii_5007_n_sigma'].value 
    oiii_5007_pars['oiii_5007_n_center'].value = out.params['oiii_5007_n_center'].value 
    oiii_5007_pars['oiii_5007_b_amplitude'].value = out.params['oiii_5007_b_amplitude'].value
    oiii_5007_pars['oiii_5007_b_sigma'].value = out.params['oiii_5007_b_sigma'].value 
    oiii_5007_pars['oiii_5007_b_center'].value = out.params['oiii_5007_b_center'].value 
   

    oiii_5007_eqw = get_eqw(oiii_5007_n_xs,
                            oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs),
                            w0=w0, 
                            bkgdmod=bkgdmod, 
                            bkgdpars_k=bkgdpars_k, 
                            sp_fe=sp_fe,
                            subtract_fe=subtract_fe)


    
    oiii_5007_lum = get_lum(oiii_5007_n_xs,
                            oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs),
                            z=z,
                            w0=w0,
                            spec_norm=spec_norm)  
    


    oiii_5007_fwhm = get_fwhm(oiii_5007_n_xs,
                              oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs))

    # median for blueshift 
    oiii_5007_pdf = oiii_5007_mod.eval(params=oiii_5007_pars, x=oiii_5007_n_xs)
    oiii_5007_norm = np.sum(oiii_5007_pdf * dv)
    oiii_5007_cdf = np.cumsum(oiii_5007_pdf / oiii_5007_norm)       
    oiii_5007_05_percentile = out.params['oiii_5007_n_center'].value - oiii_5007_n_xs[np.argmin( np.abs(oiii_5007_cdf - 0.5))]

    # 0.25 quartile OIII composite 
    oiii_5007_025_percentile = out.params['oiii_5007_n_center'].value - oiii_5007_n_xs[np.argmin( np.abs(oiii_5007_cdf - 0.25))]
    oiii_5007_01_percentile = out.params['oiii_5007_n_center'].value - oiii_5007_n_xs[np.argmin( np.abs(oiii_5007_cdf - 0.1))]


    #-------------------------------------------------------------------------------------------
    
    narrow_fwhm = out.params['oiii_5007_n_sigma'].value * 2.35 
    narrow_voff = out.params['oiii_5007_n_center'].value  - wave2doppler(5008.239*u.AA, w0).value  

    if hb_narrow is True:
    
        narrow_mod = GaussianModel()
        narrow_pars = narrow_mod.make_params()
        narrow_pars['amplitude'].value = out.params['hb_n_amplitude'].value
        narrow_pars['sigma'].value = out.params['hb_n_sigma'].value 
        narrow_pars['center'].value = out.params['hb_n_center'].value 

        narrow_lum = get_lum(xs*xscale,
                             narrow_mod.eval(params=narrow_pars, x=xs),
                             z=z,
                             w0=w0,
                             spec_norm=spec_norm)            
    
    else:
    
        narrow_lum = np.nan * (u.erg / u.s)          

    fit_out = {'name':plot_title, 
               'fwhm':fwhm,
               'fwhm_1':broad_fwhms[0],
               'fwhm_2':broad_fwhms[1],
               'sigma': sd,
               'median': md,
               'cen': func_center,
               'cen_1':broad_cens[0],
               'cen_2':broad_cens[1],
               'eqw': eqw,
               'broad_lum':np.log10(broad_lum.value),
               'peak':peak_flux, 
               'amplitude_1':broad_amps[0],
               'amplitude_2':broad_amps[1],
               'very_broad_frac':very_broad_frac,
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
               'oiii_5007_05_percentile':oiii_5007_05_percentile, 
               'oiii_5007_025_percentile':oiii_5007_025_percentile, 
               'oiii_5007_01_percentile':oiii_5007_01_percentile}

   
    return fit_out 

def get_stats_ha(out=None, 
                 line_region=[-10000.0, 10000.0]*(u.km/u.s), 
                 xscale=1.0,
                 w0=6564.89*u.AA,
                 bkgdmod=None,
                 bkgdpars_k=None,
                 sp_fe=None,
                 subtract_fe=False,
                 spec_norm=None,
                 z=0.0,
                 mod=None,
                 nGaussians=1,
                 plot_title='',
                 varray=None,
                 farray=None,
                 sarray=None,
                 emission_line='Ha'):

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

    dv = 1.0 # i think if this was anything else might need to change intergration.
    xs = np.arange(line_region.value[0], line_region[1].value, dv) / xscale 

    fwhm, sd, md, eqw, peak_flux, func_center, broad_lum, shape = get_broad_stats(xs*xscale, 
                                                                                 integrand(xs),
                                                                                 dv=dv, 
                                                                                 w0=w0,
                                                                                 bkgdmod=bkgdmod,
                                                                                 bkgdpars_k=bkgdpars_k,
                                                                                 subtract_fe=subtract_fe,
                                                                                 sp_fe=sp_fe,
                                                                                 spec_norm=spec_norm,
                                                                                 z=z)    



    # I know that the Gaussians are normalised, so the area is just the amplitude, so surely
    # I shouldn't have to do this function evaluation. 

    broad_fwhms = np.zeros(nGaussians)
    broad_cens = np.zeros(nGaussians)
    broad_amps = np.zeros(nGaussians)

    for i in range(nGaussians):   
        broad_fwhms[i] = np.array(out.params['ha_b_{}_fwhm'.format(i)].value)
        broad_cens[i] = np.array(out.params['ha_b_{}_center'.format(i)].value)
        broad_amps[i] = np.array(out.params['ha_b_{}_amplitude'.format(i)].value)

    inds = np.argsort(broad_fwhms)[::-1] # sorts in ascending order
        
    broad_fwhms = broad_fwhms[inds]
    broad_cens = broad_cens[inds]
    broad_amps = broad_amps[inds]

    if len(broad_fwhms) == 1:

        broad_fwhms = np.append(broad_fwhms, np.nan)
        broad_cens = np.append(broad_cens, np.nan)
        broad_amps = np.append(broad_amps, np.nan)

        very_broad_frac = np.nan 

    else: 

        very_broad_mod = GaussianModel()
        very_broad_pars = very_broad_mod.make_params()

        for key, value in out.params.valuesdict().iteritems():
            if key.startswith('ha_b_{}'.format(inds[0])):
                very_broad_pars[key.replace('ha_b_{}_'.format(inds[0]), '')].value = value 
     
        quite_broad_mod = GaussianModel()
        quite_broad_pars = quite_broad_mod.make_params()

        for key, value in out.params.valuesdict().iteritems():
            if key.startswith('ha_b_{}'.format(inds[1])):
                quite_broad_pars[key.replace('ha_b_{}_'.format(inds[1]), '')].value = value 

        very_broad_integrand = very_broad_mod.eval(params=very_broad_pars, x=np.array(xs))
        quite_broad_integrand = quite_broad_mod.eval(params=quite_broad_pars, x=np.array(xs))

        very_broad_frac = np.sum(very_broad_integrand) / (np.sum(quite_broad_integrand) + np.sum(very_broad_integrand))


    narrow_mod = GaussianModel()
    narrow_pars = narrow_mod.make_params()
    narrow_pars['amplitude'].value = out.params['ha_n_amplitude'].value
    narrow_pars['sigma'].value = out.params['ha_n_sigma'].value 
    narrow_pars['center'].value = out.params['ha_n_center'].value 

    narrow_lum = get_lum(xs*xscale, 
                         narrow_mod.eval(params=narrow_pars, x=xs), 
                         z=z,
                         w0=w0,
                         spec_norm=spec_norm)

    narrow_fwhm = out.params['ha_n_fwhm'].value 
    narrow_voff = out.params['ha_n_center'].value 
     
    #------------------------------------------------------------------------

    """Calculate redshift from peak/median of broad+narrow components"""

    mod_ha = GaussianModel(prefix='ha_n_')

    for i in range(nGaussians):
        mod_ha += GaussianModel(prefix='ha_b_{}_'.format(i))  
    
    pars_ha = mod_ha.make_params()

    for key, value in out.params.valuesdict().iteritems():
        if key.startswith('ha_'):
            pars_ha[key].value = value 

    integrand_ha = lambda x: mod_ha.eval(params=pars_ha, x=np.array(x))

    peak = xs[np.argmax(integrand_ha(xs))]*xscale

    w = doppler2wave(peak*(u.km/u.s), w0) * (1.0 + z)
    peak_z = (w / w0) - 1.0    

  
    # Calculate S/N of within w90 

    norm_ha = np.sum(integrand_ha(xs) * dv)
    pdf_ha = integrand_ha(xs) / norm_ha
    cdf_ha = np.cumsum(pdf_ha)

    vmin_ha = np.argmin(np.abs(varray - xs[np.argmin(np.abs(cdf_ha - 0.10))]))
    vmax_ha = np.argmin(np.abs(varray - xs[np.argmin(np.abs(cdf_ha - 0.90))]))
    
    varray_ha = varray[vmin_ha: vmax_ha]
    farray_ha = farray[vmin_ha: vmax_ha]
    sarray_ha = sarray[vmin_ha: vmax_ha]

    f_ha = interp1d(xs*xscale, integrand_ha(xs))
    
    # We use the model flux rather than the data to calculate the signal-to-noise
    snr_ha = np.median(f_ha(varray_ha) / sarray_ha)

    #------------------------------------------------------------------------
    # Velocity offset of broad component relative to narrow component  
    # Already been sorted above

    broad_offset = broad_cens[0] - broad_cens[1]

    #------------------------------------------------------------------------

    fit_out = {'name':plot_title, 
               'fwhm':fwhm,
               'fwhm_1':broad_fwhms[0],
               'fwhm_2':broad_fwhms[1],
               'sigma': sd,
               'median': md,
               'cen': func_center,
               'cen_1':broad_cens[0],
               'cen_2':broad_cens[1],
               'eqw': eqw,
               'broad_lum':np.log10(broad_lum.value),
               'peak':peak_flux, 
               'amplitude_1':broad_amps[0],
               'amplitude_2':broad_amps[1],
               'very_broad_frac':very_broad_frac,
               'narrow_fwhm':narrow_fwhm,
               'narrow_lum':np.log10(narrow_lum.value) if ~np.isnan(narrow_lum.value) else np.nan,
               'narrow_voff':narrow_voff,
               'peak_z':peak_z,
               'broad_offset':broad_offset,
               'snr_line':snr_ha}

    if emission_line == 'Hb':

        fit_out['oiii_5007_eqw'] = np.nan
        fit_out['oiii_5007_lum'] = np.nan
        fit_out['oiii_5007_n_lum'] = np.nan
        fit_out['oiii_5007_b_lum'] = np.nan
        fit_out['oiii_fwhm'] = np.nan
        fit_out['oiii_n_fwhm'] = np.nan
        fit_out['oiii_b_fwhm'] = np.nan
        fit_out['oiii_5007_b_voff'] = np.nan
        fit_out['oiii_5007_05_percentile'] = np.nan
        fit_out['oiii_5007_025_percentile'] = np.nan
        fit_out['oiii_5007_01_percentile'] = np.nan    

    return fit_out

 

def get_stats_gh(out=None, 
                 line_region=[-10000.0, 10000.0]*(u.km/u.s), 
                 xscale=1.0,
                 w0=np.mean([1548.202,1550.774])*u.AA,
                 bkgdmod=None,
                 bkgdpars_k=None,
                 sp_fe=None,
                 subtract_fe=False,
                 spec_norm=None,
                 z=0.0,
                 mod=None,
                 n_samples=1,
                 verbose=True,
                 plot_title=''):

    """
    Calculate emission line properties
    Works for gauss hermite and siiv models
    """

    integrand = lambda x: mod.eval(params=out.params, x=np.array(x))

    dv = 1.0 # i think if this was anything else might need to change intergration.
    xs = np.arange(line_region.value[0], line_region[1].value, dv) / xscale 
    
    fwhm, sd, md, eqw, peak_flux, func_center, broad_lum, shape = get_broad_stats(xs*xscale, 
                                                                                  integrand(xs),
                                                                                  dv=dv, 
                                                                                  w0=w0,
                                                                                  bkgdmod=bkgdmod,
                                                                                  bkgdpars_k=bkgdpars_k,
                                                                                  subtract_fe=subtract_fe,
                                                                                  spec_norm=spec_norm,
                                                                                  z=z)

    fit_out = {'name':plot_title, 
               'fwhm':fwhm,
               'sigma':sd,
               'median':md,
               'cen': func_center,
               'eqw':eqw,
               'broad_lum':np.log10(broad_lum.value),
               'peak':peak_flux,
               'shape':shape}    

    return fit_out 

def get_stats_multigauss(out=None, 
                         line_region=[-10000.0, 10000.0]*(u.km/u.s), 
                         xscale=1.0,
                         w0=np.mean([1548.202,1550.774])*u.AA,
                         bkgdmod=None,
                         bkgdpars_k=None,
                         sp_fe=None,
                         subtract_fe=False,
                         spec_norm=None,
                         z=0.0,
                         mod=None,
                         nGaussians=1,
                         plot_title='',
                         varray=None,
                         farray=None,
                         sarray=None,
                         emission_line='Ha'):

    integrand = lambda x: mod.eval(params=out.params, x=np.array(x))

    dv = 1.0 # i think if this was anything else might need to change intergration.
    xs = np.arange(line_region.value[0], line_region[1].value, dv) / xscale 
    
    fwhm, sd, md, eqw, peak_flux, func_center, broad_lum, shape = get_broad_stats(xs*xscale, 
                                                                                  integrand(xs),
                                                                                  dv=dv, 
                                                                                  w0=w0,
                                                                                  bkgdmod=bkgdmod,
                                                                                  bkgdpars_k=bkgdpars_k,
                                                                                  subtract_fe=subtract_fe,
                                                                                  sp_fe=sp_fe,
                                                                                  spec_norm=spec_norm,
                                                                                  z=z)

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

        very_broad_frac = np.nan 

    else: 

        very_broad_mod = GaussianModel()
        very_broad_pars = very_broad_mod.make_params()

        for key, value in out.params.valuesdict().iteritems():
            if key.startswith('g{}'.format(inds[0])):
                very_broad_pars[key.replace('g{}_'.format(inds[0]), '')].value = value 
     
        quite_broad_mod = GaussianModel()
        quite_broad_pars = quite_broad_mod.make_params()

        for key, value in out.params.valuesdict().iteritems():
            if key.startswith('g{}'.format(inds[1])):
                quite_broad_pars[key.replace('g{}_'.format(inds[1]), '')].value = value 

        very_broad_integrand = very_broad_mod.eval(params=very_broad_pars, x=np.array(xs))
        quite_broad_integrand = quite_broad_mod.eval(params=quite_broad_pars, x=np.array(xs))

        very_broad_frac = np.sum(very_broad_integrand) / (np.sum(quite_broad_integrand) + np.sum(very_broad_integrand))

    # get redshift from peak of full profile 
    w = doppler2wave(func_center*(u.km/u.s), w0) * (1.0 + z)
    peak_z = (w / w0) - 1.0    


    # Calculate S/N of within w90 -------------------------------------- 

    norm = np.sum(integrand(xs) * dv)
    pdf = integrand(xs) / norm
    cdf = np.cumsum(pdf)

    vmin = np.argmin(np.abs(varray - xs[np.argmin(np.abs(cdf - 0.10))]))
    vmax = np.argmin(np.abs(varray - xs[np.argmin(np.abs(cdf - 0.90))]))
    
    f = interp1d(xs*xscale, integrand(xs))
    
    # We use the model flux rather than the data to calculate the signal-to-noise
    snr_line = np.median(f(varray[vmin: vmax]) / sarray[vmin: vmax])


    #------------------------------------------------------------------------
    # Velocity offset of broad component relative to narrow component  
    # Already been sorted above

    broad_offset = broad_cens[0] - broad_cens[1]




    #------------------------------------------------------------------------


    fit_out = {'name':plot_title, 
               'fwhm':fwhm,
               'fwhm_1':broad_fwhms[0],
               'fwhm_2':broad_fwhms[1],
               'sigma': sd,
               'median': md,
               'cen': func_center,
               'cen_1':broad_cens[0],
               'cen_2':broad_cens[1],
               'peak_z':peak_z,
               'broad_offset':broad_offset,
               'snr_line':snr_line,
               'eqw': eqw,
               'broad_lum':np.log10(broad_lum.value),
               'peak':peak_flux, 
               'amplitude_1':broad_amps[0],
               'amplitude_2':broad_amps[1],
               'very_broad_frac':very_broad_frac,
               'narrow_fwhm':np.nan,
               'narrow_lum':np.nan,
               'narrow_voff':np.nan,
               'shape':shape}

    if emission_line == 'Hb':

        fit_out['oiii_5007_eqw'] = np.nan
        fit_out['oiii_5007_lum'] = np.nan
        fit_out['oiii_5007_n_lum'] = np.nan
        fit_out['oiii_5007_b_lum'] = np.nan
        fit_out['oiii_fwhm'] = np.nan
        fit_out['oiii_n_fwhm'] = np.nan
        fit_out['oiii_b_fwhm'] = np.nan
        fit_out['oiii_5007_b_voff'] = np.nan
        fit_out['oiii_5007_05_percentile'] = np.nan
        fit_out['oiii_5007_025_percentile'] = np.nan
        fit_out['oiii_5007_01_percentile'] = np.nan




    return fit_out 


    


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
         oiii_broad_off,
         oiii_template,
         fix_oiii_peak_ratio,
         load_fit):


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

        sp_fe = None            

        bkgdmod = Model(PLModel, 
                        param_names=['amplitude','exponent'], 
                        independent_vars=['x'])

    xscale = 1.0 

    if fit_model == 'GaussHermite':

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
                       'sigma':np.nan,
                       'median':np.nan,
                       'cen':np.nan,
                       'cen_1':np.nan,
                       'cen_2':np.nan,
                       'eqw':np.nan,
                       'broad_lum':np.nan,
                       'peak':np.nan, 
                       'amplitude_1':np.nan,
                       'amplitude_2':np.nan,
                       'very_broad_frac':np.nan,
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
                       'oiii_5007_05_percentile':np.nan,
                       'oiii_5007_025_percentile':np.nan,
                       'oiii_5007_01_percentile':np.nan,
                       'redchi':np.nan,
                       'snr':np.nan,
                       'dv':np.nan, 
                       'monolum':np.nan,
                       'fe_ew':np.nan,
                       'shape':np.nan}

            return fit_out 

        mod, pars = make_model_gh(xscale=xscale,
                                  gh_order=gh_order,
                                  fmin=wave2doppler(fitting_region[0], w0).value,
                                  fmax=wave2doppler(fitting_region[1], w0).value) 
   

    elif fit_model == 'MultiGauss':

        mod, pars = make_model_multigauss(x,
                                          y,
                                          nGaussians=nGaussians,
                                          fix_broad_peaks=fix_broad_peaks)
                

    elif fit_model == 'Hb':

        mod, pars = make_model_hb(xscale=xscale,
                                  w0=w0,
                                  nGaussians=nGaussians, 
                                  hb_narrow=hb_narrow,
                                  fix_broad_peaks=fix_broad_peaks,
                                  oiii_broad_off=oiii_broad_off,
                                  oiii_template=oiii_template)

    elif fit_model == 'Ha':

        mod, pars = make_model_ha(xscale=xscale,
                                  w0=w0,
                                  nGaussians=nGaussians,
                                  fix_broad_peaks=fix_broad_peaks,
                                  ha_narrow_fwhm=ha_narrow_fwhm,
                                  ha_narrow_voff=ha_narrow_voff,
                                  ha_narrow_vary=ha_narrow_vary,
                                  x=x)


    elif fit_model == 'OIII':

        mod, pars = make_model_oiii(xscale=xscale,
                                    w0=w0,
                                    nGaussians=nGaussians, 
                                    hb_narrow=hb_narrow,
                                    fix_broad_peaks=fix_broad_peaks,
                                    oiii_broad_off=oiii_broad_off,
                                    oiii_template=oiii_template,
                                    fix_oiii_peak_ratio=fix_oiii_peak_ratio)        
    

    elif fit_model == 'siiv':

        mod, pars = make_model_siiv(w0=w0)


    if load_fit:

        file = os.path.join(save_dir, 'out.pickle')
        parfile = open(file, 'rb')
        out = pickle.load(parfile)
        parfile.close()

    else: 

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
                           options={'maxiter':1e4, 'maxfev':1e4} 
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

            # remove expressions from Parameters instance - https://groups.google.com/forum/#!topic/lmfit-py/6tCcTNe307I
            params_dump = deepcopy(out.params)
            for v in params_dump:
                params_dump[v].expr = None
                
            param_file = os.path.join(save_dir, 'my_params.txt')
            parfile = open(param_file, 'w')
            params_dump.dump(parfile)
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

            out_file = os.path.join(save_dir, 'out.pickle')
            parfile = open(out_file, 'wb')
            pickle.dump(out, parfile, -1)
            parfile.close()

            fittxt = ''    
            fittxt += strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n \n'
            fittxt += plot_title + '\n \n'
            fittxt += r'Converged with chi-squared = ' + str(out.chisqr) + ', DOF = ' + str(out.nfree) + '\n \n'
            fittxt += fit_report(out.params)
    
            with open(os.path.join(save_dir, 'fit.txt'), 'w') as f:
                f.write(fittxt) 

    # Calculate stats 

    if line_region.unit == u.AA:
        line_region = wave2doppler(line_region, w0)

    if fit_model == 'OIII':

        fit_out = get_stats_oiii(out,
                                 mod=mod,
                                 xscale=xscale, 
                                 w0=w0,
                                 plot_title=plot_title,
                                 bkgdmod=bkgdmod,
                                 bkgdpars_k=bkgdpars_k,
                                 sp_fe=sp_fe,
                                 subtract_fe=subtract_fe,
                                 z=z,
                                 spec_norm=spec_norm,
                                 varray=np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value/xscale),
                                 farray=ma.getdata(y[~ma.getmaskarray(y)]),
                                 sarray=ma.getdata(er[~ma.getmaskarray(er)]),
                                 nGaussians=nGaussians,
                                 oiii_broad_off=oiii_broad_off,
                                 hb_narrow=hb_narrow)


    elif fit_model == 'Hb':

        fit_out = get_stats_hb(out=out,
                               line_region=line_region,
                               xscale=xscale,
                               w0=w0,
                               bkgdmod=bkgdmod,
                               bkgdpars_k=bkgdpars_k,
                               subtract_fe=subtract_fe,
                               spec_norm=spec_norm,
                               sp_fe=sp_fe,
                               z=z,
                               mod=mod,
                               nGaussians=nGaussians,
                               plot_title=plot_title,
                               hb_narrow=hb_narrow)



    elif fit_model == 'Ha':

        fit_out = get_stats_ha(out=out,
                               line_region=line_region,
                               xscale=xscale,
                               w0=w0,
                               bkgdmod=bkgdmod,
                               bkgdpars_k=bkgdpars_k,
                               subtract_fe=subtract_fe,
                               spec_norm=spec_norm,
                               sp_fe=sp_fe,
                               z=z,
                               mod=mod,
                               varray=np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value/xscale),
                               farray=ma.getdata(y[~ma.getmaskarray(y)]),
                               sarray=ma.getdata(er[~ma.getmaskarray(er)]),
                               nGaussians=nGaussians,
                               plot_title=plot_title,
                               emission_line=emission_line)

    elif (fit_model == 'GaussHermite') | (fit_model == 'siiv'):

        fit_out = get_stats_gh(out=out,
                               line_region=line_region,
                               xscale=xscale,
                               w0=w0,
                               bkgdmod=bkgdmod,
                               bkgdpars_k=bkgdpars_k,
                               subtract_fe=subtract_fe,
                               spec_norm=spec_norm,
                               sp_fe=sp_fe,
                               z=z,
                               mod=mod,
                               n_samples=n_samples,
                               verbose=verbose,
                               plot_title=plot_title)

    elif fit_model == 'MultiGauss':

        fit_out = get_stats_multigauss(out=out,
                                       line_region=line_region,
                                       xscale=xscale,
                                       w0=w0,
                                       bkgdmod=bkgdmod,
                                       bkgdpars_k=bkgdpars_k,
                                       subtract_fe=subtract_fe,
                                       sp_fe=sp_fe,
                                       spec_norm=spec_norm,
                                       z=z,
                                       mod=mod,
                                       varray=np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value/xscale),
                                       farray=ma.getdata(y[~ma.getmaskarray(y)]),
                                       sarray=ma.getdata(er[~ma.getmaskarray(er)]),
                                       nGaussians=nGaussians,
                                       plot_title=plot_title,
                                       emission_line=emission_line)

    fit_out['redchi'] = out.redchi
    fit_out['snr'] = snr_k
    fit_out['dv'] = spec_dv
    fit_out['monolum'] = np.log10(mono_lum_k)
    fit_out['fe_ew'] = eqw_fe_k

    
    if n_samples == 1:

        if verbose: 

            print '\n'
            print 'Monochomatic luminosity at {0} = {1:.3f}'.format(mono_lum_wav, np.log10(mono_lum_k)) 
            print '\n'
    
            if emission_line == 'Ha':
    
                """
                Nicely print out important fitting info
                """     
                
                if fit_model == 'Ha':
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
    
                print fit_out['name'] + ',',\
                      format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
                      format(fit_out['fwhm_1'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_1']) else ',',\
                      format(fit_out['fwhm_2'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_2']) else ',',\
                      format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
                      format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
                      format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
                      format(fit_out['cen_1'], '.2f') + ',' if ~np.isnan(fit_out['cen_1']) else ',',\
                      format(fit_out['cen_2'], '.2f') + ',' if ~np.isnan(fit_out['cen_2']) else ',',\
                      format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
                      format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
                      format(fit_out['amplitude_1'], '.2f') + ',' if ~np.isnan(fit_out['amplitude_1']) else ',',\
                      format(fit_out['amplitude_2'], '.2f') + ',' if ~np.isnan(fit_out['amplitude_2']) else ',',\
                      format(fit_out['very_broad_frac'], '.2f') + ',' if ~np.isnan(fit_out['very_broad_frac']) else ',',\
                      format(fit_out['narrow_fwhm'], '.2f') + ',' if ~np.isnan(fit_out['narrow_fwhm']) else ',',\
                      format(fit_out['narrow_lum'], '.2f') + ',' if ~np.isnan(fit_out['narrow_lum']) else ',',\
                      format(fit_out['narrow_voff'], '.2f') + ',' if ~np.isnan(fit_out['narrow_voff']) else ',',\
                      format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''
    
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
                       'Reduced chi-squared: {0:.2f} \n'.format(fit_out['redchi']),\
                       'S/N: {0:.2f} \n'.format(fit_out['snr']), \
                       'dv: {0:.1f} km/s \n'.format(fit_out['dv'].value), \
                       'Monochomatic luminosity: {0:.2f} erg/s \n'.format(fit_out['monolum'])
    


            if emission_line == 'Hb':
    
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
        
    
       
    
                print fit_out['name'] + ',',\
                      format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
                      format(fit_out['fwhm_1'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_1']) else ',',\
                      format(fit_out['fwhm_2'], '.2f') + ',' if ~np.isnan(fit_out['fwhm_2']) else ',',\
                      format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
                      format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
                      format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
                      format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
                      format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
                      format(fit_out['amplitude_1'], '.2f') + ',' if ~np.isnan(fit_out['amplitude_1']) else ',',\
                      format(fit_out['amplitude_2'], '.2f') + ',' if ~np.isnan(fit_out['amplitude_2']) else ',',\
                      format(fit_out['very_broad_frac'], '.2f') + ',' if ~np.isnan(fit_out['very_broad_frac']) else ',',\
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
                      format(fit_out['oiii_5007_05_percentile'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_05_percentile']) else ',',\
                      format(fit_out['oiii_5007_025_percentile'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_025_percentile']) else ',',\
                      format(fit_out['oiii_5007_01_percentile'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_01_percentile']) else ',',\
                      format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''
               
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
                      'OIII5007 blueshift: {0:.2f} km/s \n'.format(fit_out['oiii_5007_05_percentile']),\
                      'OIII5007 quartile: {0:.2f} km/s \n'.format(fit_out['oiii_5007_025_percentile']),\
                      'OIII5007 0.1 percentile: {0:.2f} km/s \n'.format(fit_out['oiii_5007_01_percentile']),\
                      'Reduced chi-squared: {0:.2f} \n'.format(fit_out['redchi']),\
                      'S/N: {0:.2f} \n'.format(fit_out['snr']), \
                      'dv: {0:.1f} km/s \n'.format(fit_out['dv'].value), \
                      'Monochomatic luminosity: {0:.2f} erg/s \n'.format(fit_out['monolum'])
    
                
    
            if emission_line == 'CIV':
    
                print fit_out['name'] + ',',\
                      format(fit_out['fwhm'], '.2f') + ',' if ~np.isnan(fit_out['fwhm']) else ',',\
                      format(fit_out['sigma'], '.2f') + ',' if ~np.isnan(fit_out['sigma']) else ',',\
                      format(fit_out['median'], '.2f') + ',' if ~np.isnan(fit_out['median']) else ',',\
                      format(fit_out['cen'], '.2f') + ',' if ~np.isnan(fit_out['cen']) else ',',\
                      format(fit_out['eqw'], '.2f') + ',' if ~np.isnan(fit_out['eqw']) else ',',\
                      format(fit_out['broad_lum'], '.2f') + ',' if ~np.isnan(fit_out['broad_lum']) else ',',\
                      format(fit_out['peak'], '.2e') + ',' if ~np.isnan(fit_out['peak']) else ',',\
                      format(fit_out['redchi'], '.2f') if ~np.isnan(fit_out['redchi']) else ''

            if emission_line == 'siiv':
    
                print fit_out['name'] + ',',\
                      format(fit_out['peak'], '.2e') + ',' if ~np.isnan(fit_out['peak']) else ''
            
            if emission_line == 'OIII':
                
                print 'OIII5007 EQW: {0:.2f} A \n'.format(fit_out['oiii_5007_eqw']),\

                print  colored(fit_out['name'], 'red') + ',',\
                       format(fit_out['oiii_5007_v5'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_v5']) else ',',\
                       format(fit_out['oiii_5007_v10'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_v10']) else ',',\
                       format(fit_out['oiii_5007_v25'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_v25']) else ',',\
                       format(fit_out['oiii_5007_v50'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_v50']) else ',',\
                       format(fit_out['oiii_5007_v75'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_v75']) else ',',\
                       format(fit_out['oiii_5007_v90'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_v90']) else ',',\
                       format(fit_out['oiii_5007_v95'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_v95']) else ',',\
                       format(fit_out['oiii_5007_eqw'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_eqw']) else ',',\
                       format(fit_out['oiii_5007_lum'], '.2f') + ',' if ~np.isnan(fit_out['oiii_5007_lum']) else ',',\
                       format(fit_out['oiii_peak_ratio'], '.2f') if ~np.isnan(fit_out['oiii_peak_ratio']) else ''

                print '\n'
                print colored('oiii_5007_n_fwhm', 'green'),\
                      out.params['oiii_5007_n_sigma'].value*2.35,\
                      colored(out.params['oiii_5007_n_sigma'].min*2.35, 'yellow'),\
                      colored(out.params['oiii_5007_n_sigma'].max*2.35, 'red')
                print colored('oiii_5007_b_fwhm', 'green'),\
                      out.params['oiii_5007_b_sigma'].value*2.35,\
                      colored(out.params['oiii_5007_b_sigma'].min*2.35, 'yellow'),\
                      colored(out.params['oiii_5007_b_sigma'].max*2.35, 'red')
                print colored('oiii_5007_n_center', 'green'),\
                      out.params['oiii_5007_n_center'].value - wave2doppler(5008.239*u.AA, w0).value,\
                      colored(out.params['oiii_5007_n_center'].min - wave2doppler(5008.239*u.AA, w0).value, 'yellow'),\
                      colored(out.params['oiii_5007_n_center'].max - wave2doppler(5008.239*u.AA, w0).value, 'red') 
                print colored('oiii_5007_b_center_delta', 'green'),\
                      out.params['oiii_5007_b_center_delta'].value,\
                      colored(out.params['oiii_5007_b_center_delta'].min, 'yellow'),\
                      colored(out.params['oiii_5007_b_center_delta'].max, 'red')

                print 'Reduced chi-squared: {}'.format(out.redchi) 

                print 'OIII peak ratio: {}'.format(out.params['oiii_peak_ratio'].value)

        if plot:
    
            if subtract_fe is True:
                flux_plot = flux - resid(params=bkgdpars_k, 
                                         x=wav.value, 
                                         model=bkgdmod,
                                         sp_fe=sp_fe)
            
            if subtract_fe is False:
                flux_plot = flux - resid(params=bkgdpars_k, 
                                         x=wav.value, 
                                         model=bkgdmod)
    
            if plot_region.unit == (u.km/u.s):
                plot_region = doppler2wave(plot_region, w0)

            plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])

            # if not flux in plotting region. only really relevant if plotting region < fitting region
            # which is only true for the OIII fit 

            if plot_region_inds.any():


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
                          verbose=verbose,
                          emission_line=emission_line)
        

    else:

        print plot_title, k, out.nfev 
    
    return fit_out 

        
def fit5(obj,
         comps=None,
         comps_wav=None,
         weights=None,
         fitting_method='nelder',
         verbose=False,
         plot_title='',
         mfica_n_weights=10,
         save_dir=None,
         plot=False,
         n_samples=1,
         plot_savefig=None,
         z=0.0):

    """
    Fit MFICA components
    """

    k = obj[0]
    x = obj[1]
    y = obj[2]
    er = obj[3]


    out = minimize(mfica_resid,
                   weights,
                   kws={'comps':comps,
                        'comps_wav':comps_wav,
                        'x':np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value),  
                        'data':ma.getdata(y[~ma.getmaskarray(y)]),
                        'sigma':ma.getdata(er[~ma.getmaskarray(er)])},
                   method=fitting_method,
                   options={'maxiter':1e4, 'maxfev':1e4} 
                   ) 


    
    chis = []
    shifts = np.arange(-10, 11, dtype=float) / 10.0

    for shift in shifts: 

        weights['w1'].value = out.params['w1'].value 
        weights['w2'].value = out.params['w2'].value 
        weights['w3'].value = out.params['w3'].value 
        weights['w4'].value = out.params['w4'].value 
        weights['w5'].value = out.params['w5'].value 
        weights['w6'].value = out.params['w6'].value 
        weights['w7'].value = out.params['w7'].value 
        weights['w8'].value = out.params['w8'].value 
        weights['w9'].value = out.params['w9'].value 
        weights['w10'].value = out.params['w10'].value 
        weights['shift'].value = shift 

        xi = np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value)
        yi = ma.getdata(y[~ma.getmaskarray(y)])
        dyi = ma.getdata(er[~ma.getmaskarray(er)])

        # just focus on area around oiii peak 

        region = (xi > 4980) & (xi < 5035)

        xi = xi[region]
        yi = yi[region]
        dyi = dyi[region]

        val = mfica_resid(weights=weights, 
                          comps=comps,
                          comps_wav=comps_wav,
                          x=xi, 
                          data=yi, 
                          sigma=dyi)

        chis.append(val.sum())

        # fit_out = {'name':plot_title,
        #             'w1': weights['w1'].value,
        #             'w2': weights['w2'].value,
        #             'w3': weights['w3'].value,
        #             'w4': weights['w4'].value,
        #             'w5': weights['w5'].value,
        #             'w6': weights['w6'].value,
        #             'w7': weights['w7'].value,
        #             'w8': weights['w8'].value,
        #             'w9': weights['w9'].value,
        #             'w10': weights['w10'].value,
        #             'shift':weights['shift'].value}

        # plot_mfica_fit(mfica_n_weights=mfica_n_weights,
        #                xi=x,
        #                yi=y,
        #                dyi=er,
        #                out=fit_out,
        #                comps=comps,
        #                comps_wav=comps_wav,
        #                verbose=verbose)




    # now minimize again with new starting point for shift

    weights['shift'].value = shifts[np.argmin(np.array(chis))]

    # fig, ax = plt.subplots() 
    # ax.plot(shifts, chis)
    # plt.show() 

    out = minimize(mfica_resid,
                   weights,
                   kws={'comps':comps,
                        'comps_wav':comps_wav,
                        'x':np.asarray(ma.getdata(x[~ma.getmaskarray(x)]).value),  
                        'data':ma.getdata(y[~ma.getmaskarray(y)]),
                        'sigma':ma.getdata(er[~ma.getmaskarray(er)])},
                   method=fitting_method,
                   options={'maxiter':1e4, 'maxfev':1e4} 
                   ) 


    fit_out = {'name':plot_title,
               'w1': out.params['w1'].value,
               'w2': out.params['w2'].value,
               'w3': out.params['w3'].value,
               'w4': out.params['w4'].value,
               'w5': out.params['w5'].value,
               'w6': out.params['w6'].value,
               'w7': out.params['w7'].value,
               'w8': out.params['w8'].value,
               'w9': out.params['w9'].value,
               'w10': out.params['w10'].value,
               'shift':out.params['shift'].value}
    
    fit_out = dict(oiii_reconstruction(fit_out).items() + fit_out.items())

    # seems weird that this depends on where to measure it
    # maybe this is why people normally use a log scale? 

    fit_out['z_ica'] = ((5008.239 + (out.params['shift'].value * 10.0)) * (1.0 + z)) / 5008.239 - 1.0 

   
    if verbose: 

        print out.message 

        for key, value in fit_out.iteritems():

            
            if key == 'name':
                pass
            else:
                print key + ' {0:.5f}'.format(value) 

        print colored(out.redchi, 'red')

        print  fit_out['name'] + ',',\
               format(fit_out['w1'], '.5f') + ',' if ~np.isnan(fit_out['w1']) else ',',\
               format(fit_out['w2'], '.5f') + ',' if ~np.isnan(fit_out['w2']) else ',',\
               format(fit_out['w3'], '.5f') + ',' if ~np.isnan(fit_out['w3']) else ',',\
               format(fit_out['w4'], '.5f') + ',' if ~np.isnan(fit_out['w4']) else ',',\
               format(fit_out['w5'], '.5f') + ',' if ~np.isnan(fit_out['w5']) else ',',\
               format(fit_out['w6'], '.5f') + ',' if ~np.isnan(fit_out['w6']) else ',',\
               format(fit_out['w7'], '.5f') + ',' if ~np.isnan(fit_out['w7']) else ',',\
               format(fit_out['w8'], '.5f') + ',' if ~np.isnan(fit_out['w8']) else ',',\
               format(fit_out['w9'], '.5f') + ',' if ~np.isnan(fit_out['w9']) else ',',\
               format(fit_out['w10'], '.5f') + ',' if ~np.isnan(fit_out['w10']) else ',',\
               format(fit_out['shift'], '.1f') + ',' if ~np.isnan(fit_out['shift']) else ',',\
               format(fit_out['mfica_oiii_v05'], '.1f') + ',' if ~np.isnan(fit_out['mfica_oiii_v05']) else ',',\
               format(fit_out['mfica_oiii_v10'], '.1f') + ',' if ~np.isnan(fit_out['mfica_oiii_v10']) else ',',\
               format(fit_out['mfica_oiii_v25'], '.1f') + ',' if ~np.isnan(fit_out['mfica_oiii_v25']) else ',',\
               format(fit_out['mfica_oiii_v50'], '.1f') + ',' if ~np.isnan(fit_out['mfica_oiii_v50']) else ',',\
               format(fit_out['mfica_oiii_v75'], '.1f') + ',' if ~np.isnan(fit_out['mfica_oiii_v75']) else ',',\
               format(fit_out['mfica_oiii_v90'], '.1f') + ',' if ~np.isnan(fit_out['mfica_oiii_v90']) else ',',\
               format(fit_out['mfica_oiii_v95'], '.1f') + ',' if ~np.isnan(fit_out['mfica_oiii_v95']) else ',',\
               format(fit_out['z_ica'], '.5f') if ~np.isnan(fit_out['z_ica']) else ''



    if save_dir is not None:

        fittxt = ''    
        fittxt += strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n \n'
        fittxt += plot_title + '\n \n'
        fittxt += r'Converged with chi-squared = ' + str(out.chisqr) + ', DOF = ' + str(out.nfree) + '\n \n'
        fittxt += fit_report(out.params)
    
        with open(os.path.join(save_dir, 'fit.txt'), 'w') as f:
            f.write(fittxt) 

    if plot & (n_samples == 1): 

        plot_mfica_fit(mfica_n_weights=mfica_n_weights,
                       xi=x,
                       yi=y,
                       dyi=er,
                       out=fit_out,
                       comps=comps,
                       comps_wav=comps_wav,
                       plot_savefig=plot_savefig,
                       verbose=verbose,
                       save_dir=save_dir) 

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
             oiii_broad_off=False,
             oiii_template=False,       
             mfica_n_weights=12, 
             z_IR = None,
             z_mfica = None,
             fix_oiii_peak_ratio = True,
             show_continuum_fit = True,
             load_fit = False,
             debug=False,
             append_errors=False):


    """
    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    In previos version I did not include the errors in the background fit, which is why
    the results are slightly different 

    mfica_z if not none overwrites z_IR 

    """


    if n_samples > 1: 

        if save_dir is not None:
    
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if emission_line == 'Ha':

                columns = ['fwhm', 
                           'fwhm_1', 
                           'fwhm_2', 
                           'sigma', 
                           'median', 
                           'cen', 
                           'eqw', 
                           'broad_lum', 
                           'peak', 
                           'narrow_fwhm', 
                           'narrow_lum', 
                           'narrow_voff', 
                           'peak_z',
                           'broad_offset',
                           'redchi', 
                           'snr', 
                           'monolum', 
                           'fe_ew'] 
                
            if emission_line == 'Hb':

                columns = ['fwhm', 
                           'fwhm_1', 
                           'fwhm_2', 
                           'sigma', 
                           'median', 
                           'cen', 
                           'eqw', 
                           'broad_lum', 
                           'peak', 
                           'narrow_fwhm', 
                           'narrow_lum', 
                           'narrow_voff', 
                           'oiii_5007_eqw', 
                           'oiii_5007_lum', 
                           'oiii_5007_n_lum', 
                           'oiii_5007_b_lum', 
                           'oiii_fwhm', 
                           'oiii_n_fwhm', 
                           'oiii_b_fwhm', 
                           'oiii_5007_b_voff', 
                           'oiii_5007_05_percentile', 
                           'oiii_5007_025_percentile', 
                           'oiii_5007_01_percentile', 
                           'redchi', 
                           'snr', 
                           'monolum', 
                           'fe_ew']
                
            if emission_line == 'CIV':

                columns = ['fwhm', 
                           'sigma', 
                           'median', 
                           'cen', 
                           'eqw', 
                           'broad_lum', 
                           'peak', 
                           'redchi', 
                           'snr', 
                           'monolum',
                           'shape']

            if pseudo_continuum_fit:

                columns = ['monolum', 
                           'fe_ew']

            if emission_line == 'siiv':

                columns = ['fwhm', 
                           'sigma', 
                           'median', 
                           'cen', 
                           'eqw', 
                           'broad_lum', 
                           'peak', 
                           'redchi', 
                           'snr']

            if emission_line == 'MFICA':
                
                columns = ['w1',
                           'w2',
                           'w3',
                           'w4',
                           'w5',
                           'w6',
                           'w7',
                           'w8',
                           'w9',
                           'w10',
                           'shift',
                           'mfica_oiii_v05',
                           'mfica_oiii_v10',
                           'mfica_oiii_v25',
                           'mfica_oiii_v50',
                           'mfica_oiii_v75',
                           'mfica_oiii_v90',
                           'mfica_oiii_v95',
                           'z_ica']

            if emission_line == 'OIII':

                columns = ['oiii_5007_v5', 
                           'oiii_5007_v10', 
                           'oiii_5007_v25', 
                           'oiii_5007_v50', 
                           'oiii_5007_v75', 
                           'oiii_5007_v90', 
                           'oiii_5007_v95',
                           'oiii_5007_eqw', 
                           'oiii_5007_lum',
                           'oiii_5007_snr',
                           'oiii_5007_fwhm',
                           'oiii_peak_ratio',
                           'oiii_5007_narrow_z',
                           'oiii_5007_full_peak_z',
                           'oiii_5007_full_peak_vel',
                           'hb_snr', 
                           'hb_z',
                           'redchi']
                              



            index = np.arange(n_samples)

            df_out = pd.DataFrame(index=index, columns=columns)     


    home_dir = expanduser("~")

    # probably not ideal - just a way to get a more reliable z for the component fit
    # without having to change the masterlist file, run make_html etc. 
    # just use for quick solution and then change masterfile

    if (emission_line == 'OIII') & (z_IR is not None):

        z = z_IR  

    if (emission_line == 'MFICA') & (z_mfica is not None):

        z = z_mfica


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

    wav, flux, err = rebin_simple(wav, flux, err, n_rebin)

    n_elements = len(wav)
    n_elements_norebin = len(wav_norebin)


    # index of the region we want to fit
    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    if len(wav[(wav > fitting_region[0]) & (wav < fitting_region[1])]) == 0:
       
        if verbose:
            print 'No flux in fitting region'

        if n_samples == 1:
       
            if emission_line == 'MFICA':

                fit_out = {'name':plot_title,
                           'w1':np.nan, 
                           'w2':np.nan, 
                           'w3':np.nan, 
                           'w4':np.nan, 
                           'w5':np.nan, 
                           'w6':np.nan, 
                           'w7':np.nan, 
                           'w8':np.nan, 
                           'w9':np.nan, 
                           'w10':np.nan, 
                           'mfica_oiii_v05':np.nan,
                           'mfica_oiii_v10':np.nan,
                           'mfica_oiii_v25':np.nan,
                           'mfica_oiii_v50':np.nan,
                           'mfica_oiii_v75':np.nan,
                           'mfica_oiii_v90':np.nan,
                           'mfica_oiii_v95':np.nan,
                           'shift':np.nan,
                           'z_ica':np.nan}

            elif emission_line == 'OIII':

                fit_out = {'name':plot_title,
                           'oiii_5007_v5':np.nan,
                           'oiii_5007_v10':np.nan,
                           'oiii_5007_v25':np.nan,
                           'oiii_5007_v50':np.nan,
                           'oiii_5007_v75':np.nan,
                           'oiii_5007_v90':np.nan,
                           'oiii_5007_v95':np.nan,
                           'oiii_5007_eqw':np.nan,
                           'oiii_5007_lum':np.nan,
                           'oiii_5007_snr':np.nan,
                           'oiii_5007_fwhm':np.nan,
                           'oiii_peak_ratio':np.nan,
                           'oiii_5007_narrow_z':np.nan,
                           'oiii_5007_full_peak_z':np.nan,
                           'oiii_5007_full_peak_vel':np.nan,
                           'hb_snr':np.nan,
                           'snr':np.nan, # in continuum 
                           'hb_z':np.nan,
                           'redchi':np.nan}

            else: 

                fit_out = {'name':plot_title, 
                           'fwhm':np.nan,
                           'fwhm_1':np.nan,
                           'fwhm_2':np.nan,
                           'sigma':np.nan,
                           'median':np.nan,
                           'cen':np.nan,
                           'cen_1':np.nan,
                           'cen_2':np.nan,
                           'eqw':np.nan,
                           'broad_lum':np.nan,
                           'peak':np.nan, 
                           'peak_z':np.nan, 
                           'broad_offset':np.nan,
                           'snr_line':np.nan,
                           'amplitude_1':np.nan,
                           'amplitude_2':np.nan,
                           'very_broad_frac':np.nan,
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
                           'oiii_5007_05_percentile':np.nan,
                           'oiii_5007_025_percentile':np.nan,
                           'oiii_5007_01_percentile':np.nan,
                           'redchi':np.nan,
                           'snr':np.nan,
                           'dv':np.nan, 
                           'monolum':np.nan,
                           'fe_ew':np.nan,
                           'shape':np.nan}
            
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
                          verbose=verbose,
                          emission_line=emission_line)
    
            
            return fit_out 

        else:

            if save_dir is not None:
                if append_errors:
                    with open(os.path.join(save_dir, 'fit_errors.txt'), 'a') as f: 
                        df_out.to_csv(f, header=False, index=False)
                else:
                    df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 

            return None 


    # might have trouble if err is negative, but do whatever and mask out before fit. 

    if n_samples > 1: 

        flux_array = np.asarray(np.random.normal(flux,
                                                 err,
                                                 size=(n_samples, n_elements)),
                                dtype=np.float32)



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

    mask = (ma.getdata(wav_array).value < plot_region[0].value) |\
           (ma.getdata(wav_array).value > plot_region[1].value) 

    flux_array_plot = ma.masked_where(mask, flux_array)
    wav_array_plot = ma.masked_where(mask, wav_array)
    err_array_plot = ma.masked_where(mask, err_array)
    vdat_array_plot = ma.masked_where(mask, vdat_array)   
    
    # get rid of nans / infs / negative errors 


    mask = (np.isnan(flux_array)) | (err_array < 0.0) 
    mask_norebin = (np.isnan(flux_array_norebin)) | (err_array_norebin < 0.0) 

    # print flux_array

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

            if maskout.unit == (u.km/u.s):  
                             
                mask = (ma.getdata(vdat_array).value > item[0].value) & (ma.getdata(vdat_array).value < item[1].value) 
    
            elif maskout.unit == u.AA: 
  
                mask = (ma.getdata(wav_array).value > item[0].value) & (ma.getdata(wav_array).value < item[1].value)  

            flux_array = ma.masked_where(mask, flux_array)
            wav_array = ma.masked_where(mask, wav_array)
            err_array = ma.masked_where(mask, err_array)
            vdat_array = ma.masked_where(mask, vdat_array)   

        for item in maskout:

            if maskout.unit == (u.km/u.s):  
                             
                mask_norebin = (ma.getdata(vdat_array_norebin).value > item[0].value) & (ma.getdata(vdat_array_norebin).value < item[1].value) 
    
            elif maskout.unit == u.AA: 
            
                mask_norebin = (ma.getdata(wav_array_norebin).value > item[0].value) & (ma.getdata(wav_array_norebin).value < item[1].value) 

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

    if emission_line != 'MFICA':

        if len(ma.getdata(wav_array_blue[0, ~ma.getmaskarray(wav_array_blue[0, :])]).value) + len(ma.getdata(wav_array_red[0, ~ma.getmaskarray(wav_array_red[0, :])]).value) == 0:
    
            if verbose:
                print 'No flux in continuum region'
        
            if n_samples == 1:
            
                fit_out = {'name':plot_title, 
                           'fwhm':np.nan,
                           'fwhm_1':np.nan,
                           'fwhm_2':np.nan,
                           'sigma':np.nan,
                           'median':np.nan,
                           'cen':np.nan,
                           'cen_1':np.nan,
                           'cen_2':np.nan,
                           'eqw':np.nan,
                           'broad_lum':np.nan,
                           'peak':np.nan, 
                           'amplitude_1':np.nan,
                           'amplitude_2':np.nan,
                           'very_broad_frac':np.nan,
                           'narrow_fwhm':np.nan,
                           'narrow_lum':np.nan,
                           'narrow_voff':np.nan, 
                           'peak_z':np.nan,
                           'oiii_5007_eqw':np.nan,
                           'oiii_5007_lum':np.nan,
                           'oiii_5007_n_lum':np.nan,
                           'oiii_5007_b_lum':np.nan,
                           'oiii_fwhm':np.nan,
                           'oiii_n_fwhm':np.nan,
                           'oiii_b_fwhm':np.nan,
                           'oiii_5007_b_voff':np.nan,
                           'oiii_5007_05_percentile':np.nan,
                           'oiii_5007_025_percentile':np.nan,
                           'oiii_5007_01_percentile':np.nan,
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
                              verbose=verbose,
                              emission_line=emission_line)
            
                
                return fit_out 
        
            else:
    

                if save_dir is not None:
                    
                    if append_errors:
                        with open(os.path.join(save_dir, 'fit_errors.txt'), 'a') as f: 
                            df_out.to_csv(f, header=False, index=False)
                    else:
                        df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 
        
                return None 

        """
        Calculate S/N ratio per pixel 
        Do this for the un-rebinned spectra  
        Not perfect - there's also iron in the continuum windows
        """
       
       
        mask_blue = (err_array_blue_norebin < 0.0) | np.isnan(flux_array_blue_norebin)
        mask_red = (err_array_red_norebin < 0.0) | np.isnan(flux_array_red_norebin)
        
        wa = np.concatenate((ma.masked_where(mask_blue, wav_array_blue_norebin),
                             ma.masked_where(mask_red, wav_array_red_norebin)),
                            axis=1)

        fl = np.concatenate((ma.masked_where(mask_blue, flux_array_blue_norebin),
                             ma.masked_where(mask_red, flux_array_red_norebin)),
                            axis=1)

        er = np.concatenate((ma.masked_where(mask_blue, err_array_blue_norebin),
                             ma.masked_where(mask_red, err_array_red_norebin)),
                            axis=1)

        
        # snr_blue = ma.median(fl, axis=1) / ma.std(fl, axis=1)
        # snr_red = ma.median(fl, axis=1) / ma.std(fl, axis=1)

        snr = np.nanmedian(fl / er, axis=1)
        # print snr

        # 33.02 is the A per resolution element I measured from the Arc spectrum for Liris 
        if verbose:
            if n_samples == 1:
            # print 'S/N per resolution element in continuum: {0:.2f}'.format(np.median( np.sqrt(33.02 / np.diff(wa) ) * fl[:-1] / er[:-1] )) 
                print 'S/N per pixel in continuum: {0:.2f}'.format(snr[0]) 



        # fig, ax = plt.subplots() 
        # ax.hist(fl.flatten() / er.flatten(), bins=np.arange(-20, 50, 1))
        # plt.show() 
        
    #-------------------------------------------------------------------------------------


    if emission_line == 'MFICA': 

        # Approximately take out slope ---------------------------------------------------------------

        xdat_cont = ma.concatenate((wav_array_blue, wav_array_red), axis=1)
        ydat_cont = ma.concatenate((flux_array_blue, flux_array_red), axis=1)
        yerr_cont = ma.concatenate((err_array_blue, err_array_red), axis=1)

        fitobj = [] 

        for k in range(n_samples):

            fitobj.append([k, 
                           xdat_cont[k], 
                           ydat_cont[k], 
                           yerr_cont[k], 
                           flux_array_fit[k, :], 
                           err_array_fit[k, :],
                           wav_array_fit[k, :]])

        fit4_p = partial(fit4, 
                         n_samples = n_samples, 
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
                         show_continuum_fit = show_continuum_fit,
                         save_dir=save_dir)
    
        if parallel:
            
            p = Pool(cores)
            out = p.map(fit4_p, fitobj)
            p.close() 
            p.join()
    
        else:
            
            out = map(fit4_p, fitobj)


        for k, o in enumerate(out):
    
            flux_array_fit[k, :] = o[0]
            err_array_fit[k, :] = o[1] 


        # Now do fit -------------------------------------------------

        fitobj = []
    
        for k in range(n_samples):
            
            fitobj.append([k, 
                           wav_array_fit[k, :], 
                           flux_array_fit[k, :], 
                           err_array_fit[k, :]])

        comps_wav, comps, weights = make_model_mfica(mfica_n_weights=mfica_n_weights)

        fit5_p = partial(fit5,
                         comps=comps,
                         comps_wav=comps_wav,
                         weights=weights,
                         fitting_method=fitting_method,
                         verbose=verbose,
                         plot_title=plot_title,
                         mfica_n_weights=mfica_n_weights,
                         save_dir=save_dir,
                         plot=plot,
                         n_samples=n_samples,
                         plot_savefig=plot_savefig,
                         z=z)
        
        if parallel:
            
            p = Pool(cores)
            out = p.map(fit5_p, fitobj)
            p.close() 
            p.join()
        
        else:
            
            out = map(fit5_p, fitobj)
        
        
        if (save_dir is not None) & (n_samples > 1):

            
            for k in range(n_samples):
                for col in df_out:
                    df_out.loc[k, col] = out[k][col]
        
            if append_errors:
                
                with open(os.path.join(save_dir, 'fit_errors.txt'), 'a') as f: 
                    df_out.to_csv(f, header=False, index=False)
            
            else:
                df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 

             
        else:
        
            return out[0]





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
                             z = z,
                             save_dir=save_dir)
    
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
                         plot = plot,
                         save_dir = save_dir,
                         show_continuum_fit = show_continuum_fit,
                         pseudo_continuum_fit=pseudo_continuum_fit) 
    
    
    
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
        

            if pseudo_continuum_fit:

                """
                Only fitting PseudoContinuum 
                This won't work if calculating errors in fe fit 
                """
    
                fit_out = {'name':plot_title,
                           'monolum':np.log10(o[2].value),
                           'fe_ew':o[3]} 
    
                if n_samples == 1:       

                    return fit_out

                else:

                    print 'THIS NEEDS FIXING'
    

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
                     oiii_broad_off = oiii_broad_off,
                     oiii_template = oiii_template,
                     fix_oiii_peak_ratio = fix_oiii_peak_ratio,
                     load_fit = load_fit)
    
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
    
        if append_errors:
            with open(os.path.join(save_dir, 'fit_errors.txt'), 'a') as f: 
                df_out.to_csv(f, header=False, index=False)
        else:
            df_out.to_csv(os.path.join(save_dir, 'fit_errors.txt'), index=False) 

         
    else:
    
        return out[0]
     

    return None 

  

if __name__ == '__main__':

    # s = np.genfromtxt('/data/vault/phewett/LiamC/qso_hw10_template.dat')
    # wav = s[:, 0]
    # flux = s[:, 1]
    # dw = np.diff(wav)
    # err = np.repeat(0.01, len(flux))

    # t = np.genfromtxt('/home/lc585/Dropbox/IoA/nirspec/tables/shen2016_table2.txt') 
    
    t = np.genfromtxt('/home/lc585/Dropbox/IoA/nirspec/tables/hb_absorption_composite.dat', delimiter=',') 

    wav = t[:, 0]
    flux = t[:, 1]
    err = t[:, 2]
    dw = np.diff(wav)


    out = fit_line(wav,
                   dw,
                   flux,
                   err,
                   z=0.0,
                   w0=4862.721*u.AA,
                   continuum_region=[[4435,4700]*u.AA,[5100,5535]*u.AA],
                   fitting_region=[4700,5100]*u.AA,
                   plot_region=[4400,5550]*u.AA,
                   nGaussians=2,
                   nLorentzians=0,
                   line_region=[-10000,10000]*(u.km/u.s),
                   maskout=None,
                   verbose=True,
                   plot=True,
                   save_dir='/data/lc585/nearIR_spectra/linefits/Asymmetric_Hb_Comp',
                   plot_title='Composite',
                   plot_savefig='figure.png',
                   bkgd_median=False,
                   fitting_method='powell',
                   mask_negflux=False,
                   fit_model='OIII',
                   subtract_fe=True,
                   fe_FWHM = 4000.0*(u.km/u.s),
                   fe_FWHM_vary = True,
                   mono_lum_wav = 5100 * u.AA,
                   hb_narrow = False,  
                   n_rebin = 1 ,
                   reject_outliers = False, 
                   reject_width = 21,
                   reject_sigma = 3.0,
                   n_samples = 1,
                   emission_line='OIII',
                   parallel = False,
                   cores = 8,
                   fix_broad_peaks=True,
                   oiii_broad_off=True)