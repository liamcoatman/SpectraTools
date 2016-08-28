# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:13:56 2015

@author: lc585

Get monochromatic luminosity from spectra 

I'm just using a simple median, but check with Paul whether I should do power-law fit

"""
from __future__ import division
import astropy.units as u 
import numpy as np 
from astropy.cosmology import WMAP9 as cosmoWMAP
import math 
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set2_5
import numpy.ma as ma


def fit_line(wav,
             dw,
             flux,
             err,
             z=0.0,
             fitting_region=[6400,6800]*u.AA,
             plot_region=[6000,7000]*u.AA,
             verbose=False,
             plot=True,
             monolum_wav = 5100.0,
             maskout=None):

    """
    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    """

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])

    xdat = wav[fitting]
    ydat = flux[fitting] 
    yerr = err[fitting]
 
    if maskout is not None:

        mask = np.array([True] * len(xdat))
        for item in maskout:
            mask[(xdat > item[0]) & (xdat < item[1])] = False

        xdat = xdat[mask]
        ydat = ydat[mask]
        yerr = yerr[mask]

    f = np.median(ydat)

    f = f * (u.erg / u.cm / u.cm / u.s / u.AA)

    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)
    l = f * (1.0 + z) * 4.0 * math.pi * lumdist**2
    l = l * monolum_wav 

    print l, np.log10(l.value) 

    #######################################################################
    
    plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])

    xdat = wav[plot_region_inds]
    ydat = flux[plot_region_inds]
    yerr = err[plot_region_inds]

    plt.rc('axes', color_cycle=Set2_5.mpl_colors) 
    
    fig = plt.figure(figsize=(6,8))

    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    ax1.errorbar(xdat.value, ydat, yerr=yerr, linestyle='', alpha=0.5, color='grey')
    
    ax1.axvline(monolum_wav.value, linestyle='--', color='black')

    plot_region_ext = plot_region[1] - plot_region[0]
    plot_region_inds = (wav > plot_region[0] - (plot_region_ext / 2.0)) & (wav < plot_region[1] + (plot_region_ext / 2.0))

    xdat = wav[plot_region_inds]
    ydat = flux[plot_region_inds]
    yerr = err[plot_region_inds]

    ax2.errorbar(xdat.value, ydat, yerr=yerr, linestyle='', alpha=0.5, color='grey')

    ax2.axvline(monolum_wav.value, linestyle='--', color='black')

    ax1.scatter(monolum_wav.value, f.value)
    ax2.scatter(monolum_wav.value, f.value)

    xdat_masking = np.arange(xdat.min().value, xdat.max().value, 0.05)*(u.AA)
  
    mask = (xdat_masking.value < fitting_region.value[0]) | (xdat_masking.value > fitting_region.value[1])

    if maskout is not None:

        for item in maskout:
            xmin = item.value[0]
            xmax = item.value[1]
            mask = mask | ((xdat_masking.value > xmin) & (xdat_masking.value < xmax))

    xdat_masking = ma.array(xdat_masking.value)
    xdat_masking[mask] = ma.masked 

    for item in ma.extras.flatnotmasked_contiguous(xdat_masking):
        ax1.axvspan(xdat_masking[item].min(), xdat_masking[item].max(), alpha=0.4, color='moccasin')
        ax2.axvspan(xdat_masking[item].min(), xdat_masking[item].max(), alpha=0.4, color='moccasin')

    ax1.set_xlim(plot_region[0].value, plot_region[1].value)
    ax2.set_xlim(plot_region[0].value - (plot_region_ext.value / 2.0), plot_region[1].value + (plot_region_ext.value / 2.0))

    fig.tight_layout()

    plt.show()
    plt.close()


 


    return None 

