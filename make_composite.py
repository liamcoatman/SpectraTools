# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:49:50 2015

@author: lc585

Make composite spectra

"""

import numpy as np 
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt 

def make_composite(wav_new,
                   wav_array, 
                   flux_array, 
                   z_array,
                   names=None,
                   verbose=True):

    # if verbose:
    #     plt.ion() 
    #     sns.set_style() 
    #     fig, ax = plt.subplots() 
    #     ax.set_xlim(4435.0, 5535.0)
    #     ax.set_ylim(0, 50)

    nspec = len(z_array)

    # Order by redshift 
    inds = np.argsort(z_array)

    wav_array = wav_array[inds]
    flux_array = flux_array[inds]
    z_array = z_array[inds]

    new_flux_array = np.zeros((nspec, len(wav_new)))
        
    for i in range(nspec):     
    
        # First shift to rest-frame  
        wav = wav_array[i] / (1.0 + z_array[i])    
    
        # Interpolate 
        # I don't think this conserves flux 
        f = interp1d(wav, flux_array[i], bounds_error=False, fill_value=np.nan) # combine with np.nanmedian()
        
        new_flux_array[i, :] = f(wav_new)
    
    
    # Arbitarily scale first spectrum 
    new_flux_array[0, :] = 1000.0 * new_flux_array[0, :] / np.nansum(new_flux_array[0, :])
    
    # if verbose:
    #     line1, = ax.plot(wav_new, new_flux_array[0, :], zorder=1)
    #     line2, = ax.plot(wav_new, new_flux_array[0, :], alpha=0.5, zorder=2)

    # Then scale in order of 
    # redshift to the average of the flux density in the common
    # wavelength region of the mean spectrum of all the lower
    # redshift spectra. 
    # using the method described by Vanden Berk et al. (2001)

    # for i in range(1, nspec):

    
    #     # Calculate overlap region 
    #     overlap = ~np.isnan(new_flux_array[0, :]) 
    #     j = 1
    #     while j <= i: 
    #         overlap = overlap & ~np.isnan(new_flux_array[j, :])
    #         j += 1 
    
    #     # Mean spectrum of all lower redshift spectrum  
    #     meanspec = np.nanmean(new_flux_array[:i, overlap], axis=0)
        
    #     # Scale to average flux density in mean spectrum 
    #     new_flux_array[i, :] = new_flux_array[i, :] * np.nanmedian(meanspec) / np.nanmedian(new_flux_array[i, :])
    

    #     if verbose:
    #         line1.set_ydata(np.nanmedian(new_flux_array[:i, :], axis=0))
    #         line2.set_ydata(new_flux_array[i, :])
    #         fig.canvas.draw()
    #         plt.pause(2)

    

    new_flux_array = new_flux_array / np.nanmedian(new_flux_array, axis=1)[:, np.newaxis]

   

    # Now caluclate uncertainties: 
    # dividing the 68% semi-interquantile range 
    # of the flux densities by the square root 
    # of the number of spectra contributing to each bin.
    
    flux = np.nanmedian(new_flux_array, axis=0)

    ns = np.sum(~np.isnan(new_flux_array), axis=0)
    
    q84, q16 = np.nanpercentile(new_flux_array, [84, 16], axis=0)
    
    err = 0.5*(q84 - q16) / np.sqrt(ns) 

    if verbose: 
        fig, ax = plt.subplots()
        for row in new_flux_array:
            ax.plot(wav_new, row, lw=1, color='grey', alpha=0.1)
        ax.plot(wav_new, flux)
        plt.show() 

    return wav_new, flux, err, ns 

