# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:18:10 2015

@author: lc585
"""
from __future__ import division

from astropy.table import Table
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
from get_wavelength import get_wavelength
import os 

def get_sdss_dr7_spec(name):

    """
    Given SDSS DR7 name will return wavelength, dw, flux and err.
    Uses Paul's better spectra

    """

    # Open master list
    f = open('/data/vault/phewett/DR7/ordn_splist_dr7_master.lis', 'r')

    SpecFiles, SDSSNames = [], []
    # Loop over lines and extract variables of interest
    for line in f:
        line = line.strip() # removes '\n' from eol
        columns = line.split()
        SpecFiles.append(columns[0])
        SDSSNames.append(columns[1])

    f.close()


    SpecFiles = np.array(SpecFiles)
    SDSSNames = np.array(SDSSNames)

    if name[:5] != 'SDSSJ':
        name = 'SDSSJ' + name

    i = np.where( SDSSNames == name )[0]

    if len(i) != 0:
        hdulist = fits.open( SpecFiles[i][0] )
        data = hdulist[0].data
        hdr = hdulist[0].header
        hdulist.close()

        wavelength, dw = get_wavelength(hdr)
        flux = data[0,:].flatten()
        err = data[1,:].flatten()


        return wavelength, dw, flux, err

    else:
        print 'No spectrum found'
        return None, None, None, None

def get_boss_dr12_spec(name):


    """
    Given DR12 name will return spectrum
    """

    t = Table.read('/data/vault/phewett/ICAtest/DR12exp/Spectra/boss_dr12_name_file.lis', 
                  format='ascii', 
                  names=['name', 'loc'])

    if name[:5] != 'SDSSJ':
        name = 'SDSSJ' + name

    hdulist = fits.open(t[t['name'] == name]['loc'].data[0]) 

    data = hdulist[1].data
    hdr = hdulist[0].header
    hdulist.close()

    wavelength = 10**np.array([j[1] for j in data ])
    dw = wavelength * (10**hdr['COEFF1'] - 1.0 )
    flux = np.array([j[0] for j in data ])
    err = np.sqrt( np.array([j[6] for j in data ])) # square root of sky


    return wavelength, dw, flux, err

   
def read_boss_dr12_spec(f):


    """
    Given DR12 name will return spectrum
    """

   

    hdulist = fits.open(f) 

    data = hdulist[1].data
    hdr = hdulist[0].header
    hdulist.close()

    wavelength = 10**np.array([j[1] for j in data ])
    dw = wavelength * (10**hdr['COEFF1'] - 1.0 )
    flux = np.array([j[0] for j in data ])
    err = np.sqrt( np.array([j[6] for j in data ])) # square root of sky


    return wavelength, dw, flux, err

if __name__ == '__main__':
    
    wavelength, dw, flux, err = get_boss_dr12_spec('000000.66+145828.8')  

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots() 
    ax.plot(wavelength, flux)
    plt.show()