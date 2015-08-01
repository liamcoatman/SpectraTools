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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        dr12cat = Table.read('/data/lc585/SDSS/DR12Q.fits')

    i = np.where(dr12cat['SDSS_NAME'] == name)[0]

    if len(i) != 0:

        plate = dr12cat[i]['PLATE']
        fiber = dr12cat[i]['FIBERID']
        mjd = dr12cat[i]['MJD']

        url = 'http://api.sdss3.org/spectrum?plate={0}&fiber={1:04d}&mjd={2}'.format(plate, fiber, mjd)

        hdulist = fits.open(url)
        data = hdulist[1].data
        hdr = hdulist[0].header
        hdulist.close()

        wavelength = 10**np.array([j[1] for j in data ])
        dw = wavelength * (10**hdr['COEFF1'] - 1.0 )
        flux = np.array([j[0] for j in data ])
        err = np.sqrt( np.array([j[6] for j in data ])) # square root of sky

        return wavelength, dw, flux, err

    else:
        print 'No spectrum found'
        return None, None, None, None

def get_liris_spec(fname):

    """
    Give path to fits file and will return spectrum
    Works for my Liris IRAF spectra, might work for other fits
    """

    hdulist = fits.open( fname )
    hdr = hdulist[0].header
    data = hdulist[0].data
    hdulist.close()

    wavelength, dw = get_wavelength(hdr)
    flux = data[0,:,:].flatten()
    err = data[-1,:,:].flatten()

    return wavelength, dw, flux, err
