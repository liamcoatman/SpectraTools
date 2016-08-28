# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 12:21:58 2015

@author: lc585

Pyraf script to do telluric_correction

Need to be in telluric directory
"""

from __future__ import division
from pyraf import iraf
import os
from astropy.io import fits
from spectra import get_wavelength
import numpy as np
from scipy.interpolate import interp1d

def planck(wav, T):
    from scipy.constants import h, c, k
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a / ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity


def telluric_standard(vmag, kmag, spec_type):

    """
    The standard routine integrates over the appropriate bandpasses, divides
    by the exposure time, and outputs a single file containing an observation-by-observation
    listing of the observed counts within each bandpass along with the standard star fluxes.


    """

    print 'Making telluric standard...'

    if os.path.exists('std'):
        os.remove('std')
        print 'Removing file std'

    # This is wavelength calibrate telluric spectrum
    iraf.standard.setParam('input','dimcomb.ms.fits[1]')

    # Output flux file (used by SENSFUNC)
    iraf.standard.setParam('output','std')

    # Airmass - this should be set automatically
    # If I want I can calculate first using iraf routine, which updates the header.
    iraf.standard.setParam('airmass','')

    # Exposure time (seconds) - if not given should be calculated.
    # Remeber I average telluric frames, so this is just exposure time for one exposure.
    iraf.standard.setParam('exptime','')

    # If k band magnitude of star is known
    if float(kmag) > 0.1:
    	print 'using kmag'
    	# Magnitude of star
        iraf.standard.setParam('mag',kmag)
        # Magnitude type - if not the same as bandpass then a conversion is done.
        iraf.standard.setParam('magband','K')
    else:
        # Magnitude of star
        iraf.standard.setParam('mag',vmag)
        # Magnitude type - if not the same as bandpass then a conversion is done.
        iraf.standard.setParam('magband','V')

    # Effective temperature or spectral type
    iraf.standard.setParam('teff',spec_type)

    # Answer?
    iraf.standard.setParam('answer','yes')

    # If left as INDEF then bandpass calculated from calibration file.
    iraf.standard.setParam('bandwidth','INDEF')
    iraf.standard.setParam('bandsep','INDEF')

    # Absolute flux zero point used in magnitude to flux conversion
    iraf.standard.setParam('fnuzero','3.68E-20')

    # Extionction database - leave empty if not extinction corrected.
    iraf.standard.setParam('extinction','')

    # Directory containing calibration data
    iraf.standard.setParam('caldir','onedstds$blackbody/')

    # Observatory should be read from key word in header.
    # iraf.standard.setParam('observatory','')

    # Interact?
    iraf.standard.setParam('interact','no')

    # Star name in calibration list
    # This just contains the AB magnitude of a zero magnitude star in the given band
    iraf.standard.setParam('star_name','H')

    iraf.standard()

    # Star name in calibration list
    # This just contains the AB magnitude of a zero magnitude star in the given band
    iraf.standard.setParam('star_name','K')

    iraf.standard()

    print 'Done!'

    return None

def telluric_sensfunc():

    """
    sensfunc: determine sensitivity

    """
    iraf.sensfunc.unlearn()

    if os.path.exists('sens.fits'):
        os.remove('sens.fits')
        print 'Removing file sens.fits'

    # Input standard star data file (from STANDARD)
    iraf.sensfunc.setParam('standards','std')

    iraf.sensfunc.setParam('ignoreaps','yes')

    # Output root sensitivity function imagename
    iraf.sensfunc.setParam('sensitivity','sens')

    #(no|yes|NO|YES)
    iraf.sensfunc.setParam('answer','yes')

    # Fitting function
    iraf.sensfunc.setParam('function','spline3')

    # Order of fit
    iraf.sensfunc.setParam('order','9')

    # Determine sensitivity function interactively?
    iraf.sensfunc.setParam('interactive','yes')

    # Graphs to be displayed
    # s = sensitivity vs wavelength
    # r = residual sensitivty vs wavelength
    # i = flux calibrated spectrum vs wavelength
    iraf.sensfunc.setParam('graphs','si')

    iraf.sensfunc()

    return None


def sens_corr_standard():

    """
    Check this has worked by correcting stanard for wavelength-dependent sensitivity
    """

    if os.path.exists('cdimcomb.ms.fits'):
        os.remove('cdimcomb.ms.fits')
        print 'Removing file cdimcomb.ms.fits'

    iraf.calibrate.unlearn()

    # Input spectra to calibrate. Output should have 'c' prefixed.
    iraf.calibrate.setParam('input','dimcomb.ms.fits[1]')

    iraf.calibrate.setParam('output','cdimcomb.ms.fits')

    # Should be found automatically from header - check this for target spectrum.
    iraf.calibrate.setParam('airmass','')
    iraf.calibrate.setParam('exptime','')

    # Apply extinction correction? Not sure what this does
    iraf.calibrate.setParam('extinct','no')

	# Apply flux calibration?
    iraf.calibrate.setParam('flux','yes')

	# Extinction file
    iraf.calibrate.setParam('extinction','')

	# Ignore aperture numbers in flux calibration? Maybe this should be yes
    iraf.calibrate.setParam('ignoreaps','yes')

	# Image root name for sensitivity spectra
    iraf.calibrate.setParam('sensitivity','sens.fits')

	# Create spectra having units of FNU?
    iraf.calibrate.setParam('fnu','no')

    iraf.calibrate()


    return None


def instr_shape_corr(temp):


    """
    So at this point my telluric has been corrected for the wavelength sensitivity.
    Divide by blackbody to leave just the telluric lines.
    """

    print 'Removing intrinsic shape using blackbody T={}'.format(temp)

    hdulist = fits.open('cdimcomb.ms.fits')
    hdr = hdulist[0].header
    wa, dw = get_wavelength.get_wavelength(hdr)
    bbflux = planck(wa, temp)
    # It calibrates every spectrum in dimcomb.fits - which is optimally extracted, non-optimally extracted and variance spectrum.
    cal = hdulist[0].data[0,0,:]
    tspec = cal / bbflux
    tspec = tspec / np.median(tspec) # not sure why its necessary to remormalise at this point

    """
    Only works if linear scale because I don't know how to tell iraf its a log spectrum
    """

    hdu = fits.PrimaryHDU(tspec)
    hdu.header['CRVAL1'] = hdr['CRVAL1']
    hdu.header['CD1_1'] = hdr['CD1_1']
    hdu.header['CRPIX1'] = hdr['CRPIX1']
    hdulist = fits.HDUList([hdu])
    hdu.writeto('tell_line_spec.fits', clobber=True)

    # 2 so I don't get zero/negative fluxes, which telluric complains about
    hdu = fits.PrimaryHDU(2.0*np.ones(len(wa)))
    hdu.header['CRVAL1'] = hdr['CRVAL1']
    hdu.header['CD1_1'] = hdr['CD1_1']
    hdu.header['CRPIX1'] = hdr['CRPIX1']
    hdulist = fits.HDUList([hdu])
    hdu.writeto('uniform.fits', clobber=True)

    print 'Made files tell_line_spec.fits and uniform.fits \n Fit blended lines in splot then run mk1dspec \n Remember to remove existing splot.log before doing this'

    return None

def model_telluric():

    """
    Make model telluric spectrum which we will use in 'telluric'
    Need to fix output from splot for mk1dspec
    """

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    l = []
    with open('splot.log') as f:
        for line in f.readlines():
            if len(line.split()) == 7:
                if is_number(line.split()[0]):
            	    l.append(line.split())

    l = np.asarray(l, dtype=np.float)

    # in mk1dspec peak normalised to continum of one.
    # use gaussian - voigt seems to overestimate bkgd for some reason
    # wavelength peak gaussian gfwhm


    with open('splot_mod.log', 'w') as f:
        for line in l:
        	f.write(str(line[0]) + ' ' + str(1.0-line[1]+line[4]) + ' gaussian ' + str(line[5]) + '\n')

    if os.path.exists('model_tell_spec.fits'):
        os.remove('model_tell_spec.fits')
        print 'Removing file model_tell_spec.fits'

    iraf.artdata(_doprint=0)

    # List of input spectra
    iraf.mk1dspec.setParam('input','uniform.fits')

    # List of output spectra
    iraf.mk1dspec.setParam('output','model_tell_spec.fits')

    # Title of spectrum
    iraf.mk1dspec.setParam('title','telluric')

    # Number of columns
    iraf.mk1dspec.setParam('ncols',1024)

    # Starting wavelength (Angstroms)
    iraf.mk1dspec.setParam('wstart',13986.8488019)

    # Ending wavelength (Angstroms)
    iraf.mk1dspec.setParam('wend',23940.4525609)

    # Continuum at first pixel
    iraf.mk1dspec.setParam('continuum',0.)

    # Continuum slope per pixel
    iraf.mk1dspec.setParam('slope',0.)

    # Blackbody temperture (Kelvin). 0 = no blackbody
    iraf.mk1dspec.setParam('temperature',0.)

    # F-nu or F-lamda?
    iraf.mk1dspec.setParam('fnu','no')

    # List of files containing lines
    iraf.mk1dspec.setParam('lines','splot_mod.log')

    iraf.mk1dspec()

    hdulist = fits.open('dimcomb.ms.fits')
    airmass = hdulist[0].header['airmass']

    hdulist = fits.open('model_tell_spec.fits', mode='update')
    hdulist[0].header['airmass'] = airmass
    hdulist.flush()


    return None


def telluric_correct_standard():


    """
    Now do telluric correction on sensitivity corrected telluric spectrum,
    as a check.
    """

    if os.path.exists('tcdimcomb.ms.fits'):
        os.remove('tcdimcomb.ms.fits')
        print 'Removing file tcdimcomb.ms.fits'

    # List of input spectra to correct
    iraf.telluric.setParam('input', 'cdimcomb.ms.fits')

    # List of output corrected spectra
    iraf.telluric.setParam('output', 'tcdimcomb.ms.fits')

    # List of telluric calibration spectra
    iraf.telluric.setParam('cal', 'model_tell_spec.fits')

    # Airmass - of telluric spectrum I think, in this case the same as cdimcomb.ms.fits
    # should now read from header.
    # hdulist = fits.open('cdimcomb.ms.fits')
    # hdr = hdulist[0].header
    # airmass = hdr['airmass']
    # iraf.telluric.setParam('airmass', airmass)

    # Search interactively?
    iraf.telluric.setParam('answer', 'yes')

    # Ignore aperture numbers in calibration spectra?
    iraf.telluric.setParam('ignoreaps', 'yes')

    # Cross correlate for shift?
    iraf.telluric.setParam('xcorr', 'yes')

    # Tweak to minimize RMS?
    iraf.telluric.setParam('tweakrms', 'yes')

    # Interactive tweaking?
    iraf.telluric.setParam('interactive', 'yes')

    # Sample ranges
    # Don't fit bit between H and K
    iraf.telluric.setParam('sample', '15000:18000,19700:23800')

    # Threshold for calibration
    iraf.telluric.setParam('threshold', 0.)

    # Cross correlation lag (pixels)
    iraf.telluric.setParam('lag', 10)

    # Initial shift of calibration spectrum (pixels)
    iraf.telluric.setParam('shift', 0.)

    # Initial scale factor multiplying airmass ratio
    iraf.telluric.setParam('scale', 1.)

    # Initial shift search step
    iraf.telluric.setParam('dshift', 5.)

    # Initial scale factor search step
    iraf.telluric.setParam('dscale', 0.2)

    # Initial offset for graphs
    iraf.telluric.setParam('offset', 1.)

    # Smoothing box for graphs
    iraf.telluric.setParam('smooth', 3)

    iraf.telluric()

    return None

def check_magnitude():


    """
    Check Kband flux matches known K band magnitude:
    """

    hdulist = fits.open('tcdimcomb.ms.fits')
    hdr = hdulist[0].header
    wa, dw = get_wavelength.get_wavelength(hdr)
    data = hdulist[0].data[0,:,:].flatten()

    # Vega spectrum F(lambda)
    vega = np.genfromtxt('/data/vault/phewett/vista_work/vega_2007.lis')

    with open('/home/lc585/Dropbox/IoA/QSOSED/Model/Filter_Response/K.response','r') as f:
        ftrwav, ftrtrans = np.loadtxt(f,unpack=True)

    # Calculate vega zero point
    spc = interp1d(vega[:,0],
                   vega[:,1],
                   bounds_error=False,
                   fill_value=0.0)

    sum1 = np.sum( ftrtrans[:-1] * spc(ftrwav[:-1]) * ftrwav[:-1] * np.diff(ftrwav))
    sum2 = np.sum( ftrtrans[:-1] * ftrwav[:-1] * np.diff(ftrwav) )
    zromag = -2.5 * np.log10(sum1 / sum2)

    # Now calculate magnitudes
    spc = interp1d(wa,
                   data,
                   bounds_error=False,
                   fill_value=0.0)

    sum1 = np.sum( ftrtrans[:-1] * spc(ftrwav[:-1]) * ftrwav[:-1] * np.diff(ftrwav))
    sum2 = np.sum( ftrtrans[:-1] * ftrwav[:-1] * np.diff(ftrwav) )
    ftrmag = (-2.5 * np.log10(sum1 / sum2)) - zromag

    print 'Magnitude of standard star is {0:.2f} in K-band'.format(ftrmag)

    return None

def sens_corr_target():

    """
    Take out wavelength-dependent sensitivity
    """

    if not os.path.exists('sens.fits'):
    	print 'need file sens.fits'

    if not os.path.exists('model_tell_spec.fits'):
    	print 'need file model_tell_spec.fits'

    if os.path.exists('cdimcomb.ms.fits'):
        os.remove('cdimcomb.ms.fits')
        print 'Removing file cdimcomb.ms.fits'

    """
    for some reason dimcomb+bkgd doesn't have much in its fits header,
    so we need to add these keys words
    """

    # hdulist = fits.open('dimcomb.ms.fits')
    # airmass = hdulist[0].header['airmass']
    # exptime = float(hdulist[0].header['ncombine']) * 60.0

    # hdulist = fits.open('dimcomb.ms.fits', mode='update')
    # hdulist[0].header['airmass'] = airmass
    # hdulist[0].header['exptime'] = exptime
    # hdulist.flush()

    iraf.calibrate.unlearn()

    # Input spectra to calibrate. Output should have 'c' prefixed.
    iraf.calibrate.setParam('input','dimcomb.ms.fits')  # might need to specify extension.

    iraf.calibrate.setParam('output','cdimcomb.ms.fits')

    # Should be found automatically from header - check this for target spectrum.
    iraf.calibrate.setParam('airmass','')
    iraf.calibrate.setParam('exptime','')

    # Apply extinction correction? Not sure what this does
    iraf.calibrate.setParam('extinct','no')

	# Apply flux calibration?
    iraf.calibrate.setParam('flux','yes')

	# Extinction file
    iraf.calibrate.setParam('extinction','')

	# Ignore aperture numbers in flux calibration? Maybe this should be yes
    iraf.calibrate.setParam('ignoreaps','yes')

	# Image root name for sensitivity spectra
    iraf.calibrate.setParam('sensitivity','sens.fits')

	# Create spectra having units of FNU?
    iraf.calibrate.setParam('fnu','no')

    iraf.calibrate()

    return None


def telluric_corr_target():

    """
    Remove telluric lines (badly!)
    """

    if os.path.exists('tcdimcomb.ms.fits'):
        os.remove('tcdimcomb.ms.fits')
        print 'Removing file tcdimcomb.ms.fits'


    # List of input spectra to correct
    iraf.telluric.setParam('input', 'cdimcomb.ms.fits')

    # List of output corrected spectra
    iraf.telluric.setParam('output', 'tcdimcomb.ms.fits')

    # List of telluric calibration spectra
    iraf.telluric.setParam('cal', 'model_tell_spec.fits')

    # Airmass - of telluric spectrum I think, should read from header of model_tell_spec.fits
    # iraf.telluric.setParam('airmass', airmass)

    # Search interactively?
    iraf.telluric.setParam('answer', 'yes')

    # Ignore aperture numbers in calibration spectra?
    iraf.telluric.setParam('ignoreaps', 'yes')

    # Cross correlate for shift?
    iraf.telluric.setParam('xcorr', 'yes')

    # Tweak to minimize RMS?
    iraf.telluric.setParam('tweakrms', 'yes')

    # Interactive tweaking?
    iraf.telluric.setParam('interactive', 'yes')

    # Sample ranges
    # Don't fit bit between H and K
    iraf.telluric.setParam('sample', '15000:18000,19700:23800')

    # Threshold for calibration
    iraf.telluric.setParam('threshold', 0.)

    # Cross correlation lag (pixels)
    iraf.telluric.setParam('lag', 10)

    # Initial shift of calibration spectrum (pixels)
    iraf.telluric.setParam('shift', 0.)

    # Initial scale factor multiplying airmass ratio
    iraf.telluric.setParam('scale', 1.)

    # Initial shift search step
    iraf.telluric.setParam('dshift', 5.)

    # Initial scale factor search step
    iraf.telluric.setParam('dscale', 0.2)

    # Initial offset for graphs
    iraf.telluric.setParam('offset', 1.)

    # Smoothing box for graphs
    iraf.telluric.setParam('smooth', 3)

    iraf.telluric()


    return None



def iraf_telluric_correction(standard,
							 sensfunc,
							 senscorr,
							 shapecorr,
							 modtell,
							 tellcorr,
							 checkmag,
							 senscorr_targ,
							 tellcorr_targ,
							 vmag,
                             kmag,
	                         spec_type,
	                         temp):

    """
    Run this in telluric directory to prepare sens and model_tell_spec, which will be
    used to correct target for wavelength-dependent sensitivity and (badly) remove
    telluric absorption lines.
    """

    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)

    if standard == 'yes':

        if os.path.exists('dimcomb.ms.fits'):
        	hdulist = fits.open('dimcomb.ms.fits')
        	hdr = hdulist[0].header
    		if hdr['CRVAL1'] < 10:
        	    print 'Need linear wavelength scale'

        else:
        	print 'Need dispersion corrected telluric spectrum'

    	telluric_standard(vmag, kmag, spec_type)

    if sensfunc == 'yes':
    	telluric_sensfunc()

    if senscorr == 'yes':
    	sens_corr_standard()

    if shapecorr == 'yes':
    	instr_shape_corr(temp)

    if modtell == 'yes':
        model_telluric()

    if tellcorr == 'yes':
        telluric_correct_standard()

    if checkmag == 'yes':
    	check_magnitude()

    if senscorr_targ == 'yes':
    	sens_corr_target()

    if tellcorr_targ == 'yes':
    	telluric_corr_target()

    """
    Okay, if this all runs okay, I can copy sens.fits and model_tell_spec.lits to the target directory
    and run calibrate and telluric. sens_corr_standard() and telluric() should both work in the same
    way
    """


    return None

parfile = iraf.osfn('/home/lc585/spectra/iraf_telluric_correction.par')
t = iraf.IrafTaskFactory(taskname="iraf_telluric_correction", value=parfile, function=iraf_telluric_correction)

