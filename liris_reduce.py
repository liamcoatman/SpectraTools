# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 12:21:58 2015

@author: lc585

Collection of pyraf scripts primarily for reducing LIRIS spectra

You need to launch pyraf in the iraf directory, and then cd to the directory containing un-reduced spectra.
pyexecute this script, epar liris_reduce, and select desired reduction steps.

See this for a good guide

http://snova.fysik.su.se//private/near-z/spectroscopy/spectra.long.html
"""
from __future__ import division

import os
from pyraf import iraf
import shutil
import numpy as np
import glob
import uuid
from astropy.table import Table
from astropy.io import fits
import shutil

homedir = os.path.expanduser("~")

def set_instrument():

    """
    Needed so that when we do flatcombine iraf doesn't try to do flat correction
    etc. Might need to run before calling any other function. Not sure if specphot
    is appropriate but seems to work.
    """

    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.ccdred(_doprint=0)

    iraf.setinstrument.setParam('instrument','specphot')
    iraf.setinstrument()

    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/set_instrument.par') )
t = iraf.IrafTaskFactory(taskname="set_instrument", value=parfile, function=set_instrument)

def liris_pixmap(targetdir):

    """
    Currently there is a shift of pixels on the detector which does not correspond
    to the actual geometrical location. Fixes alignment between top and bottom quadrants
    """

    print 'In directory ' + targetdir
    print 'Applying lxpixmap correction...'

    iraf.lirisdr(_doprint=0)

    # Make input. Finds all files .fit and assumes image in extension 1.
    names = [name for name in os.listdir(targetdir) if (name.endswith('.fit')) & (name.startswith('r'))]

    with open(os.path.join(targetdir,'input.list'), 'w') as f:
        for name in names:
            f.write( os.path.join(targetdir,name + '\n') )

    # If output files exist, then remove them.
    for n in names:
        if os.path.exists( os.path.join(targetdir,'c' + n) ):
            os.remove(os.path.join(targetdir,'c' + n) )
            print 'Deleting file ' + os.path.join(targetdir,'c' + n)

    iraf.lcpixmap.setParam('input','@' + os.path.join(targetdir, 'input.list'))
    iraf.lcpixmap.setParam('output','c')
    iraf.lcpixmap.setParam('outlist','')

    iraf.lcpixmap()

    # Files end up in cwd so need to move them

    cwd = os.getcwd()
    for name in names:
        oldpath = os.path.join( cwd, 'c' + name )
        newpath = os.path.join( targetdir, 'c' + name )
        shutil.move( oldpath, newpath )

    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/liris_pixmap.par') )
t = iraf.IrafTaskFactory(taskname="liris_pixmap", value=parfile, function=liris_pixmap)


def combine_flats(flatdir):

    """
    Co-add flat-field images. Check no variation night to night and then add all together.

    Need to first run set_instrument

    Once combined should use imarith to divide each by the average and check see only
    noise and no residuals

    First should lcpixmap flat fields

    """

    print 'In directory ' + flatdir
    print 'Combining flats...'

    if os.path.exists( os.path.join(flatdir,'Flat.fits') ):
        os.remove( os.path.join(flatdir,'Flat.fits') )
        print 'Removing file ' + os.path.join(flatdir,'Flat.fits')

    names = [name + '[1]' for name in os.listdir(flatdir) if (name.endswith('.fit')) & (name.startswith('cr')) ]

#    # Give write permission to files
#    for name in names:
#        os.chmod(name.replace('[1]',''),0644)
#
    with open( os.path.join(flatdir,'input.list'), 'w') as f:
        for name in names:
            f.write( os.path.join(flatdir,name + '\n') )

    iraf.flatcombine.setParam('input', '@' + os.path.join(flatdir, 'input.list') )
    iraf.flatcombine.setParam('rdnoise','readnois')
    iraf.flatcombine.setParam('combine','average')
    iraf.flatcombine.setParam('reject','crreject')
    iraf.flatcombine.setParam('ccdtype','flat')
    iraf.flatcombine.setParam('scale','mode')
    iraf.flatcombine.setParam('lsigma',3.0)
    iraf.flatcombine.setParam('hsigma',3.0)
    iraf.flatcombine.setParam('output', os.path.join(flatdir,'Flat.fits') )

    iraf.flatcombine()

    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/combine_flats.par' ) )
t = iraf.IrafTaskFactory(taskname="combine_flats", value=parfile, function=combine_flats)

def check_flats():

    """
    Divide each flat-field by the average.

    Individually inspect each 2D spectrum and if see features need to discard

    """

    cwd = os.getcwd()
    print 'Current working directory is ' + cwd

    with open('input.list','w') as f:
        for name in os.listdir(cwd):
            if (name.endswith('.fit')) & (name.startswith('cr')):
                f.write( name + '[1]' + '\n')

    with open('output.list','w') as f:
        for name in os.listdir(cwd):
            if (name.endswith('.fit')) & (name.startswith('cr')):
                if os.path.exists('check_' + name):
                    os.remove('check_' + name)
                    print 'Deleting file ' + 'check_' + name
                f.write( 'check_' + name + '\n')

    iraf.imarith.setParam('operand1', '@input.list')
    iraf.imarith.setParam('operand2', 'Flat.fits')
    iraf.imarith.setParam('op','/')
    iraf.imarith.setParam('result','@output.list')

    iraf.imarith()

    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/check_flats.par') )
t = iraf.IrafTaskFactory(taskname="check_flats", value=parfile, function=check_flats)

def normalise_flats(flatdir):

    """
    Normalise flats
    """

    print 'In directory ' + flatdir
    print 'Normalising combinined flats...'

    if os.path.exists( os.path.join(flatdir,'nFlat.fits') ):
        os.remove(os.path.join(flatdir,'nFlat.fits') )
        print 'Removing file ' + os.path.join(flatdir,'nFlat.fits')

    iraf.twodspec(_doprint=0)
    iraf.longslit(_doprint=0)

    iraf.response.setParam('calibration', os.path.join(flatdir,'Flat.fits'))
    iraf.response.setParam('normalization', os.path.join( flatdir, 'Flat.fits' ))
    iraf.response.setParam('response', os.path.join( flatdir, 'nFlat') )
    iraf.response.setParam('low_reject', 3.)
    iraf.response.setParam('high_reject', 3.)
    iraf.response.setParam('order',40)

    iraf.response()

    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/normalise_flats.par') )
t = iraf.IrafTaskFactory(taskname="normalise_flats", value=parfile, function=normalise_flats)

def sort_frames(targetdir):

    """
    This checks that all objects can be sorted in to AB pairs.

    """

    print 'In directory ' + targetdir
    print 'Sorting Frames...'

    names = np.array([name + '[1]' for name in os.listdir(targetdir) if (name.endswith('.fit')) & (name.startswith('r'))])


    mjd = np.array([iraf.hselect( os.path.join(targetdir,f), 'MJD-OBS', 'yes', Stdout=1)[0] for f in names])

    # Sort files names in order of time of observation
    names = names[np.argsort(mjd)]

    for n in names: print n.replace('.fit[1]','')

    # Check everything is in the right order
    decoff = np.asarray([iraf.hselect(os.path.join(targetdir,f) , 'DECOFF', 'yes', Stdout=1)[0] for f in names],dtype='float')

    pos = np.repeat('A',len(names))
    pos[ decoff < -1.] = 'B'

    if len(names) % 2 != 0:
        print 'Odd number of nods!'
        return False

    for i in range(int(len(names)/2)):

        pair = pos[i*2] + pos[(i*2)+1]
        print pair

        if (pair == 'AA') or (pair == 'BB'):
            print 'Mismatched pairs!'
            return False

    print 'Everything is okay!'

    return True

parfile = iraf.osfn( os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/sort_frames.par') )
t = iraf.IrafTaskFactory(taskname="sort_frames", value=parfile, function=sort_frames)

def flat_correction(targetdir,flatdir):

    """
    Flat field correction
    """

    print 'Target directory is ' + targetdir
    print 'Flat directory is ' + flatdir
    print 'Applying flat field correction...'

    innames, outnames = [], []

    for n in os.listdir(targetdir):
        if (n.endswith('.fit')) & (n.startswith('cr')):
            outnames.append( os.path.join(targetdir,'f' + n) )
            innames.append( os.path.join( targetdir, n + '[1]') )
            if os.path.exists( os.path.join( targetdir, 'f' + n) ):
                print 'Removing file ' + os.path.join( targetdir, 'f' + n)
                os.remove( os.path.join( targetdir, 'f' + n) )

    with open( os.path.join(targetdir,'input.list'), 'w') as f:
        for name in innames:
            f.write( name + '\n' )

    with open( os.path.join(targetdir,'output.list'), 'w') as f:
        for name in outnames:
            f.write( name + '\n' )

    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.ccdred(_doprint=0)

    iraf.ccdproc.setParam('images', '@' + os.path.join(targetdir, 'input.list') )
    iraf.ccdproc.setParam('flatcor','yes')
    iraf.ccdproc.setParam('flat', os.path.join(flatdir,'nFlat.fits') )
    iraf.ccdproc.setParam('output', '@' + os.path.join(targetdir, 'output.list'))

    iraf.ccdproc()

    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/flat_correction.par') )
t = iraf.IrafTaskFactory(taskname="flat_correction", value=parfile, function=flat_correction)

def sky_subtraction(targetdir):

    """
    Subtract A-B frames to remove sky

    """

    print 'Target directory is ' + targetdir
    print 'Doing sky subtraction...'

    names = np.array([name + '[1]' for name in os.listdir(targetdir) if (name.endswith('.fit')) & (name.startswith('fcr'))])

    mjd = np.array([iraf.hselect( os.path.join(targetdir,f), 'MJD-OBS', 'yes', Stdout=1)[0] for f in names])

    # Sort files names in order of time of observation
    names = names[np.argsort(mjd)]
    decoff = np.asarray([iraf.hselect(os.path.join(targetdir,f) , 'DECOFF', 'yes', Stdout=1)[0] for f in names],dtype='float')

    for n,d in zip(names,decoff):
        if d > -1.0:
            spos = 'A'
        else:
            spos = 'B'
        print n.replace('.fit[1]',''), spos


    if os.path.exists(os.path.join(targetdir,'resultsAB.list') ):
        os.remove(os.path.join(targetdir,'resultsAB.list') )
        print 'Removing file ' + os.path.join(targetdir,'resultsAB.list')

    if os.path.exists(os.path.join(targetdir,'resultsBA.list') ):
        os.remove(os.path.join(targetdir,'resultsBA.list') )
        print 'Removing file ' + os.path.join(targetdir,'resultsBA.list')

    if os.path.exists(os.path.join(targetdir,'A.list') ):
        os.remove(os.path.join(targetdir,'A.list' ) )
        print 'Removing file ' + os.path.join(targetdir,'A.list' )

    if os.path.exists(os.path.join(targetdir,'B.list') ):
        os.remove(os.path.join(targetdir,'B.list') )
        print 'Removing file ' + os.path.join(targetdir,'B.list')

    for i in range(int(len(names)/2)):

        j = i * 2

        # Looping through pairs, determine which is A and which is B

        if decoff[j] > -1.0:
            nameA = names[j]
            nameB = names[j+1]
        else:
            nameA = names[j+1]
            nameB = names[j]

        with open( os.path.join( targetdir,'A.list'),'a') as f:
            f.write( os.path.join(targetdir,nameA.replace('[1]','')) + '\n')

        with open( os.path.join( targetdir,'B.list'),'a') as f:
            f.write( os.path.join(targetdir,nameB.replace('[1]','')) + '\n')

        nameResultAB = 'd' + nameA
        nameResultAB = nameResultAB.replace('.fit[1]','') + '.fit'
        nameResultAB = os.path.join(targetdir,nameResultAB)

        with open(os.path.join(targetdir,'resultsAB.list'), 'a') as f:
            f.write(nameResultAB + '\n')

        if os.path.exists(os.path.join(targetdir,nameResultAB) ):
            os.remove(os.path.join(targetdir,nameResultAB) )
            print 'Removing file ' + os.path.join(targetdir,nameResultAB)

        nameResultBA = 'd' + nameB
        nameResultBA = nameResultBA.replace('.fit[1]','') + '.fit'
        nameResultBA = os.path.join(targetdir,nameResultBA)

        with open(os.path.join(targetdir,'resultsBA.list'), 'a') as f:
            f.write(nameResultBA + '\n')

        if os.path.exists(os.path.join(targetdir,nameResultBA) ):
            os.remove(os.path.join(targetdir,nameResultBA) )
            print 'Removing file ' + os.path.join(targetdir,nameResultBA)

    iraf.imarith.setParam('operand1', '@' + os.path.join(targetdir,'A.list') )
    iraf.imarith.setParam('operand2', '@' + os.path.join(targetdir,'B.list') )
    iraf.imarith.setParam('op','-')
    iraf.imarith.setParam('result','@' + os.path.join(targetdir,'resultsAB.list') )

    iraf.imarith()

    iraf.unlearn('imarith')

    iraf.imarith.setParam('operand1', '@' + os.path.join(targetdir,'resultsAB.list') )
    iraf.imarith.setParam('operand2', -1.0)
    iraf.imarith.setParam('op','*')
    iraf.imarith.setParam('result', '@' + os.path.join(targetdir,'resultsBA.list'))

    iraf.imarith()

    return None

parfile = iraf.osfn( os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/sky_subtraction.par') )
t = iraf.IrafTaskFactory(taskname="sky_subtraction", value=parfile, function=sky_subtraction)

def image_shift(targetdir):

    """
    Shift all B-A images by 48 pixels in y-direction
    """

    print 'Target directory is ' + targetdir
    print 'Shifting images...'

    innames, outnames = [], []

    for n in os.listdir(targetdir):
        if (n.endswith('.fit')) & (n.startswith('dfcr')):
            outnames.append(os.path.join(targetdir,'s' + n ) )
            innames.append(os.path.join(targetdir,n) )
            if os.path.exists( os.path.join(targetdir,'s' + n) ):
                os.remove(os.path.join(targetdir,'s' + n))
                print 'Removing file ' + os.path.join(targetdir,'s' + n)

    with open(os.path.join(targetdir,'input.list'), 'w') as f:
        for name in innames:
            print name.replace(targetdir + '/','')
            f.write( name + '\n' )

    with open(os.path.join(targetdir,'output.list'), 'w') as f:
        for name in outnames:
            f.write( name + '\n' )

    names = np.array([name.replace('dfcr','r') + '[1]' for name in innames])

    decoff = np.asarray([iraf.hselect(f , 'DECOFF', 'yes', Stdout=1)[0] for f in names],dtype='float')

    with open(os.path.join(targetdir,'shifts.lst'),'w') as f:
        for d in decoff:
            if d >= -1.0:
                yshift = '0.0'
            else:
                yshift = '-48.0'
            f.write('0.0 ' + yshift + '\n')

    iraf.images(_doprint=0)
    iraf.imgeom(_doprint=0)

    iraf.imshift.setParam('input', '@' + os.path.join(targetdir, 'input.list'))
    iraf.imshift.setParam('output', '@' + os.path.join(targetdir, 'output.list'))
    iraf.imshift.setParam('shifts_file', os.path.join(targetdir,'shifts.lst') )

    iraf.imshift()

    return None

parfile = iraf.osfn( os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/image_shift.par') )
t = iraf.IrafTaskFactory(taskname="image_shift", value=parfile, function=image_shift)

def image_combine(targetdir):

    """
    Combine all images. Need to run image_shift before.

    """
    print 'Target directory is ' + targetdir
    print 'Combining images...'


    innames = []

    for n in os.listdir(targetdir):
        if (n.endswith('.fit')) & (n.startswith('sdfcr')):
            innames.append(os.path.join(targetdir,n) )


    with open(os.path.join(targetdir,'input.list'), 'w') as f:
        for name in innames:
            print name.replace(targetdir + '/','')
            f.write( name + '\n' )

    if os.path.exists( os.path.join(targetdir,'imcomb.fit')):
        os.remove(os.path.join(targetdir,'imcomb.fit'))
        print 'Removing file ' + os.path.join(targetdir,'imcomb.fit')

    iraf.images(_doprint=0)
    iraf.immatch(_doprint=0)

    iraf.imcombine.setParam('input','@' + os.path.join(targetdir, 'input.list'))
    iraf.imcombine.setParam('output',os.path.join(targetdir,'imcomb.fit') )
    iraf.imcombine.setParam('combine','average')
    iraf.imcombine.setParam('reject','sigclip')
    iraf.imcombine.setParam('lsigma',3.0)
    iraf.imcombine.setParam('hsigma',3.0)

    iraf.imcombine()

    # Or, combine without background subtraction


    if os.path.exists( os.path.join(targetdir,'imcomb+bkgd.fit')):
        os.remove( os.path.join(targetdir, 'imcomb+bkgd.fit') )
        print 'Removing file ' + os.path.join(targetdir, 'imcomb+bkgd.fit')

    hdulist = fits.open( os.path.join( targetdir, 'imcomb.fit'))
    data = hdulist[0].data
    hdulist.close()

    # import matplotlib.pyplot as plt
    # from spectra.range_from_zscale import range_from_zscale
    # fig, ax = plt.subplots()
    # z1, z2, iteration = range_from_zscale(data)
    # ax.imshow(data,cmap='gray',vmin=z1,vmax=z2)
    # plt.gca().invert_yaxis()

    data = np.concatenate( ( data[32:345,:], data[520:650,:], data[740:940,:]))

    sigmaspec = np.std( data, axis = 0)

    # fig, ax = plt.subplots()
    # ax.plot( sigmaspec, lw=1 )

    # Convert sky from adu to electron
    skyspec = (4 * sigmaspec)**2
    sky_adu = skyspec / 4.0

    # add to each pixel

    hdulist = fits.open( os.path.join( targetdir, 'imcomb.fit'))
    data = hdulist[0].data
    hdulist.close()

    sky_adu = np.repeat(sky_adu, np.shape(data)[0] ).reshape( np.shape(data)[::-1] ).T

    data = data + sky_adu

    hdu = fits.PrimaryHDU(data)
    hdu.writeto( os.path.join( targetdir, 'imcomb+bkgd.fit') )


    return None

parfile = iraf.osfn( os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/image_combine.par') )
t = iraf.IrafTaskFactory(taskname="image_combine", value=parfile, function=image_combine)

def extract_spectrum(targetdir,
                     trace,
                     arcspec,
                     refspec,
                     t_nsum,
                     t_step,
                     line,
                     ylevel,
                     interactive):

    """
    Extract spectrum

    Must be in target directory.

    """

    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    iraf.twodspec(_doprint=0)
    iraf.apextract(_doprint=0)

    basedir = '/data/lc585/WHT_20150331/OBS/'
    targetdir = os.path.join(basedir,targetdir,'Reduced')

    print 'Target directory is ' + targetdir
    print 'Extracting spectrum...'


    if os.path.exists( os.path.join(targetdir, 'imcomb.ms.fits') ):
        os.remove( os.path.join( targetdir, 'imcomb.ms.fits') )
        print 'Removing file ' + os.path.join( targetdir, 'imcomb.ms.fits')

    # If can't fit trace use trace from nearby object

    if trace == 'no':

        dest = os.path.join(targetdir,'database')

        if not os.path.exists(dest):
            os.makedirs(dest)

        db = os.path.join(basedir,refspec,'Reduced','database','ap_data_lc585_WHT_20150331_OBS_'+refspec+'_Reduced_imcomb')

        shutil.copy(db,dest)

        iraf.apall.setParam('references',os.path.join(basedir,refspec,'Reduced','imcomb.fit')) # List of aperture reference images

        print 'Using trace from reference spectra ' + refspec

    # Since frame is averaged I think we need to scale down read nosie but gain will stay the same.
    names = []
    for n in os.listdir(targetdir):
        if (n.endswith('.fit')) & (n.startswith('r')):
            names.append(n)

    nframes = float(len(names))

    # Doesn't seem to work if I give it absolute path to input!
    iraf.apall.setParam('input','imcomb.fit') # List of input images
    iraf.apall.setParam('output','') # List of output spectra
    iraf.apall.setParam('apertur','') # Apertures
    iraf.apall.setParam('format','multispec') # Extracted spectra format
    iraf.apall.setParam('referen','') # List of aperture reference images
    iraf.apall.setParam('profile','') # List of aperture profile images

    iraf.apall.setParam('interac',interactive) # Run task interactively?
    iraf.apall.setParam('find','no') # Find apertures?
    iraf.apall.setParam('recente','no') # Recenter apertures?
    iraf.apall.setParam('resize','no') # Resize apertures?
    iraf.apall.setParam('edit','yes') # Edit apertures?
    iraf.apall.setParam('trace',trace) # Trace apertures?
    iraf.apall.setParam('fittrac',interactive) # Fit the traced points interactively?
    iraf.apall.setParam('extract','yes') # Extract spectra?
    iraf.apall.setParam('extras','yes') # Extract sky, sigma, etc.?
    iraf.apall.setParam('review',interactive) # Review extractions?

    iraf.apall.setParam('line',line) # Dispersion line
    iraf.apall.setParam('nsum',20) # Number of dispersion lines to sum or median

                                # DEFAULT APERTURE PARAMETERS

    iraf.apall.setParam('lower',-5.) # Lower aperture limit relative to center
    iraf.apall.setParam('upper',5.) # Upper aperture limit relative to center
    iraf.apall.setParam('apidtab','') # Aperture ID table (optional)

                                # DEFAULT BACKGROUND PARAMETERS
    # Background is now a constant at each wavelength
    iraf.apall.setParam('b_funct','chebyshev') # Background function
    iraf.apall.setParam('b_order',1) # Background function order
    iraf.apall.setParam('b_sampl','-10:-6,6:10') # Background sample regions
    iraf.apall.setParam('b_naver',-3) # Background average or median
    iraf.apall.setParam('b_niter',2) # Background rejection iterations
    iraf.apall.setParam('b_low_r',3.) # Background lower rejection sigma
    iraf.apall.setParam('b_high_',3.) # Background upper rejection sigma
    iraf.apall.setParam('b_grow',0.) # Background rejection growing radius

                                # APERTURE CENTERING PARAMETERS

    iraf.apall.setParam('width',5.) # Profile centering width
    iraf.apall.setParam('radius',10.) # Profile centering radius
    iraf.apall.setParam('thresho',0.) # Detection threshold for profile centering

                                # AUTOMATIC FINDING AND ORDERING PARAMETERS

    iraf.apall.setParam('nfind','') # Number of apertures to be found automatically
    iraf.apall.setParam('minsep',5.) # Minimum separation between spectra
    iraf.apall.setParam('maxsep',100000.) # Maximum separation between spectra
    iraf.apall.setParam('order','increasing') # Order of apertures

                                # RECENTERING PARAMETERS

    iraf.apall.setParam('aprecen','') # Apertures for recentering calculation
    iraf.apall.setParam('npeaks','INDEF') # Select brightest peaks
    iraf.apall.setParam('shift','yes') # Use average shift instead of recentering?

                                # RESIZING PARAMETERS

    iraf.apall.setParam('llimit','INDEF') # Lower aperture limit relative to center
    iraf.apall.setParam('ulimit','INDEF') # Upper aperture limit relative to center
    iraf.apall.setParam('ylevel',0.2) # Fraction of peak or intensity for automatic widt
    iraf.apall.setParam('peak','yes') # Is ylevel a fraction of the peak?
    iraf.apall.setParam('bkg','yes') # Subtract background in automatic width?
    iraf.apall.setParam('r_grow',0.) # Grow limits by this factor
    iraf.apall.setParam('avglimi','no') # Average limits over all apertures?

                                # TRACING PARAMETERS

    iraf.apall.setParam('t_nsum',20) # Number of dispersion lines to sum
    iraf.apall.setParam('t_step', 20) # Tracing step
    iraf.apall.setParam('t_nlost',3) # Number of consecutive times profile is lost befo
    iraf.apall.setParam('t_funct','spline3') # Trace fitting function
    iraf.apall.setParam('t_order',2) # Trace fitting function order
    iraf.apall.setParam('t_sampl','*') # Trace sample regions
    iraf.apall.setParam('t_naver',1) # Trace average or median
    iraf.apall.setParam('t_niter',2) # Trace rejection iterations
    iraf.apall.setParam('t_low_r',3.) # Trace lower rejection sigma
    iraf.apall.setParam('t_high_',3.) # Trace upper rejection sigma
    iraf.apall.setParam('t_grow',0.) # Trace rejection growing radius

                                # EXTRACTION PARAMETERS

    iraf.apall.setParam('backgro','none') # Background to subtract
    iraf.apall.setParam('skybox',1) # Box car smoothing length for sky
    iraf.apall.setParam('weights','variance') # Extraction weights (none|variance)
    iraf.apall.setParam('pfit','fit1d') # Profile fitting type (fit1d|fit2d)
    iraf.apall.setParam('clean','yes') # Detect and replace bad pixels?
    iraf.apall.setParam('saturat',300000.) # Saturation level
    # iraf.apall.setParam('readnoi',17.0)
    iraf.apall.setParam('readnoi',17./np.sqrt(nframes)) # Read out noise sigma (photons)
    iraf.apall.setParam('gain',4.) # Photon gain (photons/data number)
    iraf.apall.setParam('lsigma',4.) # Lower rejection threshold
    iraf.apall.setParam('usigma',4.) # Upper rejection threshold
    iraf.apall.setParam('nsubaps',1) # Number of subapertures per aperture
    iraf.apall.setParam('mode','q') # h = hidden, q = query, l = learn

    iraf.apall()

    # Now extract arc through same aperture for wavelength calibration

    print '\n' '\n' '\n'
    print 'Extracting Arc through same aperture...'

    if os.path.exists( os.path.join(targetdir,'aimcomb.fits')):
        os.remove( os.path.join(targetdir, 'aimcomb.fits') )
        print 'Removing file ' + os.path.join(targetdir, 'aimcomb.fits')


    arcspec = os.path.join(basedir,arcspec)

    iraf.apall.setParam('input', arcspec)
    iraf.apall.setParam('output', 'aimcomb')
    iraf.apall.setParam('references', 'imcomb.fit' )
    iraf.apall.setParam('recenter','no')
    iraf.apall.setParam('trace','no')
    iraf.apall.setParam('background','no')
    iraf.apall.setParam('interactive','no')

    iraf.apall()


    if os.path.exists( os.path.join(targetdir, 'imcomb+bkgd.ms.fits') ):
        os.remove( os.path.join( targetdir, 'imcomb+bkgd.ms.fits') )
        print 'Removing file ' + os.path.join( targetdir, 'imcomb+bkgd.ms.fits')


    iraf.apall.setParam('input','imcomb+bkgd.fit') # List of input images
    iraf.apall.setParam('output','') # List of output spectra
    iraf.apall.setParam('referen','imcomb.fit') # List of aperture reference images

    iraf.apall.setParam('interac','yes') # Run task interactively?
    iraf.apall.setParam('find','yes') # Find apertures?
    iraf.apall.setParam('recenter','no') # Recenter apertures?
    iraf.apall.setParam('resize','no') # Resize apertures?
    iraf.apall.setParam('edit','yes') # Edit apertures?
    iraf.apall.setParam('trace','no') # Trace apertures?
    iraf.apall.setParam('fittrac',interactive) # Fit the traced points interactively?
    iraf.apall.setParam('extract','yes') # Extract spectra?
    iraf.apall.setParam('extras','yes') # Extract sky, sigma, etc.?
    iraf.apall.setParam('review','yes') # Review extractions?

                                # DEFAULT BACKGROUND PARAMETERS
    # Background is now a constant at each wavelength
    iraf.apall.setParam('b_funct','chebyshev') # Background function
    iraf.apall.setParam('b_order',1) # Background function order
    iraf.apall.setParam('b_sampl','-10:-6,6:10') # Background sample regions
    iraf.apall.setParam('b_naver',-3) # Background average or median
    iraf.apall.setParam('b_niter',2) # Background rejection iterations
    iraf.apall.setParam('b_low_r',3.) # Background lower rejection sigma
    iraf.apall.setParam('b_high_',3.) # Background upper rejection sigma
    iraf.apall.setParam('b_grow',0.) # Background rejection growing radius

                                # EXTRACTION PARAMETERS

    # before i wasn't dividing by the square root of the frames, but surely this must be true if I'm taking the average

    iraf.apall.setParam('backgro','median') # Background to subtract
    iraf.apall.setParam('skybox',1) # Box car smoothing length for sky
    iraf.apall.setParam('weights','variance') # Extraction weights (none|variance)
    iraf.apall.setParam('pfit','fit1d') # Profile fitting type (fit1d|fit2d)
    iraf.apall.setParam('clean','yes') # Detect and replace bad pixels?
    iraf.apall.setParam('saturat',300000.) # Saturation level
    # iraf.apall.setParam('readnoi',17.0)
    iraf.apall.setParam('readnoi',17.0/np.sqrt(nframes)) # Read out noise sigma (photons)
    iraf.apall.setParam('gain',4.) # Photon gain (photons/data number)
    iraf.apall.setParam('lsigma',4.) # Lower rejection threshold
    iraf.apall.setParam('usigma',4.) # Upper rejection threshold
    iraf.apall.setParam('nsubaps',1) # Number of subapertures per aperture

    iraf.apall()

    hdulist = fits.open(os.path.join(targetdir, 'imcomb+bkgd.ms.fits'))
    sigma = hdulist[0].data[3,0,:]
    hdulist.close()

    hdulist = fits.open(os.path.join(targetdir, 'imcomb.ms.fits'), mode='update')
    hdulist[0].data[2,0,:] = sigma
    hdulist.flush()
    hdulist.close()

    return None

def wavelength_calibration(targetdir):

    """
    Does wavelength calibration.

    Writes every fit to database so make sure it's using the correct one.

    This needs to be run in object directory for database

    """

    print 'Target directory is ' + targetdir
    print 'Doing wavelength calibration...'

    if os.getcwd() != targetdir:

        print 'Warning: current working directory must be target directory!'

        return None

    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)

    iraf.unlearn('identify')

    iraf.identify.setParam('images','aimcomb.fits')
    iraf.identify.setParam('coordli','/home/lc585/Dropbox/IoA/WHT_Proposal_2015a/argon+xenon.dat')
    iraf.identify.setParam('niterat',1)
    iraf.identify.setParam('function','spline3')
    iraf.identify.setParam('order',3)
    iraf.identify.setParam('zwidth',200.0) #  Zoom graph width in user units
    iraf.identify.setParam('database','database')

    iraf.identify()

    # Update fits header

    print '\n' '\n' '\n'
    print 'Updating fits header...'

    iraf.hedit.setParam('images','imcomb.ms.fits')
    iraf.hedit.setParam('fields','REFSPEC1')
    iraf.hedit.setParam('value','aimcomb.fits') # should be wavelength calibrated?
    iraf.hedit.setParam('add','yes')
    iraf.hedit.setParam('verify','yes')
    iraf.hedit.setParam('show','yes')

    iraf.hedit()

    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/wavelength_calibration.par') )
t = iraf.IrafTaskFactory(taskname="wavelength_calibration", value=parfile, function=wavelength_calibration)



def wavelength_solution(targetdir,
                        islog,
                        w1,
                        w2,
                        npix,
                        dw,
                        dv):

    """
    Applies wavelength solution using dispcor. Must be run in target directory.
    """

    print 'Target directory is ' + targetdir
    print 'Applying wavelength solution...'

    if os.getcwd() != targetdir:

        raise ValueError('Current working directory must be target directory!')

    if os.path.exists('dimcomb.ms.fits'):
        os.remove('dimcomb.ms.fits')

    iraf.unlearn('dispcor')

    iraf.dispcor.setParam('input','imcomb.ms.fits')
    iraf.dispcor.setParam('output','dimcomb.ms.fits')
    iraf.dispcor.setParam('confirm','yes')

    if len(w1) == 0:
        w1 = None
    if len(w2) == 0:
        w2 = None
    if len(npix) == 0:
        npix = None
    if len(dw) == 0:
        dw = None
    if len(dv) == 0:
        dv = None

    if islog == 'no':

        iraf.dispcor.setParam('log','no')

        if None not in (w1,w2,npix,dw):
            w1 = float(w1)
            w2 = float(w2)
            npix = float(npix)
            dw = float(dw)

        elif None not in (w1, w2, npix):
            w1 = float(w1)
            w2 = float(w2)
            npix = float(npix)
            dw = 'INDEF'

        elif None not in (w1, w2, dw):
            w1 = float(w1)
            w2 = float(w2)
            dw = float(dw)
            npix = 'INDEF'

        elif None not in (w1, dw, npix):
            w1 = float(w1)
            npix = float(npix)
            dw = float(dw)
            w2 = 'INDEF'

        elif None not in (w2, dw, npix):
            w2 = float(w1)
            npix = float(npix)
            dw = float(dw)
            w1 = 'INDEF'

        else:
            raise ValueError('Not enough info to make a wavelength scale!')


        print 'User input w1 = {}, w2 = {}, dw = {}, nw = {}'.format(w1,w2,dw,npix)

        iraf.dispcor.setParam('w1',w1) # Starting wavelength
        iraf.dispcor.setParam('w2',w2) # Ending wavelength
        iraf.dispcor.setParam('nw',npix) # Number of output pixels
        iraf.dispcor.setParam('dw',dw) # Wavelength interval per pixel

    elif islog == 'yes':

        iraf.dispcor.setParam('log','yes')

        # Leave everything black ecept dv

        if None not in (w1,w2,dv):

            w1 = float(w1)
            w2 = float(w2)
            dv = float(dv)
            print 'Constant dv = %.3f km/s' % dv

            from scipy.constants import c
            c = c / 1e3
            dwlog = -np.log10(1 - dv / c)
            dw = dwlog * (w2 - w1) / (np.log10(w2) - np.log10(w1))

        else:
            raise ValueError('Need w1, w2 and dv!')

        print 'User input w1 = {}, w2 = {}, dv = {}, nw = INDEF'.format(w1,w2,dv)

        iraf.dispcor.setParam('w1',w1) # Starting wavelength
        iraf.dispcor.setParam('w2',w2) # Ending wavelength
        iraf.dispcor.setParam('nw','INDEF') # Number of output pixels
        iraf.dispcor.setParam('dw',dw) # Wavelength interval per pixel

    iraf.dispcor()

    return None


def telluric_correction(targetdir,telluricdir,stype):

    """
    Removes telluric lines from 1D spectrum

    Assumes both science and telluric spectrum have been extracted and
    wavelength calibrated, i.e. file dimcomb.ms.fits exists in telluric
    directory.

    """

    print 'Target directory is ' + targetdir

    if os.path.exists( os.path.join(targetdir,'dimcomb.ms.fits') ):
        print "Wavelength calibrated target spectrum 'dimcomb.ms.fits' exists"

    print 'Telluric directory is ' + telluricdir

    if os.path.exists( os.path.join(telluricdir,'dimcomb.ms.fits') ):
        print "Wavelength calibrated telluric spectrum 'dimcomb.ms.fits' exists"

    print 'Generating black-body spectrum...'

    print 'Telluric star type ' + stype
    # from this site http://www.gemini.edu/sciops/instruments/nir/photometry/temps_colors.txt
    if stype == 'A0V': bbtemp = 9480.0
    if stype == 'A1V': bbtemp = 9230.0
    if stype == 'A2V': bbtemp = 8810.0
    if stype == 'A3V': bbtemp = 8270.0
    if stype == 'A4V': bbtemp = 8200.0
    if stype == 'A5V': bbtemp = 8160.0

    print 'Fitting with blackbody, temperature = ' + str(bbtemp) + 'K'

    iraf.noao(_doprint=0)
    iraf.artdata (_doprint=0)

    if os.path.exists( os.path.join(telluricdir,'blackbody.fits') ):
        os.remove( os.path.join(telluricdir,'blackbody.fits') )

    iraf.mk1dspec.setParam('input', os.path.join(telluricdir,'blackbody.fits') )
    iraf.mk1dspec.setParam('title','blackbody')
    iraf.mk1dspec.setParam('ncols',1024)
    iraf.mk1dspec.setParam('wstart',13900)
    iraf.mk1dspec.setParam('wend',24000)
    iraf.mk1dspec.setParam('temperature',bbtemp)

    iraf.mk1dspec()

    print 'Generated blackbody spectrum'

    # Divide telluric star spectrum by black-body

    if os.path.exists( os.path.join(telluricdir,'tdimcomb.ms.fits') ):
        os.remove( os.path.join(telluricdir,'tdimcomb.ms.fits') )

    iraf.onedspec(_doprint=0)

    """
    To divide by blackbody I want to turn off the cross-correlation - which I
    think would only work if both spectra had similar features. I don't want to scale
    the spectrum - which scales the airmass using Beer's Law - or shift it. If the temperature
    is correct I should be able to simply divide. If not then I should change
    the temperature
    """

    iraf.telluric.setParam('input',os.path.join(telluricdir,'dimcomb.ms.fits') ) # List of input spectra to correct
    iraf.telluric.setParam('output',os.path.join(telluricdir,'tdimcomb.ms.fits') ) # List of output corrected spectra
    iraf.telluric.setParam('cal',os.path.join(telluricdir,'blackbody.fits') ) # List of telluric calibration spectra
    iraf.telluric.setParam('answer','yes') # Search interactively?
    iraf.telluric.setParam('xcorr', 'no') # Cross correlate for shift?
    iraf.telluric.setParam('tweakrms', 'no') # Twak to minise rms?
    iraf.telluric.setParam('interactive', 'yes') # Interactive?
    iraf.telluric.setParam('threshold',0.0)  # Threshold for calibration
    iraf.telluric.setParam('offset',1) # Displayed offset between spectra
    iraf.telluric.setParam('sample','15000:18000,19700:23800')
    iraf.telluric.setParam('dshift',5.0)
    iraf.telluric.setParam('smooth',3.0)

    iraf.telluric()

    """
    When your calibration spectrum has zero or negative intensity values,
    you have to set the "threshold" parameter accordingly. As explained in
    the help page for the TELLURIC task, you can think of the "threshold"
    value as the minimum intensity value TELLURIC will accept from your
    calibration spectra. Any intensity value lower than the threshold value
    will be replaced by the threshold.

    """

    """
    I've turned cross-correlation off, since I don't really understand it.
    Tweak is on, but doesn't seem to do that much. Still not sure if it's understanding
    the airmass
    """

    print 'Now correcting target spectrum...'

    if os.path.exists( os.path.join(targetdir,'tdimcomb+bkgd.ms.fits') ):
        os.remove( os.path.join(targetdir,'tdimcomb+bkgd.ms.fits') )

    iraf.telluric.setParam('input',os.path.join(targetdir,'dimcomb+bkgd.ms.fits') ) # List of input spectra to correct
    iraf.telluric.setParam('output',os.path.join(targetdir,'tdimcomb+bkgd.ms.fits') ) # List of output corrected spectra
    iraf.telluric.setParam('cal',os.path.join(telluricdir,'tdimcomb.ms.fits') ) # List of telluric calibration spectra
    iraf.telluric.setParam('answer','yes') # Search interactively?
    iraf.telluric.setParam('threshold',0.0)
    iraf.telluric.setParam('xcorr', 'no') # Cross correlate for shift?
    iraf.telluric.setParam('tweakrms', 'yes') # Tweak to minise rms?
    iraf.telluric.setParam('interactive', 'yes') # Interactive?
    iraf.telluric.setParam('offset',6) # Displayed offset between spectra

    hdulist = fits.open(os.path.join(targetdir,'imcomb.ms.fits'))
    hdr = hdulist[0].header
    hdulist.close()

    iraf.telluric.setParam('airmass',hdr['AIRMASS'])

    iraf.telluric()


    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/telluric_correction.par') )
t = iraf.IrafTaskFactory(taskname="telluric_correction", value=parfile, function=telluric_correction)


def liris_reduce(targetdir,
                 lcpixmap,
                 reduceflat,
                 flatcorr,
                 skysub,
                 imshift,
                 imcomb,
                 extract,
                 interactive,
                 wavcab,
                 wavsol,
                 islog,
                 w1,
                 w2,
                 npix,
                 dw,
                 dv,
                 tellurcorr):

    """

    Does all reduction steps up to extraction of spectra

    """

    if uuid.getnode() == 246562420504605:
        basedir = '/data/lc585/WHT_20150331/OBS/'
    elif uuid.getnode() == 176509009147584:
        basedir = '/home/liam/data/lc585/WHT_20150331/OBS'

    fname = '/home/lc585/Dropbox/IoA/WHT_Proposal_2015a/Iraf/obsinfo.txt'
    t = Table.read(fname,format='ascii',guess=False,delimiter=',')

    flatdir = t['Flat'][ t['Name'] == targetdir].data[0]

    flatd = os.path.join(basedir,flatdir)
    flatnewdir = os.path.join(flatd,'Reduced')

    if reduceflat == 'yes':

        if not os.path.exists( flatnewdir ):
            os.makedirs( flatnewdir )

        for f in glob.glob( os.path.join(flatd,'*.fit') ):
            if not os.path.exists( os.path.join(flatnewdir,f.replace(flatd + '/','')) ):
                shutil.copy(f, flatnewdir)

        liris_pixmap(flatnewdir)
        combine_flats(flatnewdir)
        normalise_flats(flatnewdir)


    d = os.path.join(basedir,targetdir)
    newdir = os.path.join(d,'Reduced')

    if not os.path.exists( newdir ):
        os.makedirs( newdir )

    for f in glob.glob( os.path.join(d,'*.fit') ):
        if not os.path.exists( os.path.join(newdir,f.replace(d + '/','')) ):
            shutil.copy(f, newdir)

    goodflag = sort_frames(newdir)

    if lcpixmap == 'yes':

        liris_pixmap(newdir)

    if flatcorr == 'yes':

        flat_correction(newdir,flatnewdir)

    if skysub == 'yes':

        sky_subtraction(newdir)

    if imshift == 'yes':

        image_shift(newdir)

    if imcomb == 'yes':

        image_combine(newdir)

    if extract == 'yes':

        arcspec = t['Arc'][ t['Name'] == targetdir].data[0]
        arcspec = os.path.join( basedir, arcspec, 'imcomb.fits' )

        line = t['Line'][ t['Name'] == targetdir].data[0]
        trace = t['Trace'][ t['Name'] == targetdir].data[0]
        refspec = t['Telluric'][ t['Name'] == targetdir].data[0]
        ylevel = t['ylevel'][ t['Name'] == targetdir].data[0]
        t_nsum = t['t_nsum'][ t['Name'] == targetdir].data[0]

        t_step = t_nsum

        if t['Extract'][ t['Name'] == targetdir].data[0] == 'yes':
            extract_spectrum(targetdir,
                             trace,
                             arcspec,
                             refspec,
                             t_nsum,
                             t_step,
                             line,
                             ylevel,
                             interactive)

        else:

            print 'Extract = no'

    if wavcab == 'yes':

        wavelength_calibration(newdir)

    if wavsol == 'yes':

        wavelength_solution(newdir,
                            islog,
                            w1,
                            w2,
                            npix,
                            dw,
                            dv)




    if tellurcorr == 'yes':

        tellurdir = t['Telluric'][ t['Name'] == targetdir].data[0]
        tellurd = os.path.join(basedir,tellurdir)
        tellurnewdir = os.path.join(tellurd,'Reduced')

        stype = t['TelluricType'][ t['Name'] == targetdir].data[0]
        telluric_correction(newdir,tellurnewdir,stype)


    return None

parfile = iraf.osfn(os.path.join(homedir,'Dropbox/IoA/WHT_Proposal_2015a/Iraf/liris_reduce.par') )
t = iraf.IrafTaskFactory(taskname="liris_reduce", value=parfile, function=liris_reduce)
