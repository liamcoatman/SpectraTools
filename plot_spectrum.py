# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:38:42 2015

@author: lc585

Plot spectrum. Can deal with both linear and log-linear wavelength scales.

Plots emission lines, and if linear plots the 2D spectrum on top.

If want rest-frame spectrum then rest=True

This is un-tested and is almost definitely broken

To do:
Add way to mask bits out
Add way to give dictionary of line names and wavelengths to plot
Fix 2D spectrum
Remove spectrum specific values

"""

from SpectraTools.get_wavelength import get_wavelength
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
from rebin import rebin

def plot_spectrum(wav, 
                  flux,
                  er = None,
                  file2d = None,
                  z = 0.,
                  lheight = 1.0,
                  rest = False,
                  fout = None,
                  maskout = False,
                  rebin = 1,
                  weightedrebin = False,
                  sigmaclip = False):

    fig = plt.figure()
    sns.set_style("ticks")

    box1d = [0.1, 0.1, 0.8, 0.7]
    ax1d = fig.add_axes(box1d, xlabel = r'Wavelength $(\AA)$')

    if rest:
        wav = wav / (1 + z)

    wav, flux, er = rebin(wav,
                          flux,
                          er=er,
                          n=rebin,
                          weighted=weightedrebin)


    i1 = np.argmin( np.abs( wavelength - 15087.0 ))
    i2 = np.argmin( np.abs( wavelength - 17975.0 ))
    i3 = np.argmin( np.abs( wavelength - 19672.0 ))

    if sigmaclip:
        #######################################################################
        """
        Use a median filter to smooth out single-pixel deviations.
        Then use sigma-clipping to remove large variations between the
        actual and smoothed image.

        What about y_sigma?
        """

        spectrum_sm = medfilt(spectrum, 5)
        sigma = np.median(er)
        bad = np.abs(spectrum - spectrum_sm) / sigma > 8.0
        spectrum_cr = spectrum.copy()
        spectrum_cr[bad] = spectrum_sm[bad] # replace bad pixels with median values

    else:
        spectrum_cr = spectrum


    # Need to fix this to de-emphasise masked out regions

#    ax1d.plot(wavelength[i1:i2], spectrum_cr[i1:i2], color='black', lw=1)
#    ax1d.plot(wavelength[i3:], spectrum_cr[i3:], color='black', lw=1)
#    ax1d.plot(wavelength[:i1], spectrum_cr[:i1], lw=1, color= sns.xkcd_rgb["light grey"])
#    ax1d.plot(wavelength[i2:i3], spectrum_cr[i2:i3], lw=1, color= sns.xkcd_rgb["light grey"])


    linenames = [r'Ly$\alpha$',
                 r'CIV',
                 r'CIII',
                 r'H$\beta$',
                 r'OIII',
                 r'H$\alpha$',
                 '']

    lines = [1216.0,
             1549.0,
             1909.0,
             4861.0,
             5008.24,
             6562.8,
             4960.295]

    for i in range(len(lines)):
        # positions of spectral lines in the plot
        linewav = (1 + z) * lines[i]

        ax1d.axvline(linewav, color = 'grey', ls = '--')

        # annotate names:
        if nameSDSS is not None:
            s = fluxSDSS
        else:
            s = np.concatenate(  (spectrum_cr[i1:i2],spectrum_cr[i3:]) )

        height = lheight * max(s) + np.median(s) * (1. + (-1) ** i / 4.)
        ax1d.annotate(linenames[i], xy = (linewav + 50, height), color = 'r')

    if nameSDSS is not None:
        ax1d.set_xlim(min(wavelengthSDSS)-50,max(wavelength)+50)
    else:
        ax1d.set_xlim(min(wavelength)-50,max(wavelength)+50)


    if file2d is not None:

        # 2D spectrum

        hdulist = fits.open(file2d)
        img = hdulist[0].data
        hdulist.close()

        img = img[325:475,:]
        z1, z2, iteration = range_from_zscale(img)

        if hdr['CRVAL1'] < 10.0:

            newimg = np.zeros((np.shape(img)[0],len(wavelength)))
            wav2d = np.linspace(wavelength.min(),wavelength.max(),1024)

            for j in range(np.shape(img)[0]):
                f = interp1d( wav2d, img[j,:])
                newimg[j,:] = f(wavelength)

        height = 0.001 * img.shape[0]
        box2d = [0.1, 0.9 - height / 2, 0.8, height]
        ax2d = fig.add_axes(box2d, yticks = [])
        ax2d.imshow(img, aspect = 'auto', vmin = z1, vmax = z2, cmap = cm.Greys_r)
        setp( ax2d.get_xticklabels(), visible=False)


    ax1d.set_ylim( np.min( np.concatenate(  (spectrum_cr[i1:i2],spectrum_cr[i3:]) )) -  1.0 * np.std( np.concatenate(  (spectrum_cr[i1:i2],spectrum_cr[i3:]) ) ),
                   np.max( np.concatenate(  (spectrum_cr[i1:i2],spectrum_cr[i3:]) )) + 1.5 * np.std( np.concatenate(  (spectrum_cr[i1:i2],spectrum_cr[i3:]) ) ))

    if maskout == True:
        ax1d.add_patch(Rectangle((wavelength.min(), ax1d.get_ylim()[0]), 15087.0 - wavelength.min() , ax1d.get_ylim()[1] - ax1d.get_ylim()[0], facecolor='grey',edgecolor='None'))
        ax1d.add_patch(Rectangle((17975.0, ax1d.get_ylim()[0]), 19672.0 - 17975.0, ax1d.get_ylim()[1] - ax1d.get_ylim()[0], facecolor='grey',edgecolor='None'))



    if fout is not None:
        plt.savefig(fout)
    else:
        plt.show()

    plt.close()

    return None

def plot_spectrum_2d(name):

    fig, ax = plt.subplots()

    # fname = os.path.join('/data/lc585/WHT_20150331/spec_150311/',name+'_LR','imcomb.fit')
    fname = name

    if os.path.exists(fname):
        hdulist = fits.open(fname)
        data = hdulist[0].data
        hdulist.close()

        z1, z2, iteration = range_from_zscale(data)

        ax.imshow(data,cmap='gray',vmin=z1,vmax=z2)

        fig.gca().invert_yaxis()
        fig.tight_layout()

    else:

        ax.text(0.4,0.5,'No Spectrum')

    fig.savefig( fname.replace('imcombHR.fit', '2D_HR.png') )
    plt.close()

    return None