# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:11:10 2015

@author: lc585
"""

def stats(file1d, i, j, show=False):

    """Calculates statistics (mean, standard deviation (i.e. RMS), mean
    error, etc) of the flux between two wavelength points.

    Returns::

    mean flux, RMS of flux, mean error, SNR:
         SNR = (mean flux / RMS)
    """

    hdulist = fits.open(file1d)
    hdr = hdulist[0].header
    data = hdulist[0].data
    hdulist.close()

    flux = data[0,:,:].flatten()
    err = data[-1,:,:].flatten()

#    i = np.argmin( np.abs( wavlen - wa1))
#    j = np.argmin( np.abs( wavlen - wa2))

    fl = flux[i:j]
    er = err[i:j]

    good = (er > 0) & ~np.isnan(fl)
    if len(good.nonzero()[0]) == 0:
        print('No good data in this range!')
        return np.nan, np.nan, np.nan, np.nan

    fl = fl[good]
    er = er[good]

    mfl = fl.mean()
    std = fl.std()
    mer = er.mean()

    snr = fl / er
    snr = np.median(snr)

    if show:
        print 'mean %g, std %g, er %g, snr %g' % (mfl, std, mer, snr)


    return mfl, std, mer, snr

