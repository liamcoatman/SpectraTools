"""
Combine two spectra using inverse variance weighting
"""

from astropy.io import fits 
import matplotlib.pyplot as plt 
import numpy as np 

fname = '/data/lc585/NTT_Hennawi_Survey/Redux/Sept2011/H+K/Combine/NTTJ2003-3251_1.fits'
hdulist = fits.open(fname)
flux1 = hdulist[0].data
err1 = hdulist[1].data
wav1 = hdulist[2].data 

fname = '/data/lc585/NTT_Hennawi_Survey/Redux/Sept2011/H+K/Combine/NTTJ2003-3251_2.fits'
hdulist = fits.open(fname)
flux2 = hdulist[0].data
err2 = hdulist[1].data
wav2 = hdulist[2].data 

flux = ((flux1 / err1**2) + (flux2 / err2**2)) / ((1.0 / err1**2) + (1.0 / err2**2)) 
err = 1.0 / np.sqrt((1.0 / err1**2) + (1.0 / err2**2)) 
 
hdu1 = fits.PrimaryHDU(flux, header=hdulist[0].header)
hdu2 = fits.ImageHDU(err)
hdu3 = fits.ImageHDU(wav1)

hdulist = fits.HDUList([hdu1, hdu2, hdu3])

hdulist[0].header['COMMENT'] = 'This is inverse variance weighted sum of NTTJ2003-3251_1.fits and NTTJ2003-3251_2.fits'

# fig, ax = plt.subplots()
# # ax.plot(wav1, flux1)
# ax.errorbar(wav2, flux2, yerr=err2, color='red')
# ax.errorbar(wav1, flux, yerr=err, color='black')

# plt.show()

hdulist.writeto('/data/lc585/NTT_Hennawi_Survey/Redux/Sept2011/H+K/Combine/NTTJ2003-3251.fits', output_verify='fix', clobber=True) 
