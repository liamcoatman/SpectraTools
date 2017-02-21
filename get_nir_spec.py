from astropy.io import fits
import numpy as np 
import astropy.units as u 
import pandas as pd 
from SpectraTools.get_wavelength import get_wavelength

def get_nir_spec(path, instr):

    if instr == 'LIRIS':
        
         hdulist = fits.open(path)
         hdr = hdulist[0].header
         data = hdulist[0].data
         hdulist.close()
     
         wav, dw = get_wavelength(hdr)
         flux = data[0,:,:].flatten()
         err = data[-1,:,:].flatten()
    
    elif instr == 'XSHOOT':
    
        if 'rescale' in path:
            # quasar from XQ-100

            hdulist = fits.open(path)
            data = hdulist[1].data
            flux = data['FLUX'].flatten()
            err = data['ERR_FLUX'].flatten()
            wav = data['WAVE'].flatten()
            wav = (wav*u.nm).to(u.AA).value 
            dw = np.diff(wav)

        else:

            hdulist = fits.open(path)
            flux = hdulist[0].data
            err = hdulist[1].data
            wav = hdulist[2].data
            dw = np.diff(wav) 
    
    elif instr == 'FIRE':
    
        wav, flux, err = np.genfromtxt(path, unpack=True)
        dw = np.diff(wav) 
    
        flux = flux * 1.0e-17 
        err = err * 1.0e-17 
    
    elif instr == 'TRIPLE_S15':
    
        wav, flux, err = np.genfromtxt(path, unpack=True)
        dw = np.diff(wav)    
    
        flux = flux * 1.0e-17 
        err = err * 1.0e-17     
    
    elif instr == 'GNIRS':    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 
          
        flux = flux * 1.0e-17 
        err = err * 1.0e-17   
    
    
    elif instr == 'TRIPLE':    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 
          
        flux = flux * 1.0e-17 
        err = err * 1.0e-17              
    
    elif (instr == 'SINF_KK') | (instr == 'SINF'):    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 


    elif instr == 'ISAAC':    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 
          
        flux = flux * 1.0e-17 
        err = err * 1.0e-17             
    
    elif instr == 'NIRI':    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 
          
        flux = flux * 1.0e-17 
        err = err * 1.0e-17    
    
    elif instr == 'NIRSPEC':    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 
          
        flux = flux * 1.0e-17 
        err = err * 1.0e-17                             
    
    elif instr == 'SOFI_JH':    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 
          
        flux = flux * 1.0e-17 
        err = err * 1.0e-17      
    
    elif instr == 'SOFI_LC':    
    
        hdulist = fits.open(path)
        flux = hdulist[0].data
        err = hdulist[1].data
        wav = hdulist[2].data
        dw = np.diff(wav) 
          
        flux = flux * 1.0e-17 
        err = err * 1.0e-17            

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    k = df.loc[df.NIR_PATH == path, 'k'].values[0]

    if not np.isnan(k): 
        flux = flux / k 
        err = err / k 

    return wav, dw, flux, err 


if __name__ == '__main__':

    fname = '/data/lc585/nearIR_spectra/data/GNIRS_redux/SDSSJ211832.87+004219.0.fits'
    wav, dw, flux, err = get_nir_spec(fname, 'GNIRS') 