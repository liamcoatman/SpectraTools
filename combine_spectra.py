"""
Combine two spectra using inverse variance weighting
Untested
"""

import numpy as np 

def combine_spectra(flux1, err1, flux2, err2): 

    flux = ((flux1 / err1**2) + (flux2 / err2**2)) / ((1.0 / err1**2) + (1.0 / err2**2)) 
    err = 1.0 / np.sqrt((1.0 / err1**2) + (1.0 / err2**2)) 
 
    return flux, err 