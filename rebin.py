import numpy as np 

def find_wa_edges(wa):

    """ Given wavelength bin centres, find the edges of wavelengh
    bins.

    Examples
    --------

    >>> print find_wa_edges([1, 2.1, 3.3, 4.6])
    [ 0.45  1.55  2.7   3.95  5.25]
    
    """
    wa = np.asarray(wa)
    edges = wa[:-1] + 0.5 * (wa[1:] - wa[:-1])
    edges = [2*wa[0] - edges[0]] + edges.tolist() + [2*wa[-1] - edges[-1]]

    return np.array(edges)

def rebin(wa0, fl0, er0, wa1, weighted=False):

    """ Rebins spectrum to a new wavelength scale generated using the
    keyword parameters.

    Returns the rebinned spectrum.

    Will probably get the flux and errors for the first and last pixel
    of the rebinned spectrum wrong.

    General pointers about rebinning if you care about errors in the
    rebinned values:

    1. Don't rebin to a smaller bin size.
    2. Be aware when you rebin you introduce correlations between
       neighbouring points and between their errors.
    3. Rebin as few times as possible.

    """

    # Note: 0 suffix indicates the old spectrum, 1 the rebinned spectrum.

    fl1 = np.zeros(len(wa1))
    er1 = np.zeros(len(wa1))
 
    # find pixel edges, used when rebinning
    edges0 = find_wa_edges(wa0)
    edges1 = find_wa_edges(wa1)

    widths0 = edges0[1:] - edges0[:-1]

    npts0 = len(wa0)
    npts1 = len(wa1)

    df = 0.
    de2 = 0.
    npix = 0    # number of old pixels contributing to rebinned pixel,
    
    j = 0                # index of rebinned array
    i = 0                # index of old array

    # sanity check
    if edges0[-1] < edges1[0] or edges1[-1] < edges0[0]:
        raise ValueError('Wavelength scales do not overlap!')
    
    # find the first contributing old pixel to the rebinned spectrum
    if edges0[i+1] < edges1[0]:
        # Old wa scale extends lower than the rebinned scale. Find the
        # first old pixel that overlaps with rebinned scale.
        while edges0[i+1] < edges1[0]:
            i += 1
        i -= 1
    elif edges0[0] > edges1[j+1]:
        # New rebinned wa scale extends lower than the old scale. Find
        # the first rebinned pixel that overlaps with the old spectrum
        while edges0[0] > edges1[j+1]:
            fl1[j] = np.nan
            er1[j] = np.nan
            j += 1
        j -= 1

    lo0 = edges0[i]      # low edge of contr. (sub-)pixel in old scale
    
    while True:
    
        hi0 = edges0[i+1]  # upper edge of contr. (sub-)pixel in old scale
        hi1 = edges1[j+1]  # upper edge of jth pixel in rebinned scale

        if hi0 < hi1:

            if er0[i] > 0:

                dpix = (hi0 - lo0) / widths0[i]
                
                if weighted:

                    # https://en.wikipedia.org/wiki/Inverse-variance_weighting

                    df += (fl0[i] / er0[i]**2) * dpix
                    de2 += dpix / er0[i]**2
                    npix += dpix / er0[i]**2
                
                else:

                    df += fl0[i] * dpix
                    de2 += er0[i]**2 * dpix
                    npix += dpix 

      
            lo0 = hi0
            i += 1
          
            if i == npts0:  break
        
        else:

            # We have all old pixel flux values that contribute to the
            # new pixel; append the new flux value and move to the
            # next new pixel.
            
            if er0[i] > 0:

                dpix = (hi1 - lo0) / widths0[i]

                if weighted:
           
                    df += (fl0[i] / er0[i]**2) * dpix
                    de2 += dpix / er0[i]**2
                    npix += dpix / er0[i]**2
    
                else:

                    df += fl0[i] * dpix
                    de2 += er0[i]**2 * dpix
                    npix += dpix 


            if npix > 0:
                
                # find total flux and error, then divide by number of
                # pixels (i.e. conserve flux density).
                
                fl1[j] = df / npix
               
                if weighted:

                    # Not 100% sure this is correct 
                    er1[j] = np.sqrt(1.0 / npix) 

                else:
                    # sum in quadrature and then divide by npix
                    # simply following the rules of propagation
                    # of uncertainty             
                    er1[j] = np.sqrt(de2) / npix  
            
            else:

                fl1[j] = np.nan
                er1[j] = np.nan
            
            df = 0.
            de2 = 0.
            npix = 0.
            lo0 = hi1
            j += 1
            
            if j == npts1:  break
        
    return wa1, fl1, er1  

if __name__ == '__main__':

       
     
        import sys
        sys.path.insert(0, "/home/lc585/Dropbox/IoA/nirspec/python_code") 
        from get_nir_spec import get_nir_spec

        fname = '/data/lc585/nearIR_spectra/data/LIRIS_redux/SDSSJ110454.73+095714.8.fits'
        wav0, dw, flux0, err0 = get_nir_spec(fname, 'LIRIS')

        wav1, flux1, err1 = rebin(wav0, flux0, err0, np.arange(wav0.min(), wav0.max(), 40.0), weighted=True)
    
        print np.nanmedian(err1)
        import matplotlib.pyplot as plt 

        fig, ax = plt.subplots() 
       
        # ax.errorbar(wav0, flux0, yerr=err0)
        ax.errorbar(wav1, flux1, yerr=err1)

        plt.show()
