# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:13:56 2015

@author: lc585

Fit emission line with model.

"""
from __future__ import division

import numpy as np
from rebin_spectra import rebin_spectra
import astropy.units as u
from lmfit.models import GaussianModel, LorentzianModel, PowerLawModel, ConstantModel
from lmfit import minimize, Parameters, fit_report, Model
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import os
import cPickle as pickle
from scipy.stats import norm
from scipy.ndimage.filters import median_filter
from palettable.colorbrewer.qualitative import Set2_5
from time import gmtime, strftime
from astropy.cosmology import WMAP9 as cosmoWMAP
import math
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import interp1d 

# Simple mouse click function to store coordinates
def onclick(event):

    global ix
    
    ix = event.xdata

    coords.append(ix)

    if len(coords) % 2 == 0:
        print '[{0:.0f}, {1:.0f}]'.format(coords[-2], coords[-1])  
        # fig.canvas.mpl_disconnect(cid)
        
    return None 

class line_props(object):
    
    def __init__(self, 
                 name, 
                 peak,
                 fwhm ,
                 median,
                 sigma,
                 redchi,
                 eqw):

        self.name = name
        self.peak = peak 
        self.fwhm = fwhm
        self.median = median 
        self.sigma = sigma 
        self.redchi = redchi
        self.eqw = eqw 
      
def resid(p=None, x=None, model=None, data=None, sigma=None, **kwargs):

    mod = model.eval(params=p, x=x, **kwargs)
    
    if data is not None:
        resids = mod - data

        if sigma is not None:
            weighted = np.sqrt(resids ** 2 / sigma ** 2)
            
            return weighted
        else:
            return resids
    else:
        return mod



def wave2doppler(w, w0):

    """
    function uses the Doppler equivalency between wavelength and velocity
    """
    w0_equiv = u.doppler_optical(w0)
    w_equiv = w.to(u.km/u.s, equivalencies=w0_equiv)

    return w_equiv

def doppler2wave(v, w0):

    """
    function uses the Doppler equivalency between wavelength and velocity
    """
    w0_equiv = u.doppler_optical(w0)
    w_equiv = v.to(u.AA, equivalencies=w0_equiv)

    return w_equiv

def plot_fit(wav=None,
             flux=None,
             err=None,
             pars=None,
             mod=None,
             out=None,
             plot_savefig=None,
             plot_title='',
             save_dir=None,
             maskout=None,
             z=0.0,
             w0=6564.89*u.AA,
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=None,
             line_region=[-10000,10000]*(u.km/u.s),
             mask_negflux = True):


    if plot_region.unit == (u.km/u.s):
        plot_region = doppler2wave(plot_region, w0)

    # plotting region
    plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])

    # Transform to doppler shift
    vdat = wave2doppler(wav, w0)

    xdat = wav[plot_region_inds]
    vdat = vdat[plot_region_inds]
    ydat = flux[plot_region_inds]
    yerr = err[plot_region_inds]

    plt.rc('axes', color_cycle=Set2_5.mpl_colors) 
    fig = plt.figure(figsize=(6,18))

    fit = fig.add_subplot(5,1,1)
    fit.set_xticklabels( () )
    residuals = fig.add_subplot(5,1,2)
    
    eb = fig.add_subplot(5,1,3)
    residuals.set_xticklabels( () )

    fs = fig.add_subplot(5,1,5)

    fit.scatter(vdat.value, ydat, edgecolor='None', s=15, alpha=0.9, facecolor='black')

    # Mark continuum fitting region
    # Doesn't make sense to transform to wave and then back to velocity but I'm being lazy.
    # Check if continuum is given in wavelength or doppler units
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    # Region where equivalent width etc. calculated.
    integrand = lambda x: mod.eval(params=pars, x=np.array(x))
    func_max = np.max(integrand(vdat))
    
    eb.axvline(line_region[0].value, color='black', linestyle='--')
    eb.axvline(line_region[1].value, color='black', linestyle='--')

    fit.axvline(line_region[0].value, color='black', linestyle='--')
    fit.axvline(line_region[1].value, color='black', linestyle='--')

    fs.axvline(doppler2wave(line_region[0], w0).value, color='black', linestyle='--')
    fs.axvline(doppler2wave(line_region[1], w0).value, color='black', linestyle='--')

    fit.axhline(0, color='black', linestyle='--')
    fs.axhline(0, color='black', linestyle='--')
    eb.axhline(0, color='black', linestyle='--')
    
    # Mark fitting region
    fr = wave2doppler(fitting_region, w0)

    # set y axis scale
    fit.set_ylim(-0.3*func_max, 1.3*func_max)

    # Mask out regions
    xdat_masking = np.arange(xdat.min().value, xdat.max().value, 0.05)*(u.AA)
    vdat_masking = wave2doppler(xdat_masking, w0)

    mask = (vdat_masking.value < fr.value[0]) | (vdat_masking.value > fr.value[1])

    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            for item in maskout:
                mask = mask | ((vdat_masking.value > item.value[0]) & (vdat_masking.value < item.value[1]))

        elif maskout.unit == (u.AA):

            for item in maskout:
                xmin = item.value[0] / (1.0 + z)
                xmax = item.value[1] / (1.0 + z)
                mask = mask | ((xdat_masking.value > xmin) & (xdat_masking.value < xmax))

    vdat1_masking = ma.array(vdat_masking.value)
    vdat1_masking[mask] = ma.masked 

    for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
        fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='moccasin')
        residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='moccasin')
        eb.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='moccasin')  
        fs.axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, alpha=0.4, color='moccasin')

    # Now do continuum regions, which is now in wavelength units 
    xdat_masking = np.arange(xdat.min().value, xdat.max().value, 0.05)*(u.AA)
    vdat_masking = wave2doppler(xdat_masking, w0)

    mask = (xdat_masking.value < continuum_region[0][0].value) | \
           ((xdat_masking.value > continuum_region[0][1].value) &  (xdat_masking.value < continuum_region[1][0].value))  | \
           (xdat_masking.value > continuum_region[1][1].value)

    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            for item in maskout:
                mask = mask | ((vdat_masking.value > item.value[0]) & (vdat_masking.value < item.value[1]))

        elif maskout.unit == (u.AA):

            for item in maskout:
                xmin = item.value[0] / (1.0 + z)
                xmax = item.value[1] / (1.0 + z)
                mask = mask | ((xdat_masking.value > xmin) & (xdat_masking.value < xmax))


    vdat1_masking = ma.array(vdat_masking.value)
    vdat1_masking[mask] = ma.masked 

    for item in ma.extras.flatnotmasked_contiguous(vdat1_masking):
        fit.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')
        residuals.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')
        eb.axvspan(vdat1_masking[item].min(), vdat1_masking[item].max(), alpha=0.4, color='powderblue')  
        fs.axvspan(doppler2wave(vdat1_masking[item].min()*(u.km/u.s), w0).value, doppler2wave(vdat1_masking[item].max()*(u.km/u.s), w0).value, alpha=0.4, color='powderblue')    

    line, = fit.plot(np.sort(vdat.value), resid(pars, np.sort(vdat.value), mod), color='black', lw=2)

    plotting_limits = wave2doppler(plot_region, w0)
    fit.set_xlim(plotting_limits[0].value, plotting_limits[1].value)

    # residuals.errorbar(vdat.value, (ydat - resid(pars, vdat.value, mod)) / yerr, yerr=1, linestyle='', alpha=0.4)
    # residuals.plot(vdat.value, (ydat - resid(pars, vdat.value, mod)) / yerr, color='black', lw=1)
    residuals.scatter(vdat.value, (ydat - resid(pars, vdat.value, mod)) / yerr, alpha=0.9, edgecolor='None', s=15, facecolor='black')
    # residuals.plot(vdat.value, median_filter((ydat - resid(pars, vdat.value, mod)) / yerr, 3.0), color='black')

    residuals.axhline(0.0, color='black', linestyle='--')

    residuals.set_ylim(-8,8)
    residuals.set_xlim(fit.get_xlim())

    # plot model components
    i, j = 0, 0
    for key, value in pars.valuesdict().iteritems():
        if key.startswith('g'):
            i += 1
        if key.startswith('l'):
            j += 1

    ngaussians, nlorentzians = int(float(i) / 4.0), int(float(j) / 4.0)

    if ngaussians > 1:

        for i in range(ngaussians):
            comp_mod = GaussianModel()
            comp_p = comp_mod.make_params()

            for key, v in comp_p.valuesdict().iteritems():
                comp_p[key].value = pars['g' + str(i) + '_' + key].value

            fit.plot( np.sort(vdat.value), comp_mod.eval(comp_p, x=np.sort(vdat.value)) )

    if nlorentzians > 1:

        for i in range(nlorentzians):
            comp_mod = LorentzianModel()
            comp_p = comp_mod.make_params()

            for key, v in comp_p.valuesdict().iteritems():
                comp_p[key].value = pars['l' + str(i) + '_' + key].value

            fit.plot( np.sort(vdat.value), comp_mod.eval(comp_p, x=np.sort(vdat.value)) )


    #################################################

    
    eb.errorbar(vdat.value, ydat, yerr=yerr, linestyle='', alpha=0.5, color='grey')
    eb.plot(np.sort(vdat.value), resid(pars, np.sort(vdat.value), mod), color='black', lw=2)
    eb.set_xlim(fit.get_xlim())
    eb.set_ylim(fit.get_ylim())

    fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    eb.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=10)
    eb.set_ylabel(r'F$_\lambda$', fontsize=12)
    residuals.set_ylabel("Residual")

    #######################################

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])
    xdat = wav[fitting]
    vdat = wave2doppler(xdat, w0)
    ydat = flux[fitting]
    yerr = err[fitting]
    hg = fig.add_subplot(5,1,4)
    hg.hist((ydat - resid(pars, vdat.value, mod)) / yerr, bins=np.arange(-5,5,0.25), normed=True, edgecolor='None', facecolor='lightgrey')
    x_axis = np.arange(-5, 5, 0.001)
    hg.plot(x_axis, norm.pdf(x_axis,0,1), color='black', lw=2)

    #########################################

    fs.plot(wav, median_filter(flux,5.0), color='black', lw=1)
    fs.set_xlim(wav.min().value, wav.max().value)
    fs.set_ylim(-1*func_max, 2*func_max)


    ########################################################

    """
    If any flux values are negative could cause problems in fit.
    Highlight these by coloring points 
    """
    if mask_negflux:

        xdat = wav[fitting]
        vdat = wave2doppler(xdat, w0)
        ydat = flux[fitting]
        yerr = err[fitting]
    
        bad_points = (ydat < 0.0)
    
        xdat_bad = xdat[bad_points]
        vdat_bad = vdat[bad_points]
        ydat_bad = ydat[bad_points]
        yerr_bad = yerr[bad_points]
    
        fit.scatter(vdat_bad, ydat_bad, color='darkred', s=40, marker='x')
        residuals.scatter(vdat_bad, ydat_bad, color='darkred', s=40, marker='x')
        eb.scatter(vdat_bad, ydat_bad, color='darkred', s=40, marker='x')
        fs.scatter(xdat_bad, ydat_bad, color='darkred', s=40, marker='x')


    fig.tight_layout()

    # Call click func
    global coords
    coords = [] 
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    if plot_savefig is not None:
        fig.savefig(os.path.join(save_dir, plot_savefig))



    plt.show(1)
    plt.close()

    return None

def fit_line(wav,
             dw,
             flux,
             err,
             z=0.0,
             w0=6564.89*u.AA,
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=[6000,7000]*u.AA,
             line_region=[6400,6800]*u.AA,
             nGaussians=0,
             nLorentzians=1,
             maskout=None,
             verbose=True,
             plot=True,
             plot_savefig='something.png',
             plot_title='',
             save_dir=None,
             bkgd_median=True,
             fitting_method='leastsq',
             mask_negflux = True,
             mono_lum_wav = 5100 * u.AA,
             fit_model='MultiGauss',
             subtract_fe=False):

    """
    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    """

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA
    dw = dw / (1.0 + z)

    # Normalise spectrum
    spec_norm = 1.0 / np.median(flux)
    flux = flux * spec_norm 
    err = err * spec_norm 
 
    # index of the region we want to fit
    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])

    # fit power-law to continuum region
    # For civ we use median because more robust for small wavelength intervals. 
    # Ha we will fit to the data points since windows are much larger. 

    # index of region for continuum fit 
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)


    ################################################################################
    """
    Optical FeII template

    Broaden by convolution with Gaussian using astropy 
    Add to power-law with normalisation
    Allow shifting as well?

    Shen does all fits in logarithmic wavelength space. 
    Does this make a difference? 

    Convolve with width of Balmer line - might have to do iteratively 

    Either use FWHM from fit to Ha or leave convolution as a free paramter 
    See how this compares. 
    """

    if subtract_fe is True: 

        fname = '/home/lc585/SpectraTools/irontemplate.dat'
        fe_wav, fe_flux = np.genfromtxt(fname, unpack=True)
        fe_wav = 10**fe_wav * u.AA 
        fe_flux = fe_flux / np.median(fe_flux)
        fe_flux_interp = interp1d(fe_wav, 
                                  fe_flux, 
                                  bounds_error=False, 
                                  fill_value=0.0)
        
        def PseudoContinuum(x, 
                            amplitude, 
                            exponent, 
                            fe_norm, 
                            fe_flux_interp):
        
            return fe_norm * fe_flux_interp(x) + amplitude*x**exponent 
        
        bkgdmod = Model(PseudoContinuum, 
                        param_names=['amplitude','exponent','fe_norm'], 
                        independent_vars=['x']) 
    

        bkgdpars = bkgdmod.make_params() 
        bkgdpars['fe_norm'].value = 0.1
        bkgdpars['exponent'].value = 0.0
        bkgdpars['amplitude'].value = 1.0 / 5000.0  
    
    elif subtract_fe is False:

        bkgdmod = PowerLawModel()
        bkgdpars = bkgdmod.make_params()
        bkgdpars['exponent'].value = 1.0
        bkgdpars['amplitude'].value = 1.0 / 5000.0  


    #########################################################################################

   
    blue_inds = (wav > continuum_region[0][0]) & (wav < continuum_region[0][1])
    red_inds = (wav > continuum_region[1][0]) & (wav < continuum_region[1][1])   

    xdat_blue = wav[blue_inds]
    ydat_blue = flux[blue_inds]
    yerr_blue = err[blue_inds]
    vdat_blue = wave2doppler(xdat_blue, w0)

    xdat_red = wav[red_inds]
    ydat_red = flux[red_inds]
    yerr_red = err[red_inds]
    vdat_red = wave2doppler(xdat_red, w0)

    if maskout is not None:

        mask_blue = np.array([True] * len(xdat_blue))
        mask_red = np.array([True] * len(xdat_red))

        if maskout.unit == (u.km/u.s):
            
            for item in maskout:
                mask_blue[(vdat_blue > item[0]) & (vdat_blue < item[1])] = False
                mask_red[(vdat_red > item[0]) & (vdat_red < item[1])] = False

        elif maskout.unit == (u.AA):

            for item in maskout:
                mask_blue[(xdat_blue > (item[0] / (1.0 + z))) & (xdat_blue < (item[1] / (1.0 + z)))] = False
                mask_red[(xdat_red > (item[0] / (1.0 + z))) & (xdat_red < (item[1] / (1.0 + z)))] = False

        else:
            print "Units must be km/s or angstrom"

        xdat_blue = xdat_blue[mask_blue]
        ydat_blue = ydat_blue[mask_blue]
        yerr_blue = yerr_blue[mask_blue]
        vdat_blue = vdat_blue[mask_blue]
        xdat_red = xdat_red[mask_red]
        ydat_red = ydat_red[mask_red]
        yerr_red = yerr_red[mask_red]
        vdat_red = vdat_red[mask_red]
    
    if bkgd_median is True:

        xdat_cont = np.array( [np.median(xdat_blue.value), np.median(xdat_red.value)] )
        ydat_cont = np.array( [np.median(ydat_blue), np.median(ydat_red)] )

        if subtract_fe is True:

            resid(p=bkgdpars, x=xdat_cont, model=bkgdmod, sigma=None, fe_flux_interp=fe_flux_interp)
    
            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_cont, 'model':bkgdmod, 'data':ydat_cont, 'fe_flux_interp':fe_flux_interp},
                           method='leastsq')

        elif subtract_fe is False:

            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_cont, 'model':bkgdmod, 'data':ydat_cont},
                           method='leastsq')

    
    if bkgd_median is False:

        xdat_cont = np.concatenate((xdat_blue, xdat_red))
        ydat_cont = np.concatenate((ydat_blue, ydat_red))
        yerr_cont = np.concatenate((yerr_blue, yerr_red))

        if subtract_fe is True:

            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_cont.value, 'model':bkgdmod, 'data':ydat_cont, 'sigma':yerr_cont, 'fe_flux_interp':fe_flux_interp},
                           method='leastsq') 

        elif subtract_fe is False: 
            
            out = minimize(resid,
                           bkgdpars,
                           kws={'x':xdat_cont.value, 'model':bkgdmod, 'data':ydat_cont, 'sigma':yerr_cont},
                           method='leastsq') 


    if verbose:
        print fit_report(bkgdpars)


    ####################################################################################################################
    """
    Calculate flux at wavelength mono_lum_wav
    """

    # Not sure this works very well. Need to decide what most accurate way of getting monochromatic luminosity is. 

    # Be careful with 1e18 normalisation. Should normalise inside of rather than outside script!!  
    # mono_flux = bkgdmod.eval(params=bkgdpars, x=[mono_lum_wav.value])[0] / 1.e18 * (u.erg / u.cm / u.cm / u.s / u.AA)
    # lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)
    # mono_lum = mono_flux * (1.0 + z) * 4.0 * math.pi * lumdist**2 * mono_lum_wav 
    
    # print 'Monochomatic luminosity at {0} = {1}'.format(mono_lum_wav, np.log10(mono_lum.value)) 

    ######################################################################################################################

    # subtract continuum, define region for fitting
    xdat = wav[fitting]
    ydat = flux[fitting] - resid(bkgdpars, wav[fitting].value, bkgdmod)
    yerr = err[fitting]

    # Transform to doppler shift
    vdat = wave2doppler(xdat, w0)

    """
    Remember that this is to velocity shifted array

    Accepts units km/s or angstrom (observed frame)
    """

    if maskout is not None:

        if maskout.unit == (u.km/u.s):

            mask = np.array([True] * len(vdat))
            for item in maskout:
                print 'Not fitting between {0} and {1}'.format(item[0], item[1])
                mask[(vdat > item[0]) & (vdat < item[1])] = False


        elif maskout.unit == (u.AA):

            mask = np.array([True] * len(vdat))
            for item in maskout:
                vlims = wave2doppler(item / (1.0 + z), w0)
                print 'Not fitting between {0} ({1}) and {2} ({3})'.format(item[0], vlims[0], item[1], vlims[1])
                mask[(xdat > (item[0] / (1.0 + z))) & (xdat < (item[1] / (1.0 + z)))] = False

        else:
            print "Units must be km/s or angstrom"

        vdat = vdat[mask]
        ydat = ydat[mask]
        yerr = yerr[mask]
    
    if fit_model == 'MultiGauss':

        # Make model 
        bkgd = ConstantModel()
        mod = bkgd
        pars = bkgd.make_params()
        
        # A bit unnessesary, but I need a way to do += in the loop below
        pars['c'].value = 0.0
        pars['c'].vary = False

        for i in range(nGaussians):
            gmod = GaussianModel(prefix='g{}_'.format(i))
            mod += gmod
            pars += gmod.guess(ydat, x=vdat.value)
               
        for i in range(nGaussians):
            pars['g{}_center'.format(i)].value = 0.0
            pars['g{}_center'.format(i)].min = -5000.0
            pars['g{}_center'.format(i)].max = 5000.0
            pars['g{}_amplitude'.format(i)].min = 0.0
            #pars['g{}_sigma'.format(i)].min = 1000.0
            pars['g{}_sigma'.format(i)].max = 10000.0
        
        if nGaussians == 2:
            pars['g1_center'].set(expr = 'g0_center')
        if nGaussians == 3:
            pars['g1_center'].set(expr = 'g0_center')
            pars['g2_center'].set(expr = 'g0_center')

    elif fit_model == 'Ha':

        """
        Implement the Shen+15/11 fitting procedure, with narrow Ha/NeII
        """ 

        pass 

    elif fit_model == 'Hb':

        """
        From Shen+15/11

        Need to get Boroson and Green (1992) optical iron template, 
        which people seem to convolve with a Gaussian
    
        Model each [OIII] line with Gaussian, one for the core and the other for the blue wing.
        Decide whether to fix flux ratio 3:1
        Velocity offset and FWHM of narrow Hb tied to the core [OIII] components
        Upper limit of 12000 km/s on the narrow line FWHM
        Broad Hb modelled by single gaussian, or up to 3 Gaussians each with FWHM > 1200 km/s



        """
   
        mod = GaussianModel(prefix='hb_n_')  

        mod += GaussianModel(prefix='oiii_4959_n_')

        mod += GaussianModel(prefix='oiii_5007_n_')

        mod += GaussianModel(prefix='oiii_4959_b_')

        mod += GaussianModel(prefix='oiii_5007_b_')

        for i in range(nGaussians):

            mod += GaussianModel(prefix='hb_b_{}_'.format(i))  

        pars = mod.make_params() 
        print pars 

        pars['oiii_4959_n_amplitude'].value = 1000.0
        pars['oiii_5007_n_amplitude'].value = 1000.0
        pars['oiii_4959_b_amplitude'].value = 1000.0 
        pars['oiii_5007_b_amplitude'].value = 1000.0 
        pars['hb_n_amplitude'].value = 1000.0  

        pars['oiii_4959_n_center'].value = wave2doppler(4960.295*u.AA, w0).value 
        pars['oiii_5007_n_center'].value = wave2doppler(5008.239*u.AA, w0).value 
        pars['oiii_4959_b_center'].value = wave2doppler(4960.295*u.AA, w0).value 
        pars['oiii_5007_b_center'].value = wave2doppler(5008.239*u.AA, w0).value 
        pars['hb_n_center'].value = 0.0         

        pars['oiii_5007_n_center'].set(expr = 'hb_n_center+{}'.format(wave2doppler(5008.239*u.AA, w0).value))
        pars['oiii_4959_n_center'].set(expr = 'hb_n_center+{}'.format(wave2doppler(4960.295*u.AA, w0).value))

        pars['oiii_4959_n_sigma'].value = 500.0 
        pars['oiii_5007_n_sigma'].value = 500.0 
        pars['oiii_4959_b_sigma'].value = 1200.0 
        pars['oiii_5007_b_sigma'].value = 1200.0 
        pars['hb_n_sigma'].value = 500.0 

        pars['hb_n_sigma'].max = 1200.0
        pars['oiii_4959_n_sigma'].set(expr='hb_n_sigma')
        pars['oiii_5007_n_sigma'].set(expr='hb_n_sigma')

        for i in range(nGaussians):

            pars['hb_b_{}_amplitude'.format(i)].value = 1.0 
            pars['hb_b_{}_center'.format(i)].value = 0.0  
            pars['hb_b_{}_sigma'.format(i)].value = 1000.0  
            pars['hb_b_{}_sigma'.format(i)].min = 1200.0  



    if any(y < 0.0 for y in ydat):
        print 'Warning: Negative flux values in fitting region!'

    # Remove negative flux values which can mess up fit 
    if mask_negflux:

        posflux = (ydat > 0.0) 
    
        xdat = xdat[posflux] 
        ydat = ydat[posflux] 
        yerr = yerr[posflux]
        vdat = vdat[posflux] 

    out = minimize(resid,
                   pars,
                   args=(np.asarray(vdat), mod, ydat, yerr),
                   method = fitting_method)


    if verbose:
        print fit_report(pars)


    if save_dir is not None:
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        param_file = os.path.join(save_dir, 'my_params.txt')
        parfile = open(param_file, 'w')
        pars.dump(parfile)
        parfile.close()

        wav_file = os.path.join(save_dir, 'wav.txt')
        parfile = open(wav_file, 'wb')
        pickle.dump(wav, parfile, -1)
        parfile.close()

        flx_file = os.path.join(save_dir, 'flx.txt')
        parfile = open(flx_file, 'wb')
        pickle.dump(flux - resid(bkgdpars, wav.value, bkgdmod), parfile, -1)
        parfile.close()

        err_file = os.path.join(save_dir, 'err.txt')
        parfile = open(err_file, 'wb')
        pickle.dump(err, parfile, -1)
        parfile.close()

        fittxt = ''    
        fittxt += strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n \n'
        fittxt += plot_title + '\n \n'
        fittxt += r'Converged with chi-squared = ' + str(out.chisqr) + ', DOF = ' + str(out.nfree) + '\n \n'
        fittxt += fit_report(pars)
    

        with open(os.path.join(save_dir, 'fit.txt'), 'w') as f:
            f.write(fittxt) 

    # Calculate stats 
    integrand = lambda x: mod.eval(params=pars, x=np.array(x))

    # Calculate FWHM of distribution
    if line_region.unit == u.AA:
        line_region = wave2doppler(line_region, w0)

    dv = 1.0 # i think if this was anything else might need to change intergration.
    xs = np.arange(line_region.value[0], line_region[1].value, dv) 

    norm = np.sum(integrand(xs) * dv)
    pdf = integrand(xs) / norm
    cdf = np.cumsum(pdf)
    cdf_r = np.cumsum(pdf[::-1])[::-1] # reverse cumsum

    func_center = xs[np.argmax(pdf)] 
   
    half_max = np.max(pdf) / 2.0

    i = 0
    while pdf[i] < half_max:
        i+=1

    root1 = xs[i]

    i = 0
    while pdf[-i] < half_max:
        i+=1

    root2 = xs[-i]

    md = xs[np.argmin( np.abs( cdf - 0.5))]

    m = np.sum(xs * pdf * dv)

    v = np.sum( (xs-m)**2 * pdf * dv )
    sd = np.sqrt(v)

    # Equivalent width
    flux_line = integrand(xs)
    xs_wav = doppler2wave(xs*(u.km/u.s), w0)
    flux_bkgd = bkgdmod.eval(params=bkgdpars, x=xs_wav.value)
    f = (flux_line + flux_bkgd) / flux_bkgd
    eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
    eqw = np.nansum(eqw)

    # print plot_title, '{0:.2f},'.format(root2 - root1), '{0:.2f},'.format(sd), '{0:.2f},'.format(md), '{0:.2f},'.format(func_center), '{0:.2f},'.format(eqw), '{0:.2f}'.format(out.redchi)
    print plot_title 
    print 'peak_ha = {0:.2f}*(u.km/u.s),'.format(func_center)
    print 'fwhm_ha = {0:.2f}*(u.km/u.s),'.format(root2 - root1)
    print 'median_ha = {0:.2f}*(u.km/u.s),'.format(md)
    print 'sigma_ha = {0:.2f}*(u.km/u.s),'.format(sd)
    print 'chired_ha = {0:.2f},'.format(out.redchi)
    print 'eqw_ha = {0:.2f}*u.AA,'.format(eqw)

    if save_dir is not None:

        with open(os.path.join(save_dir, 'fit.txt'), 'a') as f:
            f.write('\n \n')
            f.write('Peak: {0:.2f} km/s \n'.format(func_center))
            f.write('FWHM: {0:.2f} km/s \n'.format(root2 - root1))
            f.write('Median: {0:.2f} km/s \n'.format(md))
            f.write('Sigma: {0:.2f} km/s \n'.format(sd))
            f.write('Reduced chi-squared: {0:.2f} \n'.format(out.redchi))
            f.write('EQW: {0:.2f} A \n'.format(eqw))      

        my_line_props = line_props(plot_title, func_center, root2 - root1, md, sd, out.redchi, eqw)  
        param_file = os.path.join(save_dir, 'line_props.txt')
        parfile = open(param_file, 'w')
        pickle.dump(my_line_props, parfile, -1)
        parfile.close()
      
    # """
    # Calculate S/N ratio per resolution element
    # """

    # vdat = wave2doppler(wav, w0)
    # vmin = md - 10000.0
    # vmax = md + 10000.0

    # i = np.argmin(np.abs(vdat.value - vmin))
    # j = np.argmin(np.abs(vdat.value - vmax))

    # fl = flux[i:j]
    # er = err[i:j]
    # w = wav[i:j]
    # dw1 = dw[i:j]

    # good = (er > 0) & ~np.isnan(fl)
    # if len(good.nonzero()[0]) == 0:
    #     print('No good data in this range!')

    # fl = fl[good]
    # er = er[good]
    # w = w[good]
    # dw1 = dw1[good]

    # snr = fl / er

  
    # # 33.02 is the A per resolution element I measured from the Arc spectrum
    # # print 'snr_ha = {0:.2f}'.format(np.mean(np.sqrt(33.02/dw1) * snr))



    if plot:
        plot_fit(wav=wav,
                 flux = flux - resid(bkgdpars, wav.value, bkgdmod),
                 err=err,
                 pars=pars,
                 mod=mod,
                 out=out,
                 plot_savefig = plot_savefig,
                 save_dir = save_dir,
                 maskout = maskout,
                 z=z,
                 w0=w0,
                 continuum_region=continuum_region,
                 fitting_region=fitting_region,
                 plot_region=plot_region,
                 plot_title=plot_title,
                 line_region=line_region,
                 mask_negflux = mask_negflux)


    return None 

