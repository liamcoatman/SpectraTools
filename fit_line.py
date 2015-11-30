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
from lmfit import minimize, Parameters, fit_report
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import os
import cPickle as pickle
from scipy.stats import norm
from scipy.ndimage.filters import median_filter
from palettable.colorbrewer.qualitative import Set2_5

def resid(p,x,model,data=None,sigma=None):

        mod = model.eval( params=p, x=x )

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
             maskout=None,
             z=0.0,
             w0=6564.89*u.AA,
             continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
             fitting_region=[6400,6800]*u.AA,
             plot_region=None,
             line_region=[-10000,10000]*(u.km/u.s)):


    # plotting region
    plot_region_inds = (wav > plot_region[0]) & (wav < plot_region[1])

    # Transform to doppler shift
    vdat = wave2doppler(wav, w0)

    xdat = wav[plot_region_inds]
    vdat = vdat[plot_region_inds]
    ydat = flux[plot_region_inds]
    yerr = err[plot_region_inds]

    plt.rc('axes', color_cycle=Set2_5.mpl_colors) 
    fig = plt.figure(figsize=(6,15))

    fit = fig.add_subplot(4,1,1)
    fit.set_xticklabels( () )
    residuals = fig.add_subplot(4,1,2)

   
    fit.scatter(vdat.value, ydat, edgecolor='None', s=15, alpha=0.9, facecolor='black')

    # Mark continuum fitting region
    # Doesn't make sense to transform to wave and then back to velocity but I'm being lazy.
    # Check if continuum is given in wavelength or doppler units
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    blue_cont = wave2doppler(continuum_region[0], w0)
    red_cont = wave2doppler(continuum_region[1], w0)

    fit.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    fit.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

    residuals.axvspan(blue_cont.value[0], blue_cont.value[1], color='grey', lw = 1, alpha = 0.2 )
    residuals.axvspan(red_cont.value[0], red_cont.value[1], color='grey', lw = 1, alpha = 0.2 )

    # Region where equivalent width etc. calculated.
    integrand = lambda x: mod.eval(params=pars, x=np.array(x))
    func_center = optimize.fmin(lambda x: -integrand(x) , 0)[0]
    
    fit.axvline(line_region[0].value, color='black', linestyle='--')
    fit.axvline(line_region[1].value, color='black', linestyle='--')

    # Mark fitting region
    fr = wave2doppler(fitting_region, w0)

    # set y axis scale
    ind_center = np.argmin(np.abs(vdat.value - func_center))
    fit.set_ylim(-0.2*ydat[ind_center], 1.2*ydat[ind_center])

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




    # plt.figtext(0.05,0.23,plot_title,fontsize=10)
    # plt.figtext(0.05,0.20,r"Converged with $\chi^2$ = " + str(out.chisqr) + ", DOF = " + str(out.nfree), fontsize=10)

    # figtxt = ''
    # for i in pars.valuesdict():
    #     figtxt += i + ' = {0} \n'.format( float('{0:.4g}'.format( pars[i].value)))

    # plt.figtext(0.1,0.18,figtxt,fontsize=10,va='top')


    eb = fig.add_subplot(4,1,3)
    eb.errorbar(vdat.value, ydat, yerr=yerr, linestyle='', alpha=0.5, color='grey')
    eb.plot(np.sort(vdat.value), resid(pars, np.sort(vdat.value), mod), color='black', lw=2)
    eb.set_xlim(fit.get_xlim())
    eb.set_ylim(fit.get_ylim())

    fit.set_ylabel(r'F$_\lambda$', fontsize=12)
    eb.set_xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=12)
    eb.set_ylabel(r'F$_\lambda$', fontsize=12)
    residuals.set_ylabel("Residual")

    #######################################

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])
    xdat = wav[fitting]
    vdat = wave2doppler(xdat, w0)
    ydat = flux[fitting]
    yerr = err[fitting]
    hg = fig.add_subplot(4,1,4)
    hg.hist((ydat - resid(pars, vdat.value, mod)) / yerr, bins=np.arange(-5,5,0.25), normed=True, edgecolor='None')
    x_axis = np.arange(-5, 5, 0.001)
    hg.plot(x_axis, norm.pdf(x_axis,0,1), color='black', lw=2)

    fig.tight_layout()

    if plot_savefig is not None:
        fig = fig.savefig(plot_savefig)


    plt.show()
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
             save_dir=None):

    """
    Velocity shift added to doppler shift to change zero point (can do if HW10
    redshift does not agree with Halpha centroid)

    Fiting and continuum regions given in rest frame wavelengths with
    astropy angstrom units.

    Maskout is given in terms of doppler shift

    """

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)
    wav = wav*u.AA
    dw = dw / (1.0 + z)

    # Check if continuum is given in wavelength or doppler units
    if continuum_region[0].unit == (u.km/u.s):
        continuum_region[0] = doppler2wave(continuum_region[0], w0)
    if continuum_region[1].unit == (u.km/u.s):
        continuum_region[1] = doppler2wave(continuum_region[1], w0)

    # index is true in the region where we fit the continuum
    continuum = ((wav > continuum_region[0][0]) & \
                 (wav < continuum_region[0][1])) | \
                 ((wav > continuum_region[1][0]) & \
                 (wav < continuum_region[1][1]))

    # index of the region we want to fit

    if fitting_region.unit == (u.km/u.s):
        fitting_region = doppler2wave(fitting_region, w0)

    fitting = (wav > fitting_region[0]) & (wav < fitting_region[1])

    # fit power-law to continuum region
    # For civ we use median because more robust for small wavelength intervals. Ha we will fit to the data points since windows are much larger. 

    xdat_cont = wav[continuum]
    ydat_cont = flux[continuum]
    yerr_cont = err[continuum]

    bkgdmod = PowerLawModel()
    bkgdpars = bkgdmod.make_params()
    bkgdpars['exponent'].value = 1.0
    bkgdpars['amplitude'].value = 1.0

    # don't know if its the best idea to minimize twice. 
    out = minimize(resid,
                   bkgdpars,
                   args=(xdat_cont.value, bkgdmod, ydat_cont, yerr_cont),
                   method='nelder')

    out = minimize(resid,
                   bkgdpars,
                   args=(xdat_cont.value, bkgdmod, ydat_cont, yerr_cont),
                   method='leastsq')

    if verbose:
        print fit_report(bkgdpars)


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

    for i in range (nLorentzians):
        lmod = LorentzianModel(prefix='l{}_'.format(i))
        mod += lmod
        pars += lmod.guess(ydat, x=vdat.value)

    for i in range(nGaussians):
        pars['g{}_center'.format(i)].value = 0.0
        pars['g{}_center'.format(i)].min = -5000.0
        pars['g{}_center'.format(i)].max = 5000.0
        pars['g{}_amplitude'.format(i)].min = 0.0
        #pars['g{}_sigma'.format(i)].min = 1000.0
        pars['g{}_sigma'.format(i)].max = 10000.0

    for i in range(nLorentzians):
        pars['l{}_center'.format(i)].value = 0.0
        pars['l{}_center'.format(i)].min = -10000.0
        pars['l{}_center'.format(i)].max = 10000.0
        pars['l{}_amplitude'.format(i)].min = 0.0

    # For Ha
    if nGaussians == 2:
        pars['g1_center'].set(expr = 'g0_center')
    if nGaussians == 3:
        pars['g1_center'].set(expr = 'g0_center')
        pars['g2_center'].set(expr = 'g0_center')

    # For Hb

    # print pars
    # pars['g0_center'].value = 0.0
    # pars['g0_center'].vary = False
    # pars['g1_center'].value = wave2doppler(4960.295*u.AA, w0).value
    # pars['g1_center'].vary = False
    # pars['g2_center'].value = wave2doppler(5008.239*u.AA, w0).value
    # pars['g2_center'].vary = False
    # pars['g2_amplitude'].expr = 'g1_amplitude * 3.0'
    # pars['g2_sigma'].max = pars['g0_sigma'].value
    # pars['g1_sigma'].expr = 'g2_sigma * 1.0'


    out = minimize(resid,
                   pars,
                   args=(np.asarray(vdat), mod, ydat, yerr),
                   method ='nelder')

    if verbose:
        print fit_report(pars)

    out = minimize(resid,
                   pars,
                   args=(np.asarray(vdat), mod, ydat, yerr),
                   method ='leastsq')



    # Calculate stats 
    integrand = lambda x: mod.eval(params=pars, x=np.array(x))

    # Calculate FWHM of distribution
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
    xs_wav = doppler2wave(xs*(u.km/u.s),w0)
    flux_bkgd = bkgdmod.eval(params=bkgdpars, x=xs_wav.value)
    f = (flux_line + flux_bkgd) / flux_bkgd
    eqw = (f[:-1] - 1.0) * np.diff(xs_wav.value)
    eqw = np.nansum(eqw)

    print plot_title
    print 'peak_ha = {0:.2f}*(u.km/u.s),'.format(func_center)
    print 'fwhm_ha = {0:.2f}*(u.km/u.s),'.format(root2 - root1)
    print 'median_ha = {0:.2f}*(u.km/u.s),'.format(md)
    print 'sigma_ha = {0:.2f}*(u.km/u.s),'.format(sd)
    print 'chired_ha = {0:.2f},'.format(out.redchi)
    print 'eqw_ha = {0:.2f}*u.AA,'.format(eqw)

    # Convert Scipy cov matrix to standard covariance matrix.
    # cov = out.covar*dof / out.chisqr

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

    # vdat = wave2doppler(wav, 4862.721*u.AA)
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

    # # fig, ax = plt.subplots()
    # # ax.plot(w,np.sqrt(33.02/dw) * snr)

    # # 33.02 is the A per resolution element I measured from the Arc spectrum
    # # print 'snr_hb = {0:.2f}'.format(np.mean(np.sqrt(33.02/dw1) * snr))

    # print '\n'
    # # plt.show()

    if plot:
        plot_fit(wav=wav,
                 flux = flux - resid(bkgdpars, wav.value, bkgdmod),
                 err=err,
                 pars=pars,
                 mod=mod,
                 out=out,
                 plot_savefig = plot_savefig,
                 maskout = maskout,
                 z=z,
                 w0=w0,
                 continuum_region=continuum_region,
                 fitting_region=fitting_region,
                 plot_region=plot_region,
                 plot_title=plot_title,
                 line_region=line_region)

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


    return xdat, ydat, yerr

###Testing###

if __name__ == '__main__':

    from get_spectra import get_liris_spec
    import matplotlib.pyplot as plt

    fname = '/data/lc585/WHT_20150331/html/SDSSJ1339+1515/dimcombLR+bkgd_v138.ms.fits'
    wavelength, dw, flux, err = get_liris_spec(fname)
#    wav, flux, err = rebin_spectra(wavelength,
#                                   flux,
#                                   er=err,
#                                   n=1,
#                                   weighted=False)




#    wav, dw, flux, err = get_boss_dr12_spec('SDSSJ133916.88+151507.6')

#    fit_line(wav,
#             flux,
#             err,
#             z=2.318977,
#             w0=np.mean([1548.202,1550.774])*u.AA,
#             velocity_shift=0.0*(u.km / u.s),
#             continuum_region=[[1445.,1465.]*u.AA,[1700.,1705.]*u.AA],
#             fitting_region=[1500.,1600.]*u.AA,
#             plot_region=[1440,1720]*u.AA,
#             nGaussians=0,
#             nLorentzians=1,
#             maskout=[[5107.3, 5138.2]]*u.AA,
#             verbose=True,
#             plot=True)

#maskout=[[5107.3, 5138.2]]*u.AA,
#maskout=[[-2064,-263]]*(u.km / u.s),

    out = fit_line(wavelength,
                   flux,
                   err,
                   z=2.318977,
                   w0=6564.89*u.AA,
                   maskout=[[4735.8, 4757.8]]*(u.AA),
                   continuum_region=[[6000.,6250.]*u.AA,[6800.,7000.]*u.AA],
                   plot_title='SDSSJ133916.88+151507.6',
                   plot_region=[6000,7000]*u.AA,
                   plot=False,
                   save_dir='/data/lc585/WHT_20150331/test/SDSSJ133916.88+151507.6/Ha/')



#    parfile = open('my_params.txt', 'w')
#    out['pars'].dump(parfile)
#    parfile.close()
#
#    parfile = open('my_params.txt', 'r')
#    params = Parameters()
#    params.load(parfile)
#    parfile.close()



#
#    p = Parameters()
#    p.set

#    with open('test.p', 'wb') as f:
#        pickle.dump(out, f, -1)



