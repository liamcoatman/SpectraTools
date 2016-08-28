import numpy as np
import astropy.units as u 
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import pandas as pd
import sys
import palettable
import os 
import matplotlib
from astroML.datasets import fetch_hogg2010test
import emcee
import corner
from astropy.io import fits 

def model(weights, comps, comps_wav, x):

    fl = weights['w1'].value * comps[:, 0]

    for i in range(comps.shape[1] - 1):
    	fl += weights['w{}'.format(i+2)].value * comps[:, i+1] 

    f = interp1d(comps_wav + weights['shift'].value, 
                 fl, 
                 bounds_error=False, 
                 fill_value=np.nan)

    return f(x)

def get_comp(i, weights, comps, comps_wav, x):

	f = interp1d(comps_wav + weights['shift'].value, 
                 weights['w{}'.format(i)].value * comps[:, i-1], 
                 bounds_error=False, 
                 fill_value=np.nan)

	return f(x)

  
def resid(weights=None, 
	      comps=None,
	      comps_wav=None,
          x=None, 
          data=None, 
          sigma=None, 
          **kwargs):

    if data is not None:
        
        resids = model(weights, comps, comps_wav, x) - data 

        if sigma is not None:

            weighted = np.sqrt(resids ** 2 / sigma ** 2)       

            return weighted

        else:

            return resids
    
    else:

        return model(weights, comps)   

def do_fit(wav,
           flux,
           err,
           z=0.0,
           name='QSO001',
           n_weights=12):

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)

    spec_norm = 1.0 / np.median(flux[(flux != 0.0) & ~np.isnan(flux)])
    flux = flux * spec_norm 
    err = err * spec_norm 


    # Read components 
    comps = np.genfromtxt('/data/vault/phewett/ICAtest/VivGals/CVilforth/QS2402_12c.comp')
    comps = comps[:, :n_weights]

    comps_wav = np.genfromtxt('/data/vault/phewett/ICAtest/VivGals/CVilforth/wav33005100.dat')

    fitting_region = (wav > np.max([comps_wav.min(), 4600.0])) & (wav < np.min([comps_wav.max(), 5100.0]))

    wav = wav[fitting_region]
    flux = flux[fitting_region]
    err = err[fitting_region]

    # clean data 
    mask = np.isnan(flux) | (err <= 0.0) 

    wav = wav[~mask]
    flux = flux[~mask]
    err = err[~mask]

    weights = Parameters() 

    weights.add('w1', value=0.1283, min=0.0)

    if n_weights >= 2:
    	weights.add('w2', value=0.0858, min=0.0)

    if n_weights >= 3:
        weights.add('w3', value=0.0869, min=0.0)

    if n_weights >= 4:
        weights.add('w4', value=0.1111, min=0.0)

    if n_weights >= 5:
        weights.add('w5', value=0.1179, min=0.0)

    if n_weights >= 6:
        weights.add('w6', value=0.0169, min=0.0)

    if n_weights >= 7:
        weights.add('w7', value=0.0061, min=0.0)

    if n_weights >= 8:
        weights.add('w8', value=0.0001)

    if n_weights >= 9:
        weights.add('w9', value=0.0002)

    if n_weights >= 10:
        weights.add('w10', value=-0.0006)

    if n_weights >= 11:
        weights.add('w11', value=-0.0001)

    if n_weights == 12:
        weights.add('w12', value=0.0001)

    weights.add('shift', value=10.0, vary=True)


    out = minimize(resid,
                   weights,
                   kws={'comps':comps,
                        'comps_wav':comps_wav,
                        'x':wav,  
                        'data':flux,
                        'sigma':err},
                   method='nelder',
                   options={'maxiter':1e6, 'maxfev':1e6} 
                   )        

    print out.message 

    fig, axs = plt.subplots(2, 1, figsize=(6,8)) 

    color = matplotlib.cm.get_cmap('Set2')
    color = color(np.linspace(0, 1, n_weights))
    
    axs[0].plot(wav, model(out.params, comps, comps_wav, wav), color='black', zorder=2)
    axs[0].plot(wav, flux, color='grey', alpha=0.4)
 
    for i in range(1, n_weights+1):
    	axs[0].plot(wav, get_comp(i, out.params, comps, comps_wav, wav), c=color[i-1])
 
    axs[0].set_ylim(-0.1 * np.max(model(out.params, comps, comps_wav, wav)), 
    	            1.1 * np.max(model(out.params, comps, comps_wav, wav)))


    fig.delaxes(axs[1])

    i = 0
    for key, value in out.params.valuesdict().iteritems():
    	print key + ' {0:.3f}'.format(value)
    	if i < n_weights:
        	fig.text(0.1, 0.45 - i*(0.4/13.0), key + ' {0:.3f}'.format(value), color=color[i])
        else:
        	fig.text(0.1, 0.45 - i*(0.4/13.0), key + ' {0:.3f}'.format(value), color='black')
    	i += 1
 


    # fig.savefig(os.path.join('/data/lc585/nearIR_spectra/linefits', name, 'Hb', 'mfica_fit.png')) 



    plt.show() 

def log_likelihood(weights, comps, comps_wav, x, y, dy):

    fl = weights[0] * comps[:, 0]+\
         weights[1] * comps[:, 1]+\
         weights[2] * comps[:, 2]+\
         weights[3] * comps[:, 3]+\
         weights[4] * comps[:, 4]+\
         weights[5] * comps[:, 5]+\
         weights[6] * comps[:, 6]

    # this nan might not be a good idea  
    f = interp1d(comps_wav,
                 fl, 
                 bounds_error=False, 
                 fill_value=np.nan)

    return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + (y - f(x))** 2 / dy** 2)

def log_prior(weights):

    if np.any(np.array(weights) < 0.0):
        return -np.inf 
    else:
        return 0.0

def log_posterior(weights, comps, comps_wav, x, y, dy):
    return log_prior(weights) + log_likelihood(weights, comps, comps_wav, x, y, dy)


def do_fit_mcmc(wav,
                flux,
                err,
                z=0.0,
                name='QSO001',
                n_weights=12):

    # Transform to quasar rest-frame
    wav =  wav / (1.0 + z)

    # Read components 
    comps = np.genfromtxt('/data/vault/phewett/ICAtest/VivGals/CVilforth/QS2402_12c.comp')
    comps = comps[:, :n_weights]

    comps_wav = np.genfromtxt('/data/vault/phewett/ICAtest/VivGals/CVilforth/wav33005100.dat')

    fitting_region = (wav > np.max([comps_wav.min(), 4700.0])) & (wav < np.min([comps_wav.max(), 5100.0]))

    wav = wav[fitting_region]
    flux = flux[fitting_region]
    err = err[fitting_region]

    # clean data 
    mask = np.isnan(flux) | (err <= 0.0) 

    wav = wav[~mask]
    flux = flux[~mask]
    err = err[~mask]

    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps
    
    ndim = 7  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 2000  # number of MCMC steps to take

    p0 = [0.1283, 0.0858, 0.0869, 0.1111, 0.1179, 0.0169, 0.0061] 
    std = 0.01*np.ones_like(p0)
    starting_guesses = emcee.utils.sample_ball(p0, std, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[comps, comps_wav, wav, flux, err])
    o = sampler.run_mcmc(starting_guesses, nsteps)

    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)

    weights = np.median(trace, axis=0)

    corner.corner(trace)

    fig, ax = plt.subplots(1, 1, figsize=(6,4)) 

    color = matplotlib.cm.get_cmap('Set2')
    color = color(np.linspace(0, 1, 10))

    fl = weights[0] * comps[:, 0]+\
         weights[1] * comps[:, 1]+\
         weights[2] * comps[:, 2]+\
         weights[3] * comps[:, 3]+\
         weights[4] * comps[:, 4]+\
         weights[5] * comps[:, 5]+\
         weights[6] * comps[:, 6]

    # this nan might not be a good idea  
    f = interp1d(comps_wav,
                 fl, 
                 bounds_error=False, 
                 fill_value=np.nan)

   
    ax.plot(wav, f(wav), color='black', zorder=2)
    ax.plot(wav, flux, color='grey', alpha=0.4)

    ax.set_ylim(-0.1 * np.max(f(wav)), 
    	        1.1 * np.max(f(wav)))

 
    for i in range(7):
    	fl = weights[i] * comps[:, i]
    	f = interp1d(comps_wav,
                     fl, 
                     bounds_error=False, 
                     fill_value=np.nan)
        ax.plot(wav, f(wav), c=color[i])
 



    print weights



    plt.show()





    return None 


if __name__ == '__main__':

    s = np.genfromtxt('/data/vault/phewett/LiamC/qso_hw10_template.dat')
    wav = s[:, 0]
    flux = s[:, 1]
    err = np.repeat(0.01, len(flux))
 

    # df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    # df = df[df.INSTR == 'FIRE']
    # row = df.ix[0]

    # fname = row.NIR_PATH 

    # wav, flux, err = np.genfromtxt(fname, unpack=True)
    # dw = np.diff(wav) 

     

    do_fit_mcmc(wav,
                flux,
                err,
                z=0.0,
                name='SDSS',
                n_weights=12)



