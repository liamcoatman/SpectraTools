# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:47:44 2015

@author: lc585
"""

def BlackHoleMass(name, z, p, hdatmag, ebv):

    fwhm_ha = p['l0_fwhm'].value
    fwhm_hb = 1.07 * 1e3 * np.power(fwhm_ha*1e-3,1.03)

    plslp1 = 0.46
    plslp2 = 0.03
    plbrk = 2822.0
    bbt = 1216.0
    bbflxnrm = 0.24
    elscal = 0.71
    scahal = 0.86
    galfra = 0.31
    ebv = ebv
    imod = 18.0


    with open('/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit/input.yml', 'r') as f:
        parfile = yaml.load(f)

    fittingobj = qsrload(parfile)

    lin = fittingobj.get_lin()
    galspc = fittingobj.get_galspc()
    ext = fittingobj.get_ext()
    galcnt = fittingobj.get_galcnt()
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax()
    wavlen_rest = fittingobj.get_wavlen()
    ztran = fittingobj.get_ztran()
    lyatmp = fittingobj.get_lyatmp()
    lybtmp = fittingobj.get_lybtmp()
    lyctmp = fittingobj.get_lyctmp()
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}
    cosmo = cd.set_omega_k_0(cosmo)
    flxcorr = np.array( [1.0] * len(wavlen_rest) )

    params = Parameters()
    params.add('plslp1', value = plslp1)
    params.add('plslp2', value = plslp2)
    params.add('plbrk', value = plbrk)
    params.add('bbt', value = bbt)
    params.add('bbflxnrm', value = bbflxnrm)
    params.add('elscal', value = elscal)
    params.add('scahal', value = scahal)
    params.add('galfra', value = galfra)
    params.add('ebv', value = ebv)
    params.add('imod', value = imod)

    wavlen, flux = qsrmod(params,
                          parfile,
                          wavlen_rest,
                          z,
                          lin,
                          galspc,
                          ext,
                          galcnt,
                          ignmin,
                          ignmax,
                          ztran,
                          lyatmp,
                          lybtmp,
                          lyctmp,
                          whmin,
                          whmax,
                          cosmo,
                          flxcorr)

    zromag = fittingobj.get_zromag()[7]
    bp = fittingobj.get_bp()[7]
    dlam = fittingobj.get_dlam()[7]

    def qsrmodresid(p,
                    hdatmag,
                    flux,
                    wavlen,
                    zromag,
                    bp,
                    dlam):

        newflux = p['norm'].value * flux
        spc = interp1d(wavlen, newflux, bounds_error=False, fill_value=0.0)

        sum1 = np.sum( bp[1] * spc(bp[0]) * bp[0] * dlam)
        sum2 = np.sum( bp[1] * bp[0] * dlam)
        flxlam = sum1 / sum2
        flxlam = flxlam + 1e-200
        ftrmag = (-2.5 * np.log10(flxlam)) - zromag
        return [hdatmag - ftrmag]

    qsrmodresid_p = partial(qsrmodresid,
                            hdatmag=hdatmag,
                            flux=flux,
                            wavlen=wavlen,
                            zromag=zromag,
                            bp=bp,
                            dlam=dlam)

    p = Parameters()
    p.add('norm', value = 1e-17)

    result = minimize(qsrmodresid_p, p, method='leastsq')

    # Okay now need luminosity at 5100A in rest frame

    idx5100 = np.argmin( np.abs( (wavlen / (1.0 + z)) - 5100.))

    # Flux density in erg/cm2/s/A
    f5100 =   p['norm'].value * flux[idx5100]

    f5100 = f5100 * (u.erg / u.cm / u.cm / u.s / u.AA)

    lumdist = cosmoWMAP.luminosity_distance(z).to(u.cm)

    # Monochromatic luminosity at 5100A
    l5100 = f5100 * (1 + z) * 4 * math.pi * lumdist**2

    l5100 = l5100 * 5100.0 * (u.AA)

    # Black hole mass in units of solar mass from Banerji+12
    MBH = np.power(10,6.91) * np.power(fwhm_hb*1e-3,2) * np.power(l5100.value * 1e-44 , 0.5)
    Ledd = 3.2e4 * MBH # Eddington luminosity in units of solar luminosity
    Lbol = 7 * l5100 # Assume bolometric luminosity is 7L5100 (Bestergaard & Peterson 2006) in units erg/s
    Lbol = Lbol / 3.846e33 # in units of solar luminosity

    return np.log10(MBH), Lbol / Ledd, l5100