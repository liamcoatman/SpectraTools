# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:13:13 2015

@author: lc585

Calculate Black Hole Mass using virial scaling relation

Inputs:

Which line. If Ha will convert to Hb.
FWHM in km/s
monochromatic luminosity in erg/s (should be at 5100A for Ha/Hb and 1350 for CIV)

Need to decide what best bolometric correction to use.

"""

import astropy.units as u
import numpy as np


def BHMass(line='Ha',
           fwhm=1000*(u.km/u.s),
           fwhm_err=10*(u.km/u.s),
           l5100=1.0e46*(u.erg/u.s),
           l5100_err=1.0e46*(u.erg/u.s),
           l1350=1.0e46*(u.erg/u.s),
           l1350_err=1.0e46*(u.erg/u.s),
           calibration='vp06'):


    if line.lower() == 'ha':

        fwhm = fwhm / (1e3*(u.km/u.s))
        fwhm_err = fwhm_err / (1e3*(u.km/u.s))

        p1 = np.power(fwhm,1.03)
        e1 = np.power(fwhm, 1.03-1.0) * 1.03 * fwhm_err

        p2 = 1.07e3*(u.km/u.s)
        e2 = 0.0*(u.km/u.s)

        fwhm_hb = p1 * p2
        fwhm_hb_err = np.sqrt( (p1*e2)**2 + (p2*e1)**2 )

        fwhm_hb = fwhm_hb / (1e3*(u.km/u.s))
        fwhm_hb_err = fwhm_hb_err / (1e3*(u.km/u.s))

        l5100 = l5100 / (1e44*(u.erg/u.s))
        l5100_err = l5100_err / (1e44*(u.erg/u.s))

        p1 = np.power(10,6.91)
        p2 = np.power(fwhm_hb, 2)
        p3 = np.power(l5100, 0.5)

        e1 = 0.0
        e2 = np.power(fwhm_hb, 2.0-1.0) * 2.0 * fwhm_err
        e3 = np.power(l5100, 0.5-1.0) * 0.5 * l5100_err
 
        MBH = p1 * p2 * p3

        MBH_err = MBH * np.sqrt( (e1/p1)**2 + (e2/p2)**2 + (e3/p3)**2 )

    elif line.lower() == 'hb':

        fwhm = fwhm / (1e3*(u.km/u.s))
        fwhm_err = fwhm_err / (1e3*(u.km/u.s))

        l5100 = l5100 / (1e44*(u.erg/u.s))
        l5100_err = l5100_err / (1e44*(u.erg/u.s))

        p1 = np.power(10,6.91)
        p2 = np.power(fwhm, 2)
        p3 = np.power(l5100, 0.5)

        e1 = 0.0
        e2 = np.power(fwhm, 2.0-1.0) * 2.0 * fwhm_err
        e3 = np.power(l5100, 0.5-1.0) * 0.5 * l5100_err

        MBH = p1 * p2 * p3
        MBH_err = MBH * np.sqrt( (e1/p1)**2 + (e2/p2)**2 + (e3/p3)**2 )        


    elif line.lower() == 'civ':

        fwhm = fwhm / (1e3*(u.km/u.s))
        fwhm_err = fwhm_err / (1e3*(u.km/u.s))

        l1350 = l1350 / (1e44*(u.erg/u.s))
        l1350_err = l1350_err / (1e44*(u.erg/u.s))

        if calibration.lower() == 'vp06':

            # Vestergaard & Peterson 2006
            p1 = np.power(10,6.66)
            p2 = np.power(fwhm, 2)
            p3 = np.power(l1350, 0.53)

            e1 = 0.0
            e2 = np.power(fwhm, 2.0-1.0) * 2.0 * fwhm_err
            e3 = np.power(l1350, 0.53-1.0) * 0.53 * l1350_err

        elif calibration.lower() == 'p13':

            # Park et al. 2013
            p1 = np.power(10,7.48)
            p2 = np.power(fwhm, 0.56)
            p3 = np.power(l1350, 0.52)

            e1 = 0.0
            e2 = np.power(fwhm, 0.56-1.0) * 0.56 * fwhm_err
            e3 = np.power(l1350, 0.52-1.0) * 0.52 * l1350_err

        else:
            print 'Calibration must be vp06 or p13'

        MBH = p1 * p2 * p3
        MBH_err = MBH * np.sqrt((e1/p1)**2 +  (e2/p2)**2 + (e3/p3)**2 )
        # this is for divide rather than multiply???
    else:
        print 'line must equal Hb, Ha, or CIV'


    LogMBH = np.log10(MBH)
    LogMBH_err = (1.0/np.log(10)) * (MBH_err/MBH)


    return {'MBH':MBH,
            'MBH_err':MBH_err,
            'LogMBH':LogMBH,
            'LogMBH_err':LogMBH_err}

if __name__ == '__main__':

    out = BHMass(line='Ha', fwhm=1733.81842*(u.km/u.s), l5100=1.377e46*(u.erg/u.s))
























