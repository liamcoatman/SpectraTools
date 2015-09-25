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


def BHMass(line='Ha', fwhm=1000*(u.km/u.s), l5100=1.0e46*(u.erg/u.s), l1350=1.0e46*(u.erg/u.s)):

    if line.lower() == 'hb':
        MBH = np.power(10,6.91) * np.power(fwhm/(1e3*(u.km/u.s)),2) * np.power(l5100 / (1e44*(u.erg/u.s)) , 0.5)
        Lbol = 9.26 * l5100 # Richards et al. 2006A

    elif line.lower() == 'ha':
        fwhm = 1.07 * 1e3 * np.power(fwhm/(1e3*(u.km/u.s)),1.03) *(u.km/u.s)
        MBH = np.power(10,6.91) * np.power(fwhm/(1e3*(u.km/u.s)),2) * np.power(l5100 / (1e44*(u.erg/u.s)) , 0.5)
        Lbol = 9.26 * l5100 # Richards et al. 2006A

    elif line.lower() == 'civ':
        MBH = np.power(10,6.66) * np.power(fwhm/(1e3*(u.km/u.s)),2) * np.power(l1350 / (1e44*(u.erg/u.s)) , 0.53)
        Lbol = 3.81 * l1350  # Richards et al. 2006A

    else:
        print 'line must equal Hb, Ha, or CIV'

    Ledd = 3.2e4 * MBH # Eddington luminosity in units of solar luminosity
    EddRatio = Lbol / (3.846e33*(u.erg/u.s)) / Ledd

    return np.log10(MBH), Lbol, EddRatio

if __name__ == '__main__':

    MBH, Lbol, EddRatio = BHMass(line='Ha', fwhm=1733.81842*(u.km/u.s), l5100=1.377e46*(u.erg/u.s))
























