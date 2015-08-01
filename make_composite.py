# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:49:50 2015

@author: lc585
"""

def MakeComposite(spectra_list):

    """
    Give list of names to make composites from
    """

    fname = '/home/lc585/Dropbox/IoA/BlackHoleMasses/lineinfo_v3.dat'
    t = Table.read(fname,
                   format='ascii',
                   guess=False,
                   delimiter=',')

    spec, err2 = [], []

    for name in spectra_list:

        hdulist = fits.open( os.path.join('/data/lc585/WHT_20150331/html/',name,'dimcombLR+bkgd_v138.ms.fits') )
        hdr = hdulist[0].header
        data = hdulist[0].data
        hdulist.close()

        x_data, dw = getwave(hdr)
        y_data = data[0,:,:].flatten()
        y_sigma = data[-1,:,:].flatten()

        i = np.where( t['Name'] == name )[0][0]

        # Load filter responses into bp and calculate dlam.
        with open('/home/lc585/Dropbox/IoA/WHT_Proposal_2015a/Iraf/Filter_Response/H.response','r') as f:
            wavtmp, rsptmp = np.loadtxt(f,unpack=True)
        dlam = (wavtmp[1] - wavtmp[0])
        bp = np.ndarray(shape=(2,len(wavtmp)), dtype=float)
        bp[0,:], bp[1,:] = wavtmp, rsptmp

        sum1 = np.sum( bp[1] * (0.10893/(bp[0]**2)) * bp[0] * dlam)
        sum2 = np.sum( bp[1] * bp[0] * dlam)
        flxlam = sum1 / sum2
        zromag = -2.5 * np.log10(flxlam)

        # Now calculate magnitudes
        spc = interp1d(x_data, y_data, bounds_error=False, fill_value=0.0)

        sum1 = np.sum( bp[1] * spc(bp[0]) * bp[0] * dlam)
        sum2 = np.sum( bp[1] * bp[0] * dlam)
        flxlam = sum1 / sum2
        ftrmag = (-2.5 * np.log10(flxlam)) - zromag

        deltaH = t['HAB'][i] - ftrmag

        y_data = y_data * 10.0**( -1.0 * deltaH / 2.5 )
        y_sigma = y_sigma * 10.0**( -1.0 * deltaH / 2.5 )

        # Transform to quasar rest-frame
        x_data = x_data / (1.0 + t[i]['z_HW10'])
        x_data = x_data * (1.0 - t[i]['Ha_center']*(u.km/u.s) / c.to(u.km/u.s))

        f1 = interp1d( x_data, y_data, bounds_error=False, fill_value=np.nan )
        f2 = interp1d( x_data, y_sigma**2, bounds_error=False, fill_value=np.nan )

        spec.append( f1( np.linspace(4000.0,7600.0,1168) ) )
        err2.append( f2( np.linspace(4000.0,7600.0,1168) ) )

    spec = np.array(spec)
    err2 = np.array(err2)

    meanspec, ns = np.zeros(1168), np.zeros(1168)

    for i in range(1168):
        for j in range(len(spectra_list)):
            if ~( np.isnan(spec[j,i]) | np.isnan(err2[j,i]) ):
                ns[i] += 1.0
                meanspec[i] += spec[j,i]

    return np.linspace(4000.0,7600.0,1168), 1e17 * meanspec / ns

#fig, ax = plt.subplots()
#
#spectra_list = ['SDSSJ0738+2710',
#                'SDSSJ0858+0152',
#                'SDSSJ1236+1129',
#                'SDSSJ1306+1510',
#                'SDSSJ1336+1443',
#                'SDSSJ1400+1205',
#                'SDSSJ0806+2455',
#                'SDSSJ1530+0623',
#                'SDSSJ1618+2341']
#
#wav, flux = MakeComposite(spectra_list)
#ax.plot(wav, flux, lw=1, label=r'Small H$\alpha$ FWHM')
#
#spectra_list = ['SDSSJ1104+0957',
#                'SDSSJ0829+2423',
#                'SDSSJ0743+2457',
#                'SDSSJ0854+0317',
#                'SDSSJ1246+0426',
#                'SDSSJ1317+0806',
#                'SDSSJ1329+3241',
#                'SDSSJ1339+1515',
#                'SDSSJ1634+3014']
#
#wav, flux = MakeComposite(spectra_list)
#ax.plot(wav, flux, lw=1, label=r'Large H$\alpha$ FWHM')
#
#
#ax.set_xlabel(r'Wavelength [$\AA$]')
#ax.axvline(6564.614, color='red')
#plt.legend()
#plt.savefig('/home/lc585/Dropbox/IoA/BlackHoleMasses/large_small_fwhm_ha_composites.png')
#
#fig, axs  = plt.subplots(9,1, figsize=(6,40))
#
#
#fname = '/home/lc585/Dropbox/IoA/BlackHoleMasses/lineinfo_v3.dat'
#t = Table.read(fname,
#               format='ascii',
#               guess=False,
#               delimiter=',')
#
#for j in range(9):
#
#    hdulist = fits.open( os.path.join('/data/lc585/WHT_20150331/html/',spectra_list[j],'dimcombLR+bkgd_v138.ms.fits') )
#    hdr = hdulist[0].header
#    data = hdulist[0].data
#    hdulist.close()
#
#    x_data, dw = getwave(hdr)
#    y_data = data[0,:,:].flatten()
#    y_sigma = data[-1,:,:].flatten()
#
#    i = np.where( t['Name'] == spectra_list[j] )[0][0]
#
#    from astropy.constants import c
#    x_data = x_data / (1.0 + t[i]['z_HW10'])
#    x_data = x_data * (1.0 - t[i]['Ha_center']*(u.km/u.s) / c.to(u.km/u.s))
#
#
#    axs[j].plot(x_data , y_data, lw=1)
#    axs[j].axvline(6564.614, color='red')
#    axs[j].set_xlim(6000,7000)
#
#plt.show()
