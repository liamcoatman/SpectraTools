# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:48:37 2015

@author: lc585
"""

def ComparePeaks(sdssname,
                 name,
                 z,
                 hmag,
                 kmag,
                 outname=None,
                 civ_rebin=None,
                 ha_rebin=None,
                 civ_maskout = None,
                 ha_maskout = None,
                 civ_bkgd = None,
                 ha_bkgd = None,
                 CIV_nGaussians = 0,
                 CIV_nLorentzians = 0,
                 Ha_nGaussians = 0,
                 Ha_nLorentzians = 0,
                 boss_flag = False,
                 civ_center = None):

    """
    Compare the profiles of two peaks
    """

    pars_CIV, mod_CIV, x_data_CIV, dw_CIV, y_data_cr_CIV, y_sigma_CIV = dofit(fname = None,
                                                                              sdssname = sdssname,
                                                                              z = z,
                                                                              hmag = None,
                                                                              kmag = None,
                                                                              rebin = None,
                                                                              line_wlen = np.mean([1548.202,1550.774]),
                                                                              line_wlen_min = 15000.0,
                                                                              line_wlen_max = 20000.0,
                                                                              sigmaclipbackground = False,
                                                                              nGaussians = CIV_nGaussians,
                                                                              nSkewedGaussians = 0,
                                                                              nLorentzians = CIV_nLorentzians,
                                                                              fluxcalibrate = False,
                                                                              SDSS = True,
                                                                              boss_flag = boss_flag,
                                                                              verbose = True,
                                                                              plotname = None,
                                                                              civ_center = civ_center,
                                                                              fittingmethod = 'nelder',
                                                                              maskout = civ_maskout,
                                                                              bkgd_window = civ_bkgd,
                                                                              plot = False)



    fname = name
    pars_Ha, mod_Ha, x_data_Ha, dw_Ha, y_data_cr_Ha, y_sigma_Ha = dofit(fname = fname,
                                                                        z = z,
                                                                        hmag = hmag,
                                                                        kmag = kmag,
                                                                        rebin = None,
                                                                        line_wlen = 6564.614,
                                                                        line_wlen_min = 12000.0,
                                                                        line_wlen_max = 12000.0,
                                                                        sigmaclipbackground = False,
                                                                        nGaussians = Ha_nGaussians,
                                                                        nSkewedGaussians = 0,
                                                                        nLorentzians = Ha_nLorentzians,
                                                                        plot = False,
                                                                        fluxcalibrate = True,
                                                                        SDSS = False,
                                                                        sdssname= None,
                                                                        verbose = True,
                                                                        fittingmethod = 'nelder',
                                                                        plotname = None,
                                                                        maskout = ha_maskout,
                                                                        bkgd_window = ha_bkgd )


    x_data_CIV, y_data_cr_CIV, y_sigma_CIV = rebin_simple(x_data_CIV,
                                                          y_data_cr_CIV,
                                                          y_sigma_CIV,
                                                          civ_rebin,
                                                          weighted=False)

    x_data_Ha, y_data_cr_Ha, y_sigma_Ha = rebin_simple(x_data_Ha,
                                                       y_data_cr_Ha,
                                                       y_sigma_Ha,
                                                       ha_rebin,
                                                       weighted=False)



#    fig = plt.figure(figsize=(8,6))
#
#    ax = fig.add_subplot(1,1,1)
#
#    palette = sns.color_palette()
#
##    newpars_CIV = copy.deepcopy(pars_CIV)
##
##    regex = re.compile('(l|g)._amplitude')
##    for i in pars_CIV.valuesdict():
##        if re.match( regex, i):
##            newpars_CIV[i].value = 0.0
##
##    continuum_CIV = resid(newpars_CIV, x_data_CIV, mod_CIV)
##    f_continuum_CIV = interp1d( x_data_CIV, continuum_CIV)
##
##    newpars_Ha = copy.deepcopy(pars_Ha)
##
##    regex = re.compile('(l|g)._amplitude')
##    for i in pars_Ha.valuesdict():
##        if re.match( regex, i):
##            newpars_Ha[i].value = 0.0
##
##    continuum_Ha = resid(newpars_Ha, x_data_Ha, mod_Ha)
##    f_continuum_Ha = interp1d( x_data_Ha, continuum_Ha)
##
##    ew_CIV = np.abs( np.sum( (1 - y_data_cr_CIV / continuum_CIV) * dw_CIV) )
##    ew_Ha = np.abs( np.sum( (1 - y_data_cr_Ha / continuum_Ha) * dw_Ha ) )
##
##    ax.errorbar(x_data_CIV + np.abs( pars_Ha['l0_center'].value ),
##                (y_data_cr_CIV / continuum_CIV - 1.0) / ew_CIV ,
##                yerr= (y_sigma_CIV / continuum_CIV) / ew_CIV,
##                linestyle='', color=palette[0])
##
##    xs, step = np.linspace( x_data_CIV.min(), x_data_CIV.max(), 1000, retstep=True)
##    line, = ax.plot( xs + np.abs( pars_Ha['l0_center'].value ),
##                     (resid(pars_CIV, xs , mod_CIV) / f_continuum_CIV( xs ) - 1.0) / ew_CIV ,
##                     color=palette[0],
##                     label='CIV')
##
##
#    ax.errorbar(x_data_CIV + np.abs( pars_Ha['l0_center'].value ),
#                y_data_cr_CIV,
#                yerr= y_sigma_CIV,
#                linestyle='',
#                color=palette[0])
#
#    xs, step = np.linspace( x_data_CIV.min(),
#                            x_data_CIV.max(),
#                            1000,
#                            retstep=True)
#
#    line, = ax.plot( xs + np.abs( pars_Ha['l0_center'].value ),
#                     resid(pars_CIV, xs , mod_CIV),
#                     color = palette[0],
#                     label = 'CIV')
#
#    ax.errorbar(x_data_Ha + np.abs( pars_Ha['l0_center'].value ),
#                y_data_cr_Ha,
#                yerr = y_sigma_Ha,
#                linestyle='',
#                color=palette[1])
#
#    xs, step = np.linspace( x_data_Ha.min(),
#                           x_data_Ha.max(),
#                           1000,
#                           retstep=True)
#
#    line, = ax.plot( xs + np.abs( pars_Ha['l0_center'].value ),
#                     resid(pars_Ha, xs, mod_Ha),
#                     color = palette[1],
#                     label = r'H$\alpha$')
#
#    plt.xlabel(r"$\Delta$ v (kms$^{-1}$)", fontsize=14)
#    plt.ylabel(r'$F_{\lambda}$ (Arbitrary Units)', fontsize=14)
#
#    ax.set_title( sdssname , fontsize=14)
#
##    ax.errorbar(x_data_Ha + np.abs( pars_Ha['l0_center'].value ),
##                (y_data_cr_Ha / continuum_Ha - 1.0) / ew_Ha ,
##                yerr=(y_sigma_Ha / continuum_Ha) / ew_Ha,
##                linestyle='', color=palette[1])
##    xs, step = np.linspace( x_data_Ha.min(), x_data_Ha.max(), 1000, retstep=True)
##    line, = ax.plot( xs + np.abs( pars_Ha['l0_center'].value ),
##                     (resid(pars_Ha, xs, mod_Ha) / f_continuum_Ha( xs ) - 1.0) / ew_Ha ,
##                     color=palette[1],
##                     label=r'H$\alpha$')
##
#    plt.legend(prop={'size':14} )
#    plt.tick_params(axis='both', which='major', labelsize=12)
##
##
###    figtxt = 'CIV \n'
###    for i in pars_CIV.valuesdict():
###        figtxt += i + ' = {0} +/- {1} \n'.format( float('{0:.4g}'.format( pars_CIV[i].value)), float('{0:.4g}'.format( pars_CIV[i].stderr)) )
###
###    figtxt += 'Ha \n'
###    for i in pars_Ha.valuesdict():
###        figtxt += i + ' = {0} +/- {1} \n'.format( float('{0:.4g}'.format( pars_Ha[i].value)), float('{0:.4g}'.format( pars_Ha[i].stderr)) )
##
##
###    plt.figtext(0.1,0.45,figtxt,fontsize=14,va='top')
##
#    ax.set_xlim(-10000,10000)
#    ax.set_ylim(-4,10)
#
#    fig.tight_layout()
#
#    if outname is not None:
#        plt.savefig(outname)
#
#    plt.show()
#
#    plt.close()
    return (pars_CIV, mod_CIV, x_data_CIV, dw_CIV, y_data_cr_CIV, y_sigma_CIV, pars_Ha, mod_Ha, x_data_Ha, dw_Ha, y_data_cr_Ha, y_sigma_Ha)