# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:49:18 2015

@author: lc585
"""

def ComparePeaksAll():

    """
    Fit all Ha/CIV and then plot in grid
    Rebin affects the plotting but not the fit

    """

    t = Table.read('/home/lc585/Dropbox/IoA/BlackHoleMasses/lineinfo_v3.dat',
                   format='ascii',
                   guess=False,
                   delimiter=',')


    class PlotProperties(object):

        def __init__(self,
                     name,
                     ymin,
                     ymax,
                     civ_rebin,
                     ha_rebin,
                     civ_nGaussians,
                     civ_nLorentzians,
                     ha_nGaussians,
                     ha_nLorentzians,
                     civ_bkgd,
                     ha_bkgd,
                     civ_maskout,
                     ha_maskout,
                     boss_flag,
                     civ_center
                     ):

            self.name = name
            self.ymin = ymin
            self.ymax = ymax
            self.civ_rebin = civ_rebin
            self.ha_rebin = ha_rebin
            self.civ_nGaussians = civ_nGaussians
            self.civ_nLorentzians = civ_nLorentzians
            self.ha_nGaussians = ha_nGaussians
            self.ha_nLorentzians = ha_nLorentzians
            self.civ_bkgd = civ_bkgd
            self.ha_bkgd = ha_bkgd
            self.civ_maskout = civ_maskout
            self.ha_maskout = ha_maskout
            self.boss_flag = boss_flag
            self.civ_center = civ_center

    p = []

    p.append(PlotProperties(name = 'SDSSJ110454.73+095714.8',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-3608,-3313],[-2842,-2503]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ085437.59+031734.8',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-3658,-3334]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ161842.44+234131.7',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = None,
                            ha_maskout  = None,
                            boss_flag  = False,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ073813.19+271038.1',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = None,
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ124602.04+042658.4',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = None,
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ162701.94+313549.2',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-4824,-3730]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ133916.88+151507.6',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians  = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians  = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-1588,-403]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ082906.63+242322.9',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 3,
                            ha_rebin  = 4,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-657,-440],[-190,42]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ080651.54+245526.3',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-1412,-1044],[-852,-573]],
                            ha_maskout  = None,
                            boss_flag  = False,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ163456.15+301437.8',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = None,
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ132948.73+324124.4',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-4099,-3828],[-3622,-3274],[-1024,-729],[-464,-181]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ131749.78+080616.2',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-3038,-2151]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ152529.17+292813.2',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 4,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-5032,-4616],[-1553,-1050],[-963,-569],[874,1180],[2186,2405],[6474,6780]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ074352.61+245743.6',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = None,
                            ha_maskout  = None,
                            boss_flag  = False,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ153848.64+023341.1',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 1,
                            civ_nLorentzians = 0,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-272,99],[340,668]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ153027.37+062330.8',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  =  [[-12000,-8000],[8000,12000]],
                            civ_maskout  =  [[-406,521]],
                            ha_maskout  =  None,
                            boss_flag  =  True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ130618.60+151017.9',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 1,
                            civ_nLorentzians = 0,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = None,
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name= 'SDSSJ123611.21+112921.6',
                            ymin = -4,
                            ymax = 10,
                            civ_rebin = 2,
                            ha_rebin = 2,
                            civ_nGaussians= 1,
                            civ_nLorentzians= 0,
                            ha_nGaussians = 0,
                            ha_nLorentzians= 1,
                            civ_bkgd = [[-15000,-10000],[12000,20000]],
                            ha_bkgd = [[-12000,-8000],[8000,12000]],
                            civ_maskout = None,
                            ha_maskout = None,
                            boss_flag = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ140047.45+120504.6',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 0,
                            civ_nLorentzians = 1,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-9444,-8564],[-1914,-1095]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))

    p.append(PlotProperties(name = 'SDSSJ133646.87+144334.2',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 6,
                            ha_rebin  = 2,
                            civ_nGaussians = 1,
                            civ_nLorentzians = 0,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-20000,-10000],[12000,18000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-2419,-1209]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = -2500.0 ))

    p.append(PlotProperties(name = 'SDSSJ085856.00+015219.4',
                            ymin  = -4,
                            ymax  = 10,
                            civ_rebin  = 2,
                            ha_rebin  = 2,
                            civ_nGaussians = 1,
                            civ_nLorentzians = 0,
                            ha_nGaussians  = 0,
                            ha_nLorentzians = 1,
                            civ_bkgd  = [[-15000,-10000],[12000,20000]],
                            ha_bkgd  = [[-12000,-8000],[8000,12000]],
                            civ_maskout  = [[-8746,-8466],[-8003,-7730],[-3263,-2507],[648,927]],
                            ha_maskout  = None,
                            boss_flag  = True,
                            civ_center = None))


    fig, axs = plt.subplots(6,4,figsize=(8.267,11.692),sharey=True)

#    k = 18
#    ind = np.where( t['SDSS_Name'] == p[k].name )[0][0]
#    row = t[ind]
#
#    out = ComparePeaks(row['SDSS_Name'],
#                       os.path.join('/data/lc585/WHT_20150331/html/',row['Name'],'dimcombLR+bkgd_v138.ms.fits'),
#                       row['Redshift_ICA'],
#                       row['HAB'],
#                       row['KAB'],
#                       ha_rebin = p[k].ha_rebin,
#                       civ_rebin = p[k].civ_rebin,
#                       civ_maskout = p[k].civ_maskout,
#                       ha_maskout = p[k].ha_maskout,
#                       civ_bkgd = p[k].civ_bkgd,
#                       ha_bkgd = p[k].ha_bkgd,
#                       CIV_nGaussians = p[k].civ_nGaussians,
#                       CIV_nLorentzians = p[k].civ_nLorentzians,
#                       Ha_nGaussians = p[k].ha_nGaussians,
#                       Ha_nLorentzians = p[k].ha_nLorentzians,
#                       boss_flag = p[k].boss_flag
#                       )
#
#
#    fig, ax = plt.subplots()
#
#    palette = sns.color_palette()
#
#    xs, step = np.linspace( out[2].min(),
#                           out[2].max(),
#                           1000,
#                                    retstep=True)
#
#    int_civ = np.sum( step * out[1].eval( params=out[0], x=xs ) ) / 5e3
#
#    ax.errorbar(out[2] - out[6]['l0_center'].value ,
#                              out[4] / int_civ,
#                              yerr= out[5] / int_civ,
#                              linestyle='',
#                              color=palette[0],
#                              lw=1,
#                              alpha=1)
#
#
#    line, = ax.plot( xs - out[6]['l0_center'].value,
#                                   resid(out[0], xs , out[1]) / int_civ,
#                                   color = palette[0],
#                                   label = 'CIV')



#    xs, step = np.linspace( out[8].min(),
#                                    out[8].max(),
#                                    1000,
#                                    retstep=True)
#
#
#    int_ha = np.sum( step * out[7].eval( params=out[6], x=xs ) ) / 5e3
#
#    ax.errorbar(out[8] - out[6]['l0_center'].value ,
#                              out[10] / int_ha,
#                              yerr = out[11] / int_ha,
#                              linestyle='',
#                              color=palette[1],
#                              lw=1,
#                              alpha=1)
#
#
#
#    line, = ax.plot( xs - out[6]['l0_center'].value,
#                                   resid(out[6], xs, out[7]) / int_ha,
#                                   color = palette[1],
#                                   label = r'H$\alpha$')
#


    k = 0
    outinfo = []
    for i in range(6):
        for j in range(4):

            ind = np.where( t['SDSSName'] == p[k].name )[0][0]
            row = t[ind]
            print row['SDSSName']

            out = ComparePeaks(row['SDSSName'],
                               os.path.join('/data/lc585/WHT_20150331/html/',row['Name'],'dimcombLR+bkgd_v138.ms.fits'),
                               row['z_ICA'],
                               row['HAB'],
                               row['KAB'],
                               civ_rebin = p[k].civ_rebin,
                               ha_rebin = p[k].ha_rebin,
                               civ_maskout = p[k].civ_maskout,
                               ha_maskout = p[k].ha_maskout,
                               civ_bkgd = p[k].civ_bkgd,
                               ha_bkgd = p[k].ha_bkgd,
                               CIV_nGaussians = p[k].civ_nGaussians,
                               CIV_nLorentzians = p[k].civ_nLorentzians,
                               Ha_nGaussians = p[k].ha_nGaussians,
                               Ha_nLorentzians = p[k].ha_nLorentzians,
                               boss_flag = p[k].boss_flag
                               )

            palette = sns.color_palette()

            xs, step = np.linspace( out[2].min(),
                                    out[2].max(),
                                    1000,
                                    retstep=True)

            int_civ = np.sum( step * out[1].eval( params=out[0], x=xs ) ) / 5e3

            axs[i,j].errorbar(out[2] - out[6]['l0_center'].value ,
                              out[4] / int_civ,
                              yerr= out[5] / int_civ,
                              linestyle='',
                              color=palette[0],
                              lw=1,
                              alpha=0.4)


            line1, = axs[i,j].plot( xs - out[6]['l0_center'].value,
                                   resid(out[0], xs , out[1]) / int_civ,
                                   color = palette[0],
                                   label = 'CIV')

            xs, step = np.linspace( out[8].min(),
                                    out[8].max(),
                                    1000,
                                    retstep=True)


            int_ha = np.sum( step * out[7].eval( params=out[6], x=xs ) ) / 5e3

            axs[i,j].errorbar(out[8] - out[6]['l0_center'].value ,
                              out[10] / int_ha,
                              yerr = out[11] / int_ha,
                              linestyle='',
                              color=palette[1],
                              lw=1,
                              alpha=0.4)



            line2, = axs[i,j].plot( xs - out[6]['l0_center'].value,
                                   resid(out[6], xs, out[7]) / int_ha,
                                   color = palette[1],
                                   label = r'H$\alpha$')

            axs[i,j].set_xlim(-10000,10000)
            axs[i,j].set_ylim(-0.2,2)

            if i < 5:
                axs[i,j].xaxis.set_ticklabels([])
            else:
                setp( axs[i,j].get_xticklabels(), rotation=45)
#            if j > 0:
#                axs[i,j].yaxis.set_ticklabels([])

            for tick in axs[i,j].xaxis.get_major_ticks():
                tick.label.set_fontsize(9)

            for tick in axs[i,j].yaxis.get_major_ticks():
                tick.label.set_fontsize(9)

            axs[i,j].set_title(row['SDSSName'], fontsize=9)

            # Calculate BH masses from SED.
            logMBH, eddratio, l5100 = BlackHoleMass(row['SDSSName'], row['z_ICA'], out[6], row['HAB'], row['EBV_SED'])
            outinfo.append( [row['SDSSName'], logMBH, eddratio.value, l5100] )

            k += 1

            if k + 1 > len(p):
                break
        else:
            continue # executed if the loop ended normally (no break)
        break # executed if 'continue' was skipped (break)

    print outinfo

    legend  = plt.legend(handles=[line1,line2], loc=1)
    ax = plt.gca().add_artist(legend)

    fig.delaxes(axs[-1,-2] )
    fig.delaxes(axs[-1,-3] )
    axs[-1,-1].axis('off')

    fig.text(0.5, 0.04, r'$\Delta v$ [km/s]', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Flux [Arbitary Units]', va='center', rotation='vertical', fontsize=12)


    fig.savefig('/home/lc585/Dropbox/IoA/BlackHoleMasses/paper/gridspectra.pdf')
    plt.show()

    plt.close()





    return None
