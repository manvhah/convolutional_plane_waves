# generating figures for paper
from Soundfield import Soundfield, default_soundfield_config
from SoundfieldReconstruction import find_decomposition_clear,\
        Decomposition,\
        default_reconstruction_config,\
        SoundfieldReconstruction
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from copy import deepcopy
import warnings

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['font.size'] = 16

def gen_models(
        declist = [
            Decomposition.gpwe,
            Decomposition.lpwe,
            Decomposition.cpwe,
            ]):
    sfr_list = []

    for dec in declist:
        recopts, transopts = default_reconstruction_config(dec.name)

        # load or generate training data, might not be needed but 
        trainopts = default_soundfield_config(measurement="classroom", frequency=1000)
        learn     = Soundfield(**trainopts)
        sfrec_obj = SoundfieldReconstruction(
            training_field = learn, 
            transform_opts = transopts, 
            **recopts)
        sfrec_obj.fit()

        sfr_list.append(deepcopy(sfrec_obj))
    return sfr_list


def reconstruction_test(test_densities = [], sfrlist = None,
        valargs = dict(measurement = "classroom", frequency = 1000),mode='default',
        titles=None, b_plot_only=False, plot_intensity_idx=[],
        ):
    from src._plot_tools import add_cbar, annotatep

    nl = len(sfrlist)
    nv, nh = len(test_densities)+len(plot_intensity_idx), 2+nl

    fig = plt.figure(11, figsize=(nh*2.8+1.2,nv*3.3))
    plt.clf()
    gs = GridSpec(nv, nh, figure=fig, 
            left=0.05, right=0.93, hspace = .3, wspace = .6)
    axes = [fig.add_subplot(gsi) for gsi in gs]

    valopts = default_soundfield_config(**valargs)
    val = Soundfield(**valopts)

    improp = dict(
        cmap   = 'viridis', 
        origin = 'lower', 
        extent =  val.extent ,
        )
    fontsize = 13

    efunc = lambda x : np.abs(x)
    if ('019' in val.measurement) or ('classroom' in val.measurement):
        aopt = dict(xticks = [1.1,1.5,2.0,2.5], yticks = [3.0,3.5,4.0,4.5])
        xlabel = '$x$ [m]'
        ylabel = '$y$ [m]'
        clabel = '[dB rel $<{\mathbf{p}_{true}^2}>$]'
        norm = 10 * np.log10(np.mean(efunc(val.p)**2))
        improp.update(dict(vmin=-20, vmax=10))
    else:
        clabel = '[dB SPL]'
        aopt = dict(xticks=[-2,0,3],yticks=[-1,0,2,4])
        xlabel = '$x$ [$\lambda$]'
        ylabel = '$y$ [$\lambda$]'
        norm = 20 * np.log10(2e-5)
        improp.update(dict(
            # vmin = np.min(20 * np.log10(efunc(val.p)))-norm,
            # vmax = np.max(20 * np.log10(efunc(val.p)))-norm,
            vmin = 50,
            vmax = 80,
            ))

    props = list()
    imdata = list()
    generate_titles = 0
    if not titles: 
        generate_titles = 1
        titles = list()
    vallabel = list()
    vectorfields = list()

    for dd, density in enumerate(test_densities):
        val.__dict__.update(dict(spatial_sampling = -density))
        if ('019' in val.measurement) or ('classroom' in val.measurement): # demo case reproducability sampling
            val._seed = 387309083
            _ = val.sidx # generate common measurements

        if not b_plot_only:
            [sfr.clear_reconstruction() for sfr in sfrlist]
            [sfr.reconstruct(measured_field = val) for sfr in sfrlist]

        if ('019' in val.measurement) or ('classroom' in val.measurement):
            vallabel.append(f"{val.frequency:.0f} Hz, "+r"$N_{\mathrm{obs}}$"+f"={val.N:.0f}\n")
        else:
            vallabel.append("$N_{\mathrm{obs}}$"+f"={val.N:.0f}\n")
        
        print("sound field f, num mics, density, surface ref norm", val.f, val.N, val.mic_density, norm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            imdata.append( 20*np.log10(efunc(val.sp))  -norm)
            [imdata.append(20*np.log10(efunc(sfr.rf.p))-norm) for sfr in sfrlist]
            imdata.append( 20*np.log10(efunc(val.p))   -norm)
        if generate_titles:
            titles.append("$\mathbf{p}_{\mathrm{obs}}$ measurement")
            [titles.append("$\hat{\mathbf{p}}$ " + find_decomposition_clear(sfr.decomposition)) for sfr in sfrlist]
            titles.append('$\mathbf{p}$ true')
        if dd in plot_intensity_idx:
                imdata.append(20 * np.log10(efunc(val.sp))  - norm)
                [imdata.append(20 * np.log10(efunc(sfr.rf.IJ))) for sfr in sfrlist]
                imdata.append(20 * np.log10(efunc(val.IJ)))
                #indicate whcih plot function to use
                [vectorfields.append(ii) for ii in range(len(imdata)-nh+1,len(imdata))]
                if generate_titles:
                    titles.append("$\mathbf{p}_{\mathrm{obs}}$")
                    [titles.append("$\hat{\mathbf{I}}_{ux}$ " +
                        find_decomposition_clear(sfr.decomposition)) for sfr in sfrlist]
                    titles.append('$\mathbf{I}$ true')

        prop = np.array([(sfr.spatial_coherence, sfr.nmse, sfr.avg_nonzeros, sfr.rec_time) for sfr in sfrlist])
        props.append(prop)
        if hasattr(val,'sample_idx'): delattr(val,'sample_idx') # force re-sample sound field

    vallabel.reverse()
    for ii, data in enumerate(imdata):
        ax = axes[ii]
        if ii not in vectorfields:
            im = ax.imshow(data.squeeze(), **improp)
            ax.grid(False)
            cbar = add_cbar(im,ax)
            ax.set(**aopt)
            ax.set_title(titles[int(ii)], loc= "left", x=-.1, y= 1.3)
            subplothidx = ii%nh
            subplotvidx = int(np.floor(ii/nh))
            if (subplothidx == 0) :
                ax.set_ylabel(ylabel)
                annotatep(ax,vallabel.pop(), x = -.00, y = 1.0, align='left',color='k')
                annotatep(ax," \n({:.2f}".format(np.sqrt(test_densities[subplotvidx])) + 
                    " mics per $\lambda$)", x = -.00, y = 1.0, align='left')
            elif (subplothidx<nh-1):
                annotatep(ax, 
                    "$C\,=\,$"+"{:.2f}".format(props[subplotvidx][subplothidx-1,0]) +
                    "\n" + 
                    "NMSE$\,=\,$"+"{:.2f} dB".format(props[subplotvidx][subplothidx-1,1]) ,
                    x = -.00, y = 1.01, align='left')
            if (subplothidx==nh-1):
                cbar.set_label(clabel)
            if (subplotvidx==nv-1):
                ax.set_xlabel(xlabel)
        else:
            Q = ax.quiver(data[0],data[1])
            qk = ax.quiverkey(Q, 0.6, 1.02, 5e-6, r'5$\,\mu$Wm$^{-2}$', labelpos='E')
            ax.set(xticks=[],yticks=[],aspect='equal')


    decs = [sfr.decomposition for sfr in sfrlist]
    tag = val.measurement+"_{:.0f}_".format(val.frequency)+"_".join(decs)

    print("./figures/rec_"+tag+".pdf")
    fig.savefig("./figures/rec_"+tag+".pdf")
    plt.close()

    return sfrlist


def plot_test():
    decompositions = [
        Decomposition.gpwe,
        Decomposition.gpwe,
        Decomposition.gpwe,
        Decomposition.gpwe,
        ]
    sfr_list = gen_models(declist = decompositions)
    reconstruction_test([4.0, 12.0], sfr_list, {'frequency':1000,'measurement':'classroom'} )
    plt.close()


def csc_paper():
    """ CSC PAPER DEMO CASES
    """
    decompositions = [
        Decomposition.gpwe,
        Decomposition.lpwe,
        Decomposition.cpwe,
        Decomposition.cpwe,
        ]
    samplings=[4.0,12.0]

    titles = []
    def tappend(tag):
        titles.append( '({}) '.format(chr(len(titles)+97)) + tag)
    tappend("$\mathbf{p}_{\mathrm{obs}}$")
    tappend("$\hat{\mathbf{p}}$ global")
    tappend("$\hat{\mathbf{p}}$ local independent")
    tappend("$\hat{\mathbf{p}}$ conv. sparse")
    tappend("$\hat{\mathbf{p}}$ conv. smooth")
    tappend("$\mathbf{p}$ true")
    tappend("$\mathbf{p}_{\mathrm{obs}}$")
    tappend("$\hat{\mathbf{p}}$ global")
    tappend("$\hat{\mathbf{p}}$ local independent")
    tappend("$\hat{\mathbf{p}}$ conv. sparse")
    tappend("$\hat{\mathbf{p}}$ conv. smooth")
    tappend("$\mathbf{p}$ true")
    tappend("$\mathbf{p}_{\mathrm{obs}}$")
    tappend("$\hat{\mathbf{I}}_{ux}$ global")
    tappend("$\hat{\mathbf{I}}_{ux}$ local independent")
    tappend("$\hat{\mathbf{I}}_{ux}$ conv. sparse")
    tappend("$\hat{\mathbf{I}}_{ux}$ conv. smooth")
    tappend("$\mathbf{I}_{ux}$ true")

    sfrl = gen_models(declist = decompositions)
    sfrl[-2].transform = 'csc' # standard csc
    reconstruction_test(samplings, sfrl, {'measurement':'monoplane'}, titles=titles, 
            # plot_intensity_idx=[1],
            )
    reconstruction_test(samplings, sfrl, {'measurement':'classroom','frequency':1000}, titles=titles)
    plt.close()

if __name__ == "__main__":
    # plot_test()
    csc_paper()
