import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import warnings
from SoundfieldReconstruction import (
    SoundfieldReconstruction,
    read_sfr_from_hdf,
    find_decomposition_clear,
    Decomposition,
    ALLPARAMETERS,
)
from src._const import *

titleopts = {"loc": "left","position":(.0,1.00)}

def settitle(handle, number, title, opts=titleopts):
    handle.text(-.13,1.05, '({})'.format(chr(number+97)),
            transform=handle.transAxes,fontsize=22)
    handle.set_title(title, **opts)


def analysis(
    rr, bindex=None, **kwargs
):

    if np.all([hasattr(rr, key) for key in ["spatial_coherence", "nmse", "id"]]):
        if rr.id == rr.title():
            return
        else:
            [delattr(rr, key) for key in ["spatial_coherence", "nmse", "id"]]
            return analysis(rr, bindex, **kwargs)
    else:
        mf = rr.measured_field
        rf = rr.reconstructed_field
        fstamp = rr.fit_timestamp

        ### De-compress
        # future also re-read measurements, only store random variables
        if not hasattr(mf, "_pm"): mf._gen_field()  # re-read

        ### input, reconstruction, reference, error
        # n_points in relation to nyq sampling along one dimension
        mf.n_points = np.prod(mf.sidx.shape)
        spacing_in_nyq = (
            mf.n_points
            / np.ceil((mf.dx * (mf.shp[0] - 1)) / (c0 / mf.frequency / 2)) ** 2
        )

        rr.assess_reconstruction()

        rr.fit_timestamp = fstamp
        rr.__dict__.update({"id": rr.title(),})
        print("\t > coh: {:.2f}, nmse: {:.2f} dB, avg_nz: {:.2f}".format(
            rr.spatial_coherence , rr.nmse, rr.avg_nonzeros))


def plot_xy(figure, dframe,
    xkeys, ykey,
    select,
    xlims=None, xlabels=None, xlog=False, 
    ylims=None, ylabels=None, ylog=False,
    ):

    import matplotlib.pyplot as plt

    gridopts = { 'left': 0.12, 'right': 0.95, 'hspace': 0.60, }
    nsubs = len(select[1])
    if type(figure) is int:
        if nsubs == 2:
            fsize=(7, 7.5) # 2
            gridopts.update({
                'bottom': 0.09, 
                'top':    .80, #space for legend, else 95
                })
        elif nsubs == 3:
            fsize=(7, 10.5) # 3
            gridopts.update({
                'bottom': 0.08, 
                'top':    .86, #space for legend, else 95
                })
        else:
            fsize=(7, 13.0) # >3
        fig = plt.figure(figure, figsize=fsize) # 3
    elif type(figure) is tuple: 
        fig = plt.figure(1, figsize=figure)
    else: 
        fig = figure
    fig.clear()

    symbols = ["x", "o", "+", "*", "s", ">", "o", "v"]
    linestyles = ["-", "-.", ":", "--", "-.", "-.", "-.",":", "-", "-.", ":", "-.","-"]

    nx = len(xkeys)
    ny = len(select[1])
    decs = np.unique(dframe.decomposition, return_inverse=True)

    gs0 = gs.GridSpec( ny, nx, figure = fig, **gridopts)
    axes = list()
    for yy, sk in enumerate(select[1]):
        for xx, xkey in enumerate(xkeys):
            ff = yy * nx + xx
            fidx = dframe[select[0]] == sk
            axes.append(fig.add_subplot(gs0[ff]))
            ax = axes[-1]

            for dd, dec in enumerate(decs[0]):

                idx = np.array((dd == decs[1]) & fidx)
                xd = dframe[idx].sort_values(xkey)[xkey]
                yd = dframe[idx].sort_values(xkey)[(ykey, "mean")]
                ys = dframe[idx].sort_values(xkey)[(ykey, "std")]
                ax.fill_between(
                    xd, yd - ys, yd + ys,
                    color="C{:d}".format(Decomposition[dec].value),
                    alpha=0.4, linewidth=0, zorder=2.1
                    )
                gg = ax.plot(
                    xd,
                    yd,
                    linewidth=2,
                    color="C{:d}".format(Decomposition[dec].value),
                    linestyle=linestyles[Decomposition[dec].value], 
                    label=find_decomposition_clear(dec),
                    alpha=0.9,
                )
            ax.tick_params(which="both", direction="in")

            if xlims:   ax.set_xlim(xlims)
            if ylims:   ax.set_ylim(ylims)
            if xlog:    ax.set_xscale("log")
            if ylog:    ax.set_yscale("log")
            if ylabels: ax.set_ylabel(ylabels     )# + ", $\mu \pm \sigma$",usetex=False)
            else:       ax.set_ylabel(ykey.upper())# + ", $\mu \pm \sigma$",usetex=False)
            if 'gamma' in ylabels:
                if ((0>ylims[0])&(1.<ylims[1])):
                    ax.set_yticks([cc for cc in [.0, .5, .8, 1.]])
            if 'NMSE' in ylabels:
                ax.set_yticks([-20, -15,-10,-5,0])

            xt = dframe[xkey].sort_values().unique()
            if len(xt)>10: # find match for last idx only
                if ('spatial_sampling' in xkey):
                    if xlog:
                        xt = [5,10,20,40,80,160,320,640,1280]
                    else:
                        xt = [5,160,320,640,1000,1280]
                else:
                    xt = dframe[idx][xkey].sort_values().unique()
            xtls = np.round(xt).astype(int)
            if xkey == 'f':
                xt = [500, 600, 700, 800, 900, 1000, 1250, 1600, 2000]
                xtls = ['500',"",'700','','','1000','1250','1600','2000']
            ax.set_xticks(xt) 
            ax.set_xticklabels(xtls,)

            if xlabels: ax.set_xlabel(xlabels[xx],usetex=False)
            else:       ax.set_xlabel(xkey.upper())

    return axes


def plot_mcmeasures(dframe):
    import matplotlib.pyplot as plt
    fig = plt.figure(4, figsize=(9, 9))
    fig.clear()

    gs0 = gs.GridSpec(2, 2, figure=fig, top=0.94, bottom=0.20)
    axes = [
        fig.add_subplot(gs0[0]),
        fig.add_subplot(gs0[1]),
        fig.add_subplot(gs0[2]),
        fig.add_subplot(gs0[3]),
    ]
    symbols = ["x", "d", "*", "s", ">", "o", "v"]
    freqs = [600, 800]
    decs = np.unique(dframe.decomposition, return_inverse=True)
    c = np.unique(dframe.spatial_sampling, return_inverse=True)[1]
    vmin = np.min(c)
    vmax = np.max(c)
    for ff, freq in enumerate(freqs):
        fidx = dframe.f == freq

        fidx = fidx & (dframe.spatial_sampling == 0)
        ax = axes[2 * ff]
        gridlist = []
        for dd, dec in enumerate(decs[0]):
            idx = (dd == decs[1]) & fidx
            ydata = dframe.nmse[idx]
            ax.scatter(
                dframe.spatial_coherence[idx],
                ydata,
                marker=symbols[dd],
                label=dec.upper(),
                alpha=0.4,
                cmap="Accent",
            )
        ax.legend(ncol=1)
        settitle(ax, 2 * ff, "Projection @ {:.0f} Hz".format(freq))
        ax.set_xlim([0.62, 1.02])
        ax.set_ylim([-21, 6])
        ax.set_ylabel("NMSE $<\epsilon_n^2>$ [dB]")
        if ff == len(freqs) - 1:
            plt.xlabel("$\gamma_\mathbf{\hat{p}p}$")


def find_file(pattern, path):
    import os, fnmatch

    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # if fnmatch.fnmatch(name, pattern):
            if pattern in name:
                result.append(os.path.join(root, name))
    return result


if __name__ == "__main__":
    import sys
    import pandas as pd
    import pickle
    import os
    from SoundfieldReconstruction import SPATIAL_SAMPLING, REC_FREQ
    from Soundfield import calc_density, calc_number_of_mics

    ### setup from sys arg ###
    b_update_from_hdfs  = False
    if len(sys.argv) > 1:  # arguments?
        if "-update" in sys.argv[1:]:
            b_update_from_hdfs = True;
            sys.argv.remove('-update')
        if "-legends" in sys.argv[1:]:
            def show_legend(x): return True
            sys.argv.remove('-legends')
        else:
            def show_legend(x): return (x==0)

    filenames = []
    if len(sys.argv[1:]):
        for spath in sys.argv[1:]:  # rsearch in folder, else assume file
            if os.path.isfile(spath):
                filenames.append(spath)
            elif os.path.isdir(spath):
                for root, dirs, files in os.walk(spath):
                    for file in files:
                        if file.endswith(".h5"):
                            filenames.append(os.path.join(root, file))
    else:  # look recursively
        filenames = []
        for root, dirs, files in os.walk("./"):
            for file in files:
                if file.endswith(".h5"):
                    filenames.append(os.path.join(root, file))
        print(filenames)

    analysis_set = {
        # 'decomposition' : ['oldl','gpwe'],
        # 'training'    : ['011'],
        # 'evaluation'  : ['019'],
        # 'spatial_sampling' : [160,320],
        # 'frequency'   : 600,
    }

    if "-pdfname" in sys.argv:
        pdfname = sys.argv[sys.argv.index('-pdfname')+1]
        sys.argv.remove('-pdfname')
        sys.argv.remove(pdfname)
    else:
        pdfname = "".join( 
                [os.path.basename(string).split(".")[0] for string in sys.argv[1:]])
    pickle_names = [os.path.basename(string).split(".")[0] for string in sys.argv[1:]]

    mdflist = []
    mdatalist = []
    for pname in pickle_names:
        try:
            print(pname)
            mdf = pickle.load(open("./data/" + pname + "_mdf.pickle", "rb"))
            mdata = pickle.load(open("./data/" + pname + "_mdata.pickle", "rb"))
            if 'msne' in mdf.keys():
                mdf['nmse'] = mdf.pop('msne')
                mdata['nmse'] = mdata.pop('msne')
            mdflist  .append(mdf)
            mdatalist.append(mdata)
        except:
            Warning("no pickle found, try \" update \" option to read raw data from h5")
            continue

    mdata = {}
    mdf = pd.concat(mdflist)
    for key in mdatalist[0].keys():
        for md in mdatalist:
            if key in mdata:
                try: mdata[key].append(md[key])
                except: pass
            else: mdata[key] = md[key]

    # clean
    mdf.loc[mdf.spatial_sampling == 0, 'spatial_sampling'] = mdf.n_points[mdf.spatial_sampling == 0]

    if 'transform' in mdf.keys():
        mdf.loc[['lc' in mt for mt in mdf['transform']],'reg_lambda'] = 42
        mdf.loc[['lc' in mt for mt in mdf['transform']],'reg_tol_c']  = 42
        mdf.loc[['xv' in mt for mt in mdf['transform']],'reg_lambda'] = 66
        mdf.loc[['xv' in mt for mt in mdf['transform']],'reg_tol_c']  = 66

    # ## tolerance or lambda
    # mdf.loc[mdf.reg_lambda == 0, 'reg_lambda'] = mdf.loc[mdf.reg_lambda == 0, 'reg_tol_c'] * 11.0; del mdf["reg_tol_c"]
    mdf.loc[mdf.reg_tol_c == 0, 'reg_tol_c'] = mdf.loc[mdf.reg_tol_c == 0, 'reg_lambda'] * 11.0; del mdf["reg_lambda"]

    ## filter
    exceptions = [ ]
    for ex in exceptions:
        mdf = mdf[mdf.decomposition != ex]
    mdf = mdf[mdf.f != 1020]
    mdf = mdf[mdf.f != 980]
    mdf.loc[mdf.decomposition == 'gpr','nof_components']  = 0
    mdf.loc[mdf.decomposition == 'agpr','nof_components']  = 0
    mdf = mdf[(mdf.nof_components > 10) | (mdf.nof_components == 0)] #DL shrinking errors

    mdf = mdf.reset_index(drop=True)

    columns = [
            "f", 
            "decomposition", 
            "spatial_sampling", 
            "nmse", 
            "spatial_coherence"
            ]
    headers = [
        "f [Hz]",
        "method",
        "sampling\n(float:%, int: N)",
        "reg_tol_c",
        "NMSE",
        "MAC",
    ]

    ## select MAC maximum, but print with NMSE and corresponding regularization tolerance
    maxidx = mdf[mdf.spatial_coherence.notnull()].groupby(by=columns[:3])["spatial_coherence"].idxmax()
    maxcoh = mdf.iloc[maxidx]

    measures = ["nmse","spatial_coherence"]
    groups = [ "decomposition", "f", "spatial_sampling", "reg_tol_c", ]
    groups_eval = groups[:4]

    mdf0 = mdf[mdf.spatial_coherence.notnull()]
    mdf0 = mdf0[mdf0.nmse < 20]
    mg0  = mdf0.groupby(groups, sort=False, as_index=False)

    mdf1 = mg0[measures].agg(["mean", "std", "count"])
    mg1  = mdf1.groupby( groups_eval, sort=False, as_index=False)
    mdf2i = mg1.idxmax()['spatial_coherence',"mean"]  # prioritize spatial_coherence in sorting
    mdf2  = mdf1.loc[mdf2i]
    mdf3i = mg1.idxmin()['nmse',"mean"]  # prioritize nmse
    mdf3  = mdf1.loc[mdf3i]

    ## check occurances
    # print(mdf2[(mdf2.spatial_coherence['count']<12)].to_string())
    # print(mdf2[(mdf2.spatial_coherence['count']>=12)].to_string())

    ## Performance figures
    # x-sampling plots
    tmp = mdf2.reset_index(level=[0, 1, 2], inplace=False)
    tmp[ "sampling_density"] = np.round(calc_density(tmp.spatial_sampling,tmp.f,1.7**2), 2)
    tmp["rsampling_density"] = np.round(tmp.sampling_density,1)
    freqs = REC_FREQ

    ### PLOT CONFIG 
    # config for each criterion in "measures"
    ylabels = ["NMSE [dB]", "$\gamma$" , "$\overline{\overline{\mathbf{x}}}$"]
    ylims = [[-23.5, .5], [-0.02,1.02], [1,100]]
    ylogs = [False, False, True]
    lloc = ["best", "best","lower right"]
    lopts = {
            'loc'            : 'lower right',
            'bbox_to_anchor' : (1.0,1.17),
            'fontsize'       : 16,
            'ncol'           : 2, }

    ## for these frequencies, plot across number of mics
    freqs = [1250, 1600]

    ## for these number of mics, plot across frequency
    nofmics = [80,160,320]

    ## for these sampling densities, plot across frequency
    tmp["rsampling_density"] = np.round(tmp.sampling_density*4)/4
    densities = [4.0, 12.0]

    ### PLOTTING
    # for each nofmics across freq
    xlims = [0.95 * 500, 1.01 * np.max(tmp.f)]
    # for iim, measure in enumerate(measures):
    for iim, measure in enumerate([measures[0]]):
        axes = plot_xy( 1, tmp,
            ["f"], measure,
            ["spatial_sampling", nofmics],
            ylims   = ylims[iim],
            xlims   = xlims,
            xlabels = ["Frequency [Hz]"],
            ylabels = ylabels[iim],
            ylog    = ylogs[iim],
        )
        for ii, ak in enumerate(zip(axes, nofmics)):
            fny = np.sqrt(ak[1])/1.7*343/2
            ak[0].axvline( fny, color = 'k', linestyle = ':', alpha=.4)
            ak[0].text( (fny-xlims[0])/np.diff(xlims) +.02,
                    .1, r"$ \overline{d_m} = \lambda/2$", transform = ak[0].transAxes, alpha=.5)
            settitle(ak[0], ii, "{: >4.0f} measurements".format(ak[1]))
        if show_legend(iim): axes[0].legend(**lopts) 
        savepath = "".join( ["./figures/", pdfname, "_xfreq_mic_{}".format(measure), ".pdf", ])
        print(" >", savepath)
        plt.savefig(savepath)
        plt.close()
