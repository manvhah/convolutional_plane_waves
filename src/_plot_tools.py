import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

titleopts = {'loc' : 'left'}

def plot_dictionary(D, K, patch_size, figure, savepath=None, resolution = None,
        title = None, cmap = 'twilight_shifted', 
        grid = (2,2), order = ['abs','real','phase','imag'], overlay = None):

    # dictionary dimensions
    n, K_total = D.shape

    # patch size
    n_r = patch_size[0]

    # patches per row / column
    if type(K) is tuple:
        K01 = np.array(K)
        K = np.prod(K)
    elif K < 200:
        K01 = np.array([int(np.ceil(K/10)), 10])
    else: 
        K01 = np.array([int(np.ceil(K/25)), 25])
    b_ylabel = int(K01[0] < K01[1])

    # we need n_r*K_r+K_r+1 pixels in each direction
    dim = n_r * K01 + K01 + 1
    V = np.zeros((dim[0], dim[1])) * np.max(D)
    V *= np.nan

    # compute the patches
    if K < K_total:
        selection_idx = np.random.choice(K_total,K)
    else:
        selection_idx = range(np.min([K_total,K]))
    patches = [np.reshape(D[:, i], (n_r, n_r)) for i in selection_idx]

    # place patches
    for i in range(K01[0]):
        for j in range(K01[1]):
            try:
                V[ i * n_r + 1 + i:(i + 1) * n_r + 1 + i ,
                   j * n_r + 1 + j:(j + 1) * n_r + 1 + j ] = \
                    patches[i * K01[1] + j]
            except:
                pass

    if type(figure) is int:
        fig = plt.figure(figure, figsize=(6,3))
    elif type(figure) is tuple:
        fig = plt.figure(1, figsize=figure)
    else: # figure handle
        fig = figure
    plt.clf()

    if b_ylabel:
        gs0 = GridSpec(*grid, figure=fig, 
                hspace = .16,
                top = .90, bottom = .02,
                left = .05, right = .98)
    else:
        gs0 = GridSpec(*grid, figure=fig, 
                wspace = .30,
                top = .90, right = .95,
                )

    axes = []
    for gs, quantity in zip(gs0, order):
        axes.append(fig.add_subplot(gs))
        if quantity.lower() == 'abs':
            im = axes[-1].imshow(np.abs(V), 
                    cmap = 'binary', 
                    vmin = 0,
                    )
            label_text = "$|\cdot|$" # Mag
        elif quantity.lower() == 'real':
            im = axes[-1].imshow(np.real(V), 
                    cmap='RdBu', 
                    )
            label_text = "$\Re$"
        elif quantity.lower() == 'phase':
            im = axes[-1].imshow(np.angle(V)/np.pi, 
                    cmap='twilight_shifted', alpha=1.0, 
                    vmin=-1, vmax=1, 
                    )
            label_text = r"$\angle$" # Phase
        elif quantity.lower() == 'imag':
            im = axes[-1].imshow(np.imag(V),
                    cmap='BrBG', 
                    alpha=1.0, 
                    )
            label_text = "$\Im$"

        if np.any(overlay):
            ol_mask = np.zeros((dim[0], dim[1])) * np.max(D)
            for i in range(K01[0]):
                for j in range(K01[1]):
                    try:
                        ol_mask [ i * n_r + 1 + i:(i + 1) * n_r + 1 + i ,
                           j * n_r + 1 + j:(j + 1) * n_r + 1 + j ] = \
                            1-overlay[i * K01[1] + j]
                    except:
                        pass
            axes[-1].imshow(ol_mask*.5, 
                    alpha = ol_mask*.2, 
                    cmap = 'Reds',
                    vmin = 0, vmax = 1)

        xticks = [ii*(n_r+1) for ii in range(K01[1]+1)]
        yticks = [ii*(n_r+1) for ii in range(K01[0]+1)]

        axes[-1].tick_params(length=5, color="#555555")
        axes[-1].tick_params(axis="x", 
                bottom=False, labelbottom=False,
                top   =True , labeltop=True)
        axes[-1].set_xticks(xticks[:2])
        axes[-1].set_yticks(yticks[:2])
        axes[-1].set_xticklabels(["0",r"$\lambda$"], )
        axes[-1].set_yticklabels(["0",r"$\lambda$"], )
        axes[-1].grid(0)

    if title is None:
        if resolution:
            title = ("$\mathbf{D} = [\mathbf{d}_1, \mathbf{d}_2, \dots, \mathbf{d}_N]$"+
                    ", N = {:.0f}, side length {:.1f} $\lambda$ ({}x{})"
                    .format(D.shape[1], ((patch_size[0]-1)*resolution)[0], *patch_size))
        else:
            title = ("D$ = [d_1, d_2, \dots, d_N]$"+" size {}x{}x{}"
                    .format(D.shape[1], *patch_size))

    # if title & (K < K_total): title = " ".join([title,"({:.0f}/{:.0f})".format(min(K,K_total), K_total)])
    plt.text(.5,.97,title,ha='center',va='top', transform=fig.transFigure, wrap=False)

    if savepath != None:
        if 'pdf' in savepath:
            plt.savefig(savepath)
        if 'tex' in savepath:
            import tikzplotlib
            tikz.save(savepath)
    else:
        plt.savefig(figpath() + 'decomposition_' + 
                title.split(' ')[0].split('_')[0] + '.pdf')


def add_cbar(im, ax, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    axsize = ax.bbox._bbox._points_orig
    if ((axsize[1,0]-axsize[0,0]) > .5):
        cax = divider.append_axes('right', size='4%', pad=0.04)
    else:
        cax = divider.append_axes('right', size='4%', pad=0.04)
    cb =  plt.colorbar(mappable = im, cax=cax)
    if label: cb.set_label(label)
    return cb


def annotatep(handle, text, x=1.15, y=1.30, usetex = None, align='right',color="#666666"):
    transform = handle.transAxes
    handle.text( x, y, text,
        transform=transform,
        wrap=False,
        color=color,
        horizontalalignment=align,
        verticalalignment = "bottom",
        usetex = usetex,
    )
