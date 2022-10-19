import numpy as np
import matplotlib.pyplot as plt

from csc1d_radial import g2l, patch_based_reconstruction

def gen_soundfield(dist):
    p = np.zeros(dist.shape, dtype=complex)
    phase = np.pi
    print('phase',phase)
    p += 1/(4*np.pi*dist) *np.exp(-1j*(kabs*dist+phase)) # monopole
    p /= np.max(np.abs(p)*.85) # normalize
    return p

N_wl = 13 # spacings per wavelength
domain_size_wl = 4
rx = np.linspace(0,domain_size_wl,1+domain_size_wl*N_wl)
kabs = 2*np.pi
patch_size = 1*N_wl+1

p = gen_soundfield(rx+.7)
dx = np.diff(rx[:2])

# rxhi = np.linspace(min(rx)-dx/2,max(rx)+dx/2,200)
rxhi=rx
phi = gen_soundfield(rxhi+.7)
# pwe
Nangles = 21
kangle = np.linspace(0,1,Nangles, endpoint=True)*np.pi
k = kabs * np.array([np.cos(kangle), np.sin(kangle)]).T
H = np.exp(1j*np.outer(rx,k[:,0]))/len(rx)

P = g2l(p, patch_size) # reference

every_th_sample = 3
rng = np.random.default_rng(62355)
midx = np.array([ 2, 4, 7,  9, 10, 13, 15, 19, 24, 28, 30,35, 38,  43, 48])
mask = np.zeros(p.shape)
mask[midx.astype(int)] = 1
noise = rng.normal(size=len(rx))*.01
pm = (p+5*noise)*mask
pm[mask == 0] = np.NaN
Pm = g2l(pm, patch_size) # reference

# reconstruct field
Nangles = 21
kangle = np.linspace(0,1,Nangles, endpoint=True)*np.pi
k = kabs * np.array([np.cos(kangle), np.sin(kangle)]).T
H_loc = np.exp(1j*np.outer(rx[:patch_size],k[:,0]))
H_loc /=np.linalg.norm(H_loc,axis=0)
rec = patch_based_reconstruction((p+noise)*mask, H_loc, 'lasso_cvx',zeropad=False)
Pr = rec.P
pr = rec.p

def localmatrix(P):
    # this one takes the mean, alternatively, the median can be considered as suggested by Wohlberg
    plen,pnum = P.shape
    pr = np.zeros((pnum,sum(P.shape)-1), dtype=complex)
    for loc, pl in enumerate(P.T):
        pr[loc,loc:loc+plen] += pl
    pr[pr == 0] = np.NaN
    return pr

if __name__ == "__main__":
    import matplotlib as mpl
    
    mpl.rcParams['font.size'] = 18 # csc

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['mathtext.cal'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    plt.figure(1)
    plt.close()
    fig = plt.figure(1,figsize=(8,5))
    gs0 = GridSpec(6,9,left=.1,right=.87)
    ax=[
        fig.add_subplot(gs0[1:3,:3]),
        fig.add_subplot(gs0[3:,:3]),
        fig.add_subplot(gs0[3:,3]),
        fig.add_subplot(gs0[3:,4]),
        fig.add_subplot(gs0[3:,5]),
        fig.add_subplot(gs0[3:,6:]),
        fig.add_subplot(gs0[1:3,6:]),
        ]
    opt = dict(cmap = 'twilight',vmin=-1,vmax=1)
    num_patches = (len(rx)-patch_size+1)
    ax[0].plot(rxhi,phi.real,'k',label='$\mathbf{p}$ true')
    ax[0].plot(rx[midx],pm[midx].real,'ko',label = '$\mathbf{p}_{\mathrm{obs}}$ measured')
    winmask = np.zeros((len(p),num_patches))
    winmask[patch_size+3:2*patch_size+3] = 1
    winmask[:,patch_size+3] = 1
    ax[1].imshow(np.tile(p,(num_patches,1)).real,**opt,aspect='auto',alpha=.3)
    ax[1].imshow(winmask.T,cmap='binary',aspect='auto',vmin=0,vmax=1,alpha=.25)
    ax[1].imshow(localmatrix(P).real,**opt,aspect='auto')
    im = ax[2].imshow(P.T.real,**opt,aspect='auto')
    ax[3].imshow(winmask[:patch_size].T,cmap='binary',aspect='auto',vmin=0,vmax=1,alpha=.25)
    ax[3].imshow(Pm.T.real,**opt,aspect='auto')
    ax[4].imshow(winmask[:patch_size].T,cmap='binary',aspect='auto',vmin=0,vmax=1,alpha=.25)
    ax[4].imshow(Pr.T.real,**opt,aspect='auto')
    ax[5].imshow(np.tile(p,(num_patches,1)).real,**opt,aspect='auto',alpha=.3)
    ax[5].imshow(winmask.T,cmap='binary',aspect='auto',vmin=0,vmax=1,alpha=.25)
    ax[5].imshow(localmatrix(Pr).real,**opt,aspect='auto')
    ax[6].plot(rx,pr.real,'k--',label='$\hat{\mathbf{p}}$ reconstructed')

    cax = fig.add_subplot(gs0[1:3,3])
    cb =  plt.colorbar(mappable = im, cax=cax,
            ticks=[0],
            label='[real part a.u.]',
            )
    # cb.set_ticklabels([-1,0,1])
    c0 = ax[2].get_position()
    c1 = ax[6].get_position()
    cax.set_position([c0.x0+2*(c1.x0-c0.x0), c0.y0, c0.width*.20, c0.height])
    cb.set_label('[real part a.u.]')

    lopts = dict(ncol=2,loc=(.0,1.05),fontsize=16,
            handlelength=1.0,handletextpad = .4,
            frameon=False,framealpha=0,borderpad=0)
    ax[0].legend(**lopts)
    ax[6].legend(**lopts)
    ax[0].set_ylabel("sound pressure\n[real part a.u.]",fontsize=16)
    ax[1].set_ylabel("subdomain",fontsize=16)
    ax[2].set_title('$\mathbf{P}^{\mathsf{T}}$',fontsize=18)
    ax[3].set_title('$\mathbf{P}^{\mathsf{T}}_{\mathrm{obs}}$',fontsize=18)
    ax[4].set_title('$\hat{\mathbf{P}}^{\mathsf{T}}$',fontsize=18)

    [axi.set(xticks=[],yticks=[]) for axi in ax];
    ax[0].set(yticks=[0])
    ax[1].set(yticks=[0,num_patches-1],yticklabels=['1','$S$'])

    yl = ax[0].set_ylim()
    ax[6].set_ylim(yl)
    ax[6].set_xlim([min(rx), max(rx)])
    for axi in [ax[0],ax[6]]:
        axi.set_xlim([min(rx), max(rx)])
        axi.fill_between(
                np.array([rx[patch_size+3]-dx[0]/4, rx[2*patch_size+3]-dx[0]*3/4]),
                yl[0],yl[1], lw=0,color='k',alpha=.10,zorder=1.2)
        axi.axvline(rx[1*patch_size+3]-dx/4,c='#aaaaaa',ls='--',lw=1,zorder=1.2)
        axi.axvline(rx[2*patch_size+3]-dx*3/4,c='#aaaaaa',ls='--',lw=1,zorder=1.2)
    for axi in [ax[1],ax[5]]:
        axi.axvline(1*patch_size+3-.6,c='#aaaaaa',ls='--',lw=1)
        axi.axvline(2*patch_size+3-.5,c='#aaaaaa',ls='--',lw=1)
        axi.set(xticks=[0,num_patches-1,len(rx)-1],
                xticklabels=['1','$S$','$N$'])
    ax[1].set(yticks=[0,num_patches-1], yticklabels=['1','$S$'])
    for axi in ax[1:6]:
        axi.axhline(patch_size+3-.7,c='#aaaaaa',ls='--',lw=1)
        axi.axhline(patch_size+3+.6,c='#aaaaaa',ls='--',lw=1)
    for axi in ax[2:5]:
        axi.set(xticks=[0,patch_size-1],xticklabels=['1','$N_s$'])
    for axi in ax:
        axi.grid(visible=False)

    anndict = dict(
                xycoords='data',
                textcoords='offset points',
                annotation_clip = False,
                arrowprops=dict(
                    arrowstyle="fancy,tail_width=.6,head_width=1.2",
                    color='#666666',
                    ),zorder=4,
                )
    ax[1].annotate("",
                xy=(23.5,6),
                xytext=(0,20), 
                **anndict)

    thesis=False

    if thesis:
        label1 =  r"$\frac{1}{N_{nz}}\sum$"
        ax[6].text( rx[patch_size+5],-.60, 
                label1,
                fontsize=14,color='#666666')
    else: # csc paper
        label0 =  r"$\mathbf{R}_s$"
        label1 =  r"$\mathbf{W}\sum$"
        ax[0].text( rx[patch_size+9],-.75, 
                label0,fontsize=16, color='#666666',ha='center')
        ax[6].text( rx[patch_size+9],-.75, 
                label1, fontsize=16,color='#666666',ha='center')

    ax[5].annotate("",
                color='#666666', horizontalalignment='center',
                xy=(23.5,0),
                xytext=(0,-20), 
                **anndict)
    ax[1].annotate("$N_S$",color='#666666',
                xy=(23.5,41), xycoords='data',
                xytext=(-30, -20), textcoords='offset points',
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-[,widthB=1.2",
                    color='#666666',shrinkB=0,
                    connectionstyle="angle3,angleA=0,angleB=90"
                    ),
                )

    # [axi.set(aspect='equal') for axi in ax[2:]];
    if thesis:
        plt.savefig('figures/thesis_partitioning.pdf',bbox_inches='tight',pad_inches=0.0)
    else:
        plt.savefig('figures/partitioning.pdf',bbox_inches='tight',pad_inches=0.0)
        print(" > figures/partitioning.pdf")
