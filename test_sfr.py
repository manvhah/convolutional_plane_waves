import numpy as np
from read_mc import analysis
from Soundfield import Soundfield, default_soundfield_config
from SoundfieldReconstruction import SoundfieldReconstruction, default_reconstruction_config, find_decomposition_clear

level_p = lambda p: 20*np.log10(np.abs(p))-20*np.log10(2e-5)
level_u = lambda u: 20*np.log10(np.abs(u))-20*np.log10(5e-8)
level_i = lambda i: 10*np.log10(np.abs(i))-10*np.log10(1e-12)

def sfrtest_particle_velocity(decomposition='gpwe',measurement='monoplane',
        showplot = False, title=None):
    """ GPWE reconstruction incl particle velocity
    """
    valopts = default_soundfield_config(measurement)
    recopts, transopts = default_reconstruction_config(decomposition,)
    # if decomposition=='cpwe':
        # if '019' in measurement:
            # transopts.update(dict(reg_lambda = 1e-5, reg_mu = 1e-3, rho = 1e-5)) # psize 1 lmabda
        # else:
            # transopts.update(dict(reg_lambda = 1e-4, reg_mu = 1e-2, rho = 1e-2)) # psize 1 lmabda
    print(recopts,transopts)
    sfro  = SoundfieldReconstruction(
            measured_field = Soundfield(**valopts),
            transform_opts = transopts,
            **recopts)

    #reproducability sampling
    if '019' in measurement:
        sfro.measured_field._seed = 387309083
    sfro.measured_field.__dict__.update({'spatial_sampling' : -4.0})
    _=sfro.measured_field.sidx
    del sfro.measured_field.sample_idx
    sfro.measured_field.__dict__.update({'spatial_sampling' : -12.0})
    _=sfro.measured_field.sidx

    sfro.reconstruct();
    analysis(sfro, b_showplots = showplot, b_singlefig = 2, b_saveplots = False)

    print(np.abs(np.mean(np.mean(sfro.mf.u/sfro.rf.u,axis=1),axis=1)))

    import matplotlib.pyplot as plt
    from src._plot_tools import add_cbar
    fig = plt.figure(figsize=(7,6))
    plt.clf()
    from matplotlib.gridspec import GridSpec
    axes = [fig.add_subplot(gsi) for gsi in GridSpec(2,2,hspace=.3,wspace=.3)]

    def compare(axi,qi,labels):
        """ compare to quantities: [true, test]
        """
        improp = dict(
            vmax = np.max(qi[0]),
            vmin = np.min(qi[0]),
            cmap = 'viridis', 
            origin = 'lower', 
            )
        ax = axes[axi[0]]
        im = ax.imshow(qi[0],**improp)
        ax.set(xticks=[],yticks=[],ylabel=labels[0],title=labels[2])
        add_cbar(im,ax)
        ax = axes[axi[1]]
        im = ax.imshow(qi[1],**improp)
        ax.set(xticks=[],yticks=[],ylabel=labels[1])
        add_cbar(im,ax)
    XYZ = sfro.mf.r.T.reshape((3,*sfro.mf.shp))
    if '019' in measurement:
        aopt = dict(xlabel = 'x [m]', ylabel = 'y [m]')
        clabel = '[dB rel $<{\mathbf{p}_{ref}^2}>$]'
    else:
        clabel = '[dB SPL]'
        aopt = dict()
    aopt.update(aspect='equal',xlim=sfro.mf.extent[:2],ylim=sfro.mf.extent[2:])
    maxu = 15e-5

    jj = 1

    ax = axes[0]
    mfu = sfro.mf.u
    mfunorm = np.linalg.norm(mfu,axis=0)
    uopts = dict(cmap='binary_r',
            vmin=np.min(level_u(mfunorm))-1, 
            vmax=level_u(maxu),
            extent = sfro.mf.extent, origin='lower',interpolation='bilinear',
            )
    im = ax.imshow(level_u(mfunorm),**uopts)
    cb = add_cbar(im,ax,label="$L_u$ [dB rel $u_{ref}$]")
    Q = ax.quiver(XYZ[1,::jj,::jj],XYZ[0,::jj,::jj],mfu[0,::jj,::jj].real,mfu[1,::jj,::jj].real,color='black')
    qk = ax.quiverkey(Q, 0.58, 1.12, maxu, r'150$\,\mu$ms$^{-1}$',
            labelpos='E',color='k')
    ax.grid(False)
    ax.set(ylabel='y [$\lambda$]', **aopt)
    ax.text(-.3,1.05,"true",transform=ax.transAxes,fontsize=18)
    ax.set_title("Re$\{\mathbf{u}_{xy}\}$", loc="left", y=1.05)

    rfu = sfro.rf.u
    rfunorm = np.linalg.norm(rfu,axis=0)
    ax = axes[2]
    im = ax.imshow(level_u(rfunorm),**uopts)
    cb = add_cbar(im,ax,label="$L_u$ [dB rel $u_{ref}$]")
    Q = ax.quiver(XYZ[1,::jj,::jj],XYZ[0,::jj,::jj],rfu[0,::jj,::jj].real,rfu[1,::jj,::jj].real,color='black')
    ax.grid(False)
    ax.set(ylabel='y [$\lambda$]',  xlabel='x [$\lambda$]', **aopt)
    if not title:
        title = find_decomposition_clear(decomposition)
    ax.text(-.3,1.05,title,transform=ax.transAxes,fontsize=18)

    jj = 2

    ax = axes[1]
    maxI = 2e-6
    mfIJ = sfro.mf.IJ
    mfIJnorm = np.linalg.norm(mfIJ,axis=0)
    iopts = dict(cmap='pink',
            vmin=np.min(level_i(mfIJnorm))+10, 
            vmax = level_i(maxI)+3,
            extent = sfro.mf.extent, 
            origin='lower',interpolation='bilinear',
            )
    im = ax.imshow(level_i(mfIJnorm),**iopts)
    mfIJ *= np.clip(mfIJnorm,0,maxI)/mfIJnorm # clip the norm
    cb = add_cbar(im,ax,label="$L_I$ [dB rel $I_{ref}$]")
    Q = ax.quiver(XYZ[1,::jj,::jj],XYZ[0,::jj,::jj],mfIJ[0,::jj,::jj].real,mfIJ[1,::jj,::jj].real,color='black')
    qk = ax.quiverkey(Q, 0.63, 1.12, maxI, r'2$\,\mu$Wm$^{-2}$', 
            labelpos='E',color='k')
    ax.grid(False)
    ax.set(**aopt)
    ax.set_title("Re$\{\mathbf{I}_{xy}\}$", loc="left", y=1.05)

    ax = axes[3]
    rfIJ = sfro.rf.IJ
    rfIJnorm = np.linalg.norm(rfIJ,axis=0)
    rfIJ *= np.clip(rfIJnorm,0,maxI)/rfIJnorm # clip the norm
    im = ax.imshow(level_i(rfIJnorm),**iopts)
    cb = add_cbar(im,ax,label="$L_I$ [dB rel $I_{ref}$]")
    Q = ax.quiver(XYZ[1,::jj,::jj],XYZ[0,::jj,::jj],rfIJ[0,::jj,::jj].real/2,rfIJ[1,::jj,::jj].real/2,color='black')
    ax.grid(False)
    ax.set( xlabel='x [$\lambda$]', **aopt)
    plt.savefig(f'figures/particle_velocity_{decomposition}_{measurement}.pdf')
    print(f' > figures/particle_velocity_{decomposition}_{measurement}.pdf')

if __name__ == '__main__' :
    # csc paper
    sfrtest_particle_velocity('gpwe',measurement='monoplane', title='global')
    sfrtest_particle_velocity('cpwe',measurement='monoplane', title='conv. smooth')
    sfrtest_particle_velocity('lpwe',measurement='monoplane', title='local independent')
