import numpy as np
import matplotlib.pyplot as plt
from src._inverse_problem import solve_inv_problem
import scipy.signal as sig
from matplotlib.gridspec import GridSpec

N_wl = 24 # spacings per wavelength
domain_size_wl = 10
rx = np.linspace(0,domain_size_wl,1+domain_size_wl*N_wl)
# rx = np.linspace(0,9.85,198) # almost continuous for angle 4/21
kabs = 2*np.pi
patch_size = 1*N_wl+1

# pwe
Nangles = 21
kangle = np.linspace(0,1,Nangles, endpoint=True)*np.pi
k = kabs * np.array([np.cos(kangle), np.sin(kangle)]).T
H = np.exp(1j*np.outer(rx,k[:,0]))/len(rx)

def gen_soundfield(dist):
    ## generating measurements
    p = np.zeros(dist.shape, dtype=complex)
    phase = np.pi
    p += 1/(4*np.pi*dist) *np.exp(-1j*(kabs*dist+phase)) # monopole
    p /= np.max(np.abs(p)) # normalize
    return p

dx = np.diff(rx[:2])
# padding
padl = int(patch_size-1)
rxpad = np.linspace(np.min(rx)-dx*padl,np.max(rx)+dx*padl,len(rx)+2*padl)
zpad = lambda x: np.pad(x,padl).squeeze()
crop = lambda x: x if padl==0 else x[padl:-padl]

def wrap (x): # wraps along the last dimension
    x[...,:patch_size-1] += x[...,-patch_size+1:]
    return x[...,:-patch_size+1]

class Reconstruction():
    def __init__(self, p, D, x, P=None):
        self.p = p
        self.D = D
        self.x = x
        self.P = P

    def nmse(self, pref):
        from SoundfieldReconstruction import nmse
        return nmse(self.p,pref)

def g2l(p, plen):
    return np.array([p[ii:ii+plen] for ii in range(len(p)-plen+1)]).T

def l2g(P,normalize=True):
    # this one takes the mean, alternatively, the median can be considered as
    # suggested by Wohlberg
    plen = P.shape[0]
    pr = np.zeros(sum(P.shape)-1, dtype=complex)
    ol = np.zeros(sum(P.shape)-1, dtype=complex)
    for loc, pl in enumerate(P.T):
        pr[loc:loc+plen] += pl
        ol[loc:loc+plen] += 1
    if normalize:
        pr/=ol
    return pr

def patch_based_reconstruction(p, D, mode='lasso_cvx', reg_lambda=1e-3,win=None,zeropad=True):
    """
    reconstruction via independent subdomains
    """
    patch_size, nfuncs = D.shape
    if zeropad:
        P = g2l(zpad(p), patch_size) # reference
    else:
        P = g2l(p, patch_size) # reference
    if not win:
        win = np.ones((patch_size,1))
    P *= win

    coeff,Prec = solve_inv_problem(mode, D, P, dict(reg_lambda=reg_lambda))
    if zeropad:
        overlap = np.zeros(zpad(p).shape)+1e-16
    else:
        overlap = np.zeros(p.shape)+1e-16
    for key,val in enumerate(P.T):
        if np.sum(val!=0):
            overlap[key:key+patch_size] +=win.squeeze()
    prec = l2g(Prec,normalize=False)
    prec/=overlap

    if zeropad: prec = crop(prec)
    return Reconstruction(prec, D, coeff, Prec)


H_loc = np.exp(1j*np.outer(rx[:patch_size],k[:,0]))
if False:
    from scipy.signal.windows import hann 
    win = hann(patch_size)[:,np.newaxis]
else:
    win = np.ones((patch_size,1))
H_loc*=win
H_loc /=np.linalg.norm(H_loc,axis=0)

MAXITER = 500

def sporco_csc(D, y, mask, 
                # lmbda=1e-6, mu = 1e-4, rho=1e-5, 
                lmbda=1e-7, mu = 1e-5, rho=1e-6, 
              # lmbda=1e-6, mu = 1e-2, rho=1e-4, 
        verbose=False):
    from sporco.admm import cbpdn

    y = zpad(y)
    mask = zpad(mask)

    opt ={'Verbose': verbose, 'MaxMainIter': MAXITER,
        'HighMemSolve': True, 
        'RelStopTol': 1e-4,
        'AuxVarObj': False, 'RelaxParam': 0.7,
        'rho': rho, 
        'FastSolve': True,
        'AutoRho': {'Enabled': False, 'StdResiduals': False},
        }
    D_hires = D/np.linalg.norm(D,axis=0)
    D = D[:patch_size]
    D /= np.linalg.norm(D,axis=0)

    opt = cbpdn.ConvL2L1Grd.Options(opt)
    opt.update({'GradWeight': 'inv_dict',})
    problem = cbpdn.ConvL2L1Grd(
            D, y, 
            lmbda = lmbda, mu = mu, 
            W=mask, opt=opt,
            dimK = 0, dimN = 1)
            
    coeff = problem.solve()
    y_hat = problem.reconstruct().squeeze()
    coeff = coeff.squeeze()
    Y_hat = D.dot(coeff.T)
    y_hat = crop(y_hat)
    return Reconstruction(y_hat, D, coeff, Y_hat), problem


## TESTS

from cycler import cycler
cc = ( cycler(color=['tab:blue','tab:orange','tab:green']) + 
       cycler(linestyle=['-', '--', ':']))

def single_test(mlocwl=0.5, every_th_sample = 8):
    mloc = np.min(rx)-mlocwl # end-firing
    dist = np.abs(rx - mloc) # end-fire 

    def coef_scatter(ax,amps):
        xscale, yscale = 1.2, 1.0
        xs, ys = k[:,0]/kabs*xscale, k[:,1]/kabs*yscale+.5
        xs += np.max(dist)/2-0.8
        ax.fill(xs,ys,'k',alpha=.2)
        ax.scatter(xs,ys,c=np.log10(amps+1e-16),cmap='binary',zorder=2)

    p = gen_soundfield(dist)
    H = np.exp(1j*np.outer(rx,k[:,0]))/len(rx)
    H_loc = np.exp(1j*np.outer(rx[:patch_size],k[:,0]))

    midx = np.arange(0,p.size,every_th_sample)

    mask = np.zeros(p.shape)
    mask[midx] = 1
    pm = p*mask

    #global
    xg,_   = solve_inv_problem('ridge', H[midx], pm[midx], dict(reg_lambda=1e-8))
    rec_g = Reconstruction(H.dot(xg).squeeze(), H, xg)

    #patch-based
    rec_p = patch_based_reconstruction(pm, H_loc, 'lasso_cvx')

    #convolutional
    rec_c, csc = sporco_csc( H, pm, mask = mask)


    print(rec_g.nmse(p),rec_p.nmse(p),rec_c.nmse(p))

    ### plotting
    P = g2l(zpad(p), patch_size) # reference

    plt.rc('axes', prop_cycle=cc)
    plt.rc('lines', linewidth=3)
    plt.figure(1)
    plt.close()
    fig = plt.figure(1,figsize=(8,8))
    fig.clf()
    ax = [fig.add_subplot(gsi) for gsi in GridSpec(3,1,hspace=.15)]
    ax[0].scatter(0,0,c='gray',s=20**2,marker='$\odot$',label='monopole') # for legend
    ax[0].plot(dist,np.real(p),'k',label=r'true pressure',alpha=.4)
    ax[0].plot(dist[midx],np.real(pm[midx]),'k+',ms=10,label='observation',zorder=2)
    ax[0].set_ylim([-.7,1.2])
    ax[1].plot(dist,np.real(rec_c.p),'C0-')
    ax[1].plot(dist,np.real(rec_p.p),'C1--')
    ax[1].plot(dist,np.real(rec_g.p),'C2:')
    ax[1].plot(dist,np.real(p),'k',lw=3,alpha=.4,zorder=1.1)
    ax[1].set_ylim([-.7,1.2])
    ax[2].plot(dist,np.abs(rec_g.p-p),'C2:', zorder = 3.4,)
    ax[2].plot(dist,np.abs(rec_p.p-p),'C1--', zorder = 3.3,)
    ax[2].plot(dist,np.abs(rec_c.p-p),'C0-', zorder = 3.2,)
    ax[2].set_ylim([-.05,.35])
    llabels=[
            "{: <31}, NMSE={:>5.2f}dB".format( 'global',rec_g.nmse(p)),
            "{: <24}, NMSE={:>5.2f}dB".format( 'local independent',rec_p.nmse(p),),
            "{: <20}, NMSE={:>5.2f}dB".format( 'convolutional smooth',rec_c.nmse(p),),]
    for axi in ax:
        axi.tick_params(length=0, color="#555555")
        axi.set_xticks(range(int(np.max(rxpad)+0.5)))
        axi.set(xlim=[-.3, np.max(rx)-mloc+.1])
        axi.set_ylabel(ylabel='[a.u.]')
        axi.scatter(0,0,c='gray',s=20**2,marker='$\odot$',zorder=2)
    fig.align_ylabels(ax)
    [axi.set_xticklabels([]) for axi in ax[:-1]]
    ax[0].set(yticks=[-.5,0,.5,1],)
    ax[1].set(yticks=[-.5,0,.5,1],)
    ax[2].set(yticks=[0,.1,.2,.3],)
    topts = dict(bbox=dict(facecolor='white',edgecolor='white'))
    ax[0].text(-.02,1.0,'(a) measurement',transform=ax[0].transAxes,**topts)
    ax[1].text(-.02,1.0,'(b) reconstruction',transform=ax[1].transAxes,**topts)
    ax[2].text(-.02,1.0,'(c) error',transform=ax[2].transAxes,**topts)
    ax[0].legend(ncol=2,loc='upper right', bbox_to_anchor=(1.02, 1.05))
    ax[2].legend(llabels,ncol=1,loc='upper right', bbox_to_anchor=(1.02, 1.4),fontsize=17)
    ax[-1].set_xlabel(r"radial distance [$\lambda$]")
    fig.savefig('./figures/rec_1dcsc.pdf')
    print(' > ./figures/rec_1dcsc.pdf')


def nmse_distance_monopole(every_th_sample = 8):
    nmse = list()
    mlocwls = np.logspace(-1,np.log(4),11)
    for mlocwl in mlocwls:

        ## generate reference sound field
        mloc = np.min(rx)-mlocwl # end-firing
        dist = np.abs(rx - mloc) # end-fire 
        p = gen_soundfield(dist)

        ## decimated measurements
        midx = np.arange(0,p.size,every_th_sample)
        mask = np.zeros(p.shape)
        mask[midx] = 1
        pm = p*mask

        ## reconstruct global
        xg,_  = solve_inv_problem('ridge', H[midx], pm[midx], dict(reg_lambda = 0)) # all
        nmse_g = Reconstruction(H.dot(xg).squeeze(), H, xg).nmse(p)

        #patch-based
        nmse_p = patch_based_reconstruction(pm, H_loc,'lasso_cvx').nmse(p) # default

        #convolutional
        rec_c, csc = sporco_csc( H_loc, pm, mask = mask,)

        nmse.append([ nmse_g, nmse_p, rec_c.nmse(p), ])
    nmse = np.array(nmse)
    print(nmse)

    fig = plt.figure(8)
    plt.close()
    fig = plt.figure(8,figsize=(6,4))
    plt.rc('axes', prop_cycle=cc)
    plt.rc('lines', linewidth=3)
    ax = fig.add_subplot(111)
    ax.semilogx(mlocwls, nmse[:,0],'C2:',label = 'global',zorder=3.4)
    ax.semilogx(mlocwls, nmse[:,1],'C1--',label = 'local independent',zorder=3.3)
    ax.semilogx(mlocwls, nmse[:,2],'C0-',label = 'convolutional\nsmooth',zorder=3.2)
    ax.set(ylim=[-35,0], xlim=[np.min(mlocwls),np.max(mlocwls)],)
    ax.set_ylabel('NMSE [dB]',x=.1)
    ax.set_xlabel('min. radial distance [$\lambda$]',y=.1)
    ax.grid(True,which='both')
    ax.legend(loc='upper right',bbox_to_anchor=(1.05,1.05))
    fig.savefig('./figures/nmse_distance_monopole.pdf')
    print(' > ./figures/nmse_distance_monopole.pdf')

if __name__ == "__main__":
    # csc paper
    decimation_factor = 8 # 12 is lambda/2, 8 is lambda/3
    single_test(mlocwl=0.5, every_th_sample = decimation_factor)
    nmse_distance_monopole(every_th_sample = decimation_factor)
