import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from src._monopole import random_phase
from src._const import c0

#todo: define apertures in sets, concatenate all _rms and _pms

def sample_surface(dshape, density, 
        mode="sample", pos=None, min_distance=0.07, seed=None):
    """
    reduce given complex data, set all other points to 0
    parameters:
        dshape        dshape, 2d tuple
        density       N (int) number of data points or loss factor (float)
        mode          'sample' to draw N samples from uniform
                      distribution (default) 
                      'grid' to grid_sample space
                      'lhs' for latin hypercube sampling
        pos           in case of 'sample', provide position vector to ensure a...
        min_distance  ..between the samples
        seed          rng seed, can be set for reproducability (default:None)
    """
    if seed:
        np.random.seed(seed)

    def get_grid(grid_shape,data_shape):
        Lx,Ly = data_shape
        Nx, Ny = grid_shape
        # floor: decrease the spacing to match grid
        divx = np.floor(Lx/Nx) 
        divy = np.floor(Ly/Ny)
        vx = np.arange(np.floor((Lx-1)%divx/2.00),Lx,divx)
        vy = np.arange(np.floor((Ly-1)%divy/2.00),Ly,divy)
        Vx,Vy = np.meshgrid(vx,vy)
        samples = (Vx+Vy*Lx).ravel().astype(int)
        if len(samples) != ((Nx+1)*(Ny+1)):
            print("spatial sampling failed...",len(samples),"instead of",(Nx+1)*(Ny+1),"points")
        return samples

    if not float(density) == 0.0:
        if mode == "grid":
            Nx = np.ceil(np.sqrt(density))-1
            Ny = np.floor(np.sqrt(density))-1
            return get_grid((Nx,Ny),dshape)
        else: # random sample
            dlen = np.prod(dshape)

            if type(density) == float:
                n_samples = int(dlen * (1 - density))
            elif type(density) == int:
                n_samples = density

            # if (mode == 'lhs'):
            # # apply latin hypercube sampling
            # from pyDOE import lhs
            # lhs(2, n_samples, 'corr', iteration)

            if np.any(pos):
                from scipy.spatial import distance

                pick = np.empty(n_samples, dtype=int)
                kk = 1
                pick[0] = np.random.choice(dlen)
                while True:
                    pick[kk] = np.random.choice(dlen)
                    if np.min(distance.pdist(pos[pick[: kk + 1], :])) >= min_distance:
                        kk += 1
                    if kk == n_samples:
                        break
            else:
                pick = np.random.choice(dlen, n_samples, replace=False).astype(int)
            return pick
    else:
        return np.arange(np.prod(dshape))


def _get_pidx(shp, patch_size):
    counter = np.arange(np.prod(shp))
    ref_idx = np.tile(np.reshape(counter, shp), (2, 2))  # should be 3x3

    prange = [shp[0] - patch_size[0] + 1, shp[1] - patch_size[1] + 1]

    pidx = np.zeros((np.prod(prange), *patch_size), dtype=int)
    for ii in range(prange[0]):
        for jj in range(prange[1]):
            pidx[ii * prange[1] + jj, :, :] = ref_idx[
                ii : ii + patch_size[0], jj : jj + patch_size[1]
            ]
    return pidx


def extract_patches_2d(A, patch_size):
    """ extracting MN subpatches of size patch_size mxn from matrix A with
    dimensions MxN, using no overlap """
    pidx = _get_pidx(A.shape[:2], patch_size)
    return A.ravel()[pidx], pidx


def reconstruct_from_patches_2d(patches, Adim, _=None , return_var = False):
    """ combining MN subpatches of size patch_size mxn to matrix A with
    dimensions Adim = (M,N), using overlap add """

    prange = (Adim[0] - patches.shape[1] + 1, Adim[1] - patches.shape[2] + 1)

    pidx = _get_pidx(Adim, patches.shape[1:])
    A = np.zeros(np.prod(Adim), dtype=patches[0].dtype)
    # mean 
    _, cidx_full = np.unique(pidx.ravel(), return_counts=True) # assume full overlap
    cidx_r = np.zeros(np.prod(Adim))
    cidx_i = np.zeros(np.prod(Adim))
    for ijdx, pid in enumerate(pidx):
        A[pid] += patches[ijdx]
        if np.linalg.norm(np.real(patches[ijdx])) > 1e-6:
            cidx_r[pid] += 1
        if np.linalg.norm(np.imag(patches[ijdx])) > 1e-6:
            cidx_i[pid] += 1
    cidx_r[cidx_r == 0] = np.inf
    cidx_i[cidx_i == 0] = np.inf
    scaling_r = 1 / cidx_r
    scaling_i = 1 / cidx_i
    A = np.real(A)*scaling_r + 1j * np.imag(A)*scaling_i

    if return_var:
        # variance
        A_var = np.zeros(np.prod(Adim), dtype=patches[0].dtype)
        cidx  = np.zeros(np.prod(Adim))
        for ijdx, pid in enumerate(pidx):
            A_var[pid] += (patches[ijdx] - A[pid])**2
            if np.linalg.norm(patches[ijdx]) > 1e-6:
                cidx[pid] += 1
        scaling = 1/cidx
        A_var *= scaling
        return np.reshape(A, Adim), np.reshape(A_var, Adim)
    else:
        return np.reshape(A, Adim), None


def calc_density(N,freq, area): # must match with soundfield
    # ## per mics circle of radius lambda/2 
    # return N / area * (np.pi/4 * (c0 / freq) ** 2)
    ## per area of wavelength**2
    return N / area * (c0/freq)**2
    # ## 1D per length of wavelength
    # return np.sqrt(N / area) * (c0 / freq)


def calc_number_of_mics(dens,freq,area):
    ## per area of wavelength**2
    return dens * area / (c0/freq)**2


def plot_density_mapping():
    freq      = np.linspace(500, 2e3, 70)
    Nlev = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    Dlev = [1,2,3,4,8,16,32]
    area      = 1.7**2 # room 019

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(5, figsize = (11,5.5))
    plt.clf()
    gs = GridSpec(1,2, figure=fig, wspace = .25, bottom = .13, top = .9)
    ## density for given N
    ax = fig.add_subplot(gs[0])
    densities = np.linspace(1., 32., 100)
    N = np.array([calc_number_of_mics(dens, freq, area) for dens in densities])
    # mirroring fixed values for N in SPATIAL_SAMPLING
    cs = plt.contour(freq, densities, np.log2(N), levels=np.log2(Nlev), cmap='copper')
    fmt = {}
    for ll,lev in enumerate(cs.levels):
        fmt[lev] = "N = {:.0f}".format(2**lev)
    ax.set_yscale('log')
    ax.set_yticks(Dlev)
    ax.set_yticklabels(Dlev)
    plt.title('Number of microphones $N$')
    plt.ylabel('Density $D$ (Nyq at 4.0)')
    plt.xlabel('Frequency [Hz]')
    plt.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=14)
    plt.grid(True)

    ## N for density
    ax = fig.add_subplot(gs[1])
    nofmics = np.logspace(3,11,num=100,base=2)
    D = np.array([calc_density(nn, freq, area) for nn in nofmics])
    cs = plt.contour(freq, nofmics, np.log2(D), levels=np.log2(Dlev), cmap='copper')
    fmt = {}
    for ll,lev in enumerate(cs.levels):
        fmt[lev] = "D = {:.0f}".format(2**lev)
    ax.set_yscale('log')
    ax.set_yticks(Nlev[1:-1])
    ax.set_yticklabels(Nlev[1:-1])
    plt.title('Density $D$ (N microphones per area $\lambda ^2$)')
    plt.ylabel('Number of microphones $N$')
    plt.xlabel('Frequency [Hz]')
    plt.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=14)
    plt.grid(True)

    plt.savefig("./figures/density_mapping.pdf")
    plt.close()

# def sphere_field(k,x):
    # # field by a sphere
    # # p_sphere = sum Am Hm Pm (cos theta ejwt)
    # from scipy.special import lpmv, hankel2 #spherical or cylindrical?

    # r_dist = np.linalg.norm(x,1)
    # kabs   = np.linalg.norm(k)
    # theta  = np.dot(k,x)/r_dist*2*pi/kabs

    # pressure = np.empty(r_dist.shape)
    # for ii in range(10):
        # legendre = lpmv(ii, ,np.cos(theta))
        # legendre *= 1/lpmv(ii, ,1)
        # pressure += Am * hankel2(ii, kabs*r_dist) * legendre


class Soundfield():
    def __init__(self,
            measurement  = 'synthesized_diffuse',
            frequency    = 1e3, dx = .05, psize = None,
            min_distance = .07,
            seed         = None,
            **kwargs):
        self.measurement  = measurement
        self.f            = frequency
        self.dx           = dx
        self.psize        = psize
        self.min_distance = min_distance
        self.pad_method   = None
        self.spatial_sampling  = 0.0
        self.loss_mode    = 'sample'
        self.snr          = 100 # default snr
        self._seed        = seed
        self._rng         = np.random.default_rng(seed)
        self._update_flag = 0
        self.__dict__.update(kwargs)

        if not hasattr(self.f,"__iter__"):
            self.f  = np.array([float(self.f)])

        if (not self.psize) and ('patch_size_in_lambda' in kwargs):
            self.psize = self._gen_psize()

        if "1p" in self.measurement: # forcing single patch sized sound field
            self.measurement = self.measurement.replace("1p","")
            self.b_single_patch = True
        else:
            self.b_single_patch = False

        self._gen_field()

    def _gen_psize(self, freq = None):
        if not np.any(freq):
            freq = self.frequency
        return tuple( (np.ceil(c0/(np.min(freq) * self.dx)
            * self.patch_size_in_lambda+1) *np.ones(2)).astype(int))

    def _gen_field(self):
        ## import measurement data or generate

        if np.all([
            hasattr(self, '_pm'),
            hasattr(self, 'shp'),
            hasattr(self, 'psize')]):
            return

        def select_freq_idx(fdata):
            idx = np.empty(len(self.f))
            for ii,freq in enumerate(self.f):
                idx[ii] = np.abs(fdata - freq).argmin() # find closest freq
            return idx.astype(int)

        def get_position(data,prec_in_mm = 5):
            rvec = np.round(data['xyz']/prec_in_mm)*prec_in_mm/1000
            iorder = np.argsort(rvec@np.array([1e3,1,1e6]), axis=0)
            return rvec[iorder,:], iorder

        def get_measurement(data,fidx, prec_in_mm = 5):
            rm, iorder = get_position(data, prec_in_mm)
            pm = data['response'][fidx,:][:,iorder].T
            return pm, rm

        if 'xyplane' in self.measurement:
            # plane wave in x
            from src._sphere_sampling import fibonacci_sphere
            self.rdim = np.array(self.rdim)

            x, y, z = np.arange(self.rdim[0,0],self.rdim[0,1],self.dx),\
                      np.arange(self.rdim[1,0],self.rdim[1,1],self.dx),\
                      0# np.arange(self.rdim[2,0],self.rdim[2,1],self.dx)
            yy, zz, xx = np.meshgrid(y,z,x) # order to separate data easier
            x_grid = np.array([z for z in zip(xx.ravel(),yy.ravel(),zz.ravel())])
            iorder = np.argsort(x_grid@np.array([1e3,1,1e6]), axis=0)
            x_grid = x_grid[iorder,:]

            kabs = 2*np.pi*self.frequency/c0
            self.plane_k = kabs * np.array([1,1,0])/np.sqrt(2)
            pfield = np.exp(-1j*np.dot(x_grid,self.plane_k)).squeeze()

            phase = random_phase(1, seed = self._seed)
            gain  = 1/np.max(np.abs(pfield))
            plane = pfield.ravel()*phase*gain
            uplane= plane[np.newaxis,:]*self.plane_k[:,np.newaxis]/(1.2*c0*kabs)

            self._rm, self._pm, self._um = x_grid, pfield[:,np.newaxis], uplane
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])
            self._tm = np.zeros(self._pm.shape)

        elif 'monoplane' in self.measurement:
            # monopole in free field at (0,0,0) plus plane wave
            # seen in a plane at distance lambda
            from src._sphere_sampling import fibonacci_sphere
            self.rdim = np.array(self.rdim)

            x, y, z = np.arange(self.rdim[0,0],self.rdim[0,1]+self.dx,self.dx),\
                      np.arange(self.rdim[1,0],self.rdim[1,1]+self.dx,self.dx),\
                      np.arange(self.rdim[2,0],self.rdim[2,1]+self.dx,self.dx)[0]
            yy, zz, xx = np.meshgrid(y,z,x) # order to separate data easier
            x_grid = np.array([z for z in zip(xx.ravel(),yy.ravel(),zz.ravel())])
            iorder = np.argsort(x_grid@np.array([1e3,1,1e6]), axis=0)
            x_grid = x_grid[iorder,:]

            kabs = 2*np.pi*self.f/c0
            dist = np.linalg.norm(x_grid, axis=1)
            pphase = random_phase(1,seed=self._seed)
            mono = 1/(4*np.pi*dist)*np.exp(-1j*kabs*dist)*pphase
            mono = mono.ravel()/np.linalg.norm(mono)
            radial = x_grid.T/np.linalg.norm(x_grid,axis=1)[np.newaxis,:]
            umono  = (mono/(1.2*343)*(1+1/(1j*kabs*dist)))[np.newaxis,:]*radial

            if not hasattr(self,'num_waves'):
                self.num_waves = 1234
            kindx = self._rng.choice(self.num_waves)
            self.plane_k = kabs * fibonacci_sphere( samples = self.num_waves, randomize=False)[kindx,:]
            uphase = random_phase(1,self._seed)
            plane = np.exp(-1j*np.dot(x_grid,self.plane_k))*uphase
            plane = plane/np.linalg.norm(plane)
            uplane = plane[np.newaxis,:]*self.plane_k[:,np.newaxis]/(1.2*c0*kabs)

            self._rm, self._pm, self._um = x_grid, (mono+plane)[:,np.newaxis], umono+uplane
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])
            self._tm = np.zeros(self._pm.shape)
            print(self.plane_k/kabs)


        elif 'mono' in self.measurement:
            # monopole in free field at (0,0,0) 
            # seen in a plane rz
            self.rdim = np.array(self.rdim)

            x, y, z = np.arange(self.rdim[0,0],self.rdim[0,1]+self.dx,self.dx),\
                      np.arange(self.rdim[1,0],self.rdim[1,1]+self.dx,self.dx),\
                      np.arange(self.rdim[2,0],self.rdim[2,1]+self.dx,self.dx)[0]
            yy, zz, xx = np.meshgrid(y,z,x) # order to separate data easier
            x_grid = np.array([z for z in zip(xx.ravel(),yy.ravel(),zz.ravel())])
            iorder = np.argsort(x_grid@np.array([1e3,1,1e6]), axis=0)
            x_grid = x_grid[iorder,:]

            kabs = 2*np.pi*self.f/c0
            dist = np.linalg.norm(x_grid, axis=1)
            mono = 1/(4*np.pi*dist)*np.exp(-1j*kabs*dist)
            gain = 1/np.max(np.abs(mono))
            mono = mono.ravel()*gain*random_phase(1, seed = self._seed)

            radial = x_grid.T/np.linalg.norm(x_grid,axis=1)[np.newaxis,:]
            umono  = (mono/(1.2*343)*(1+1/(1j*kabs*dist)))[np.newaxis,:]*radial

            self._rm, self._pm, self._um = x_grid, mono[:,np.newaxis], umono
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])
            self._tm = np.zeros(self._pm.shape)

        elif 'plane' in self.measurement:
            # random angle plane wave
            from src._sphere_sampling import fibonacci_sphere
            self.rdim = np.array(self.rdim)

            x, y, z = np.arange(self.rdim[0,0],self.rdim[0,1],self.dx),\
                      np.arange(self.rdim[1,0],self.rdim[1,1],self.dx),\
                      0# np.arange(self.rdim[2,0],self.rdim[2,1],self.dx)
            yy, zz, xx = np.meshgrid(y,z,x) # order to separate data easier
            x_grid = np.array([z for z in zip(xx.ravel(),yy.ravel(),zz.ravel())])
            iorder = np.argsort(x_grid@np.array([1e3,1,1e6]), axis=0)
            x_grid = x_grid[iorder,:]

            if not hasattr(self,'num_waves'): self.num_waves = 1234
            kindx = self._rng.choice(self.num_waves)
            kabs = 2*np.pi*self.frequency/c0
            self.plane_k = kabs * fibonacci_sphere(samples = self.num_waves, randomize=False)[kindx,:]
            pfield = np.exp(-1j*np.dot(x_grid,self.plane_k)).squeeze()

            phase = random_phase(1, seed = self._seed)
            gain  = 1/np.max(np.abs(pfield))
            plane = pfield.ravel()*phase*gain
            uplane= plane[np.newaxis,:]*self.plane_k[:,np.newaxis]/(1.2*c0*kabs)

            self._rm, self._pm, self._um = x_grid, pfield[:,np.newaxis], uplane
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])
            self._tm = np.zeros(self._pm.shape)

        elif 'scatter_sphere' in self.measurement:
            # random phase and arrival angle plane wave scattering on
            # sphere of radius lambda at (0,0,0), 
            # 3D sound field slice in x-y plane
            ## TODO:  remote coordinates in sphere, 
            ## set to nan when creating 2D spatial pressure distribution
            from spherical import spherical_scatter_plane_wave
            from src._sphere_sampling import fibonacci_sphere
            radius = self.scatter_sphere_radius

            self.rdim = np.array(self.rdim)
            x, y, z = np.arange(self.rdim[0,0],self.rdim[0,1],self.dx),\
                      np.arange(self.rdim[1,0],self.rdim[1,1],self.dx),\
                      0# np.arange(self.rdim[2,0],self.rdim[2,1],self.dx)
            yy, zz, xx = np.meshgrid(y,z,x) # order to separate data easier
            x_grid = np.array([z for z in zip(xx.ravel(),yy.ravel(),zz.ravel())])
            iorder = np.argsort(x_grid@np.array([1e3,1,1e6]), axis=0)
            x_grid = x_grid[iorder,:]

            direction =  fibonacci_sphere(samples = 1234, randomize = True)[self._rng.choice(1234)]
            direction = np.array([1,0,0])
            k = 2*np.pi*self.frequency/c0 * direction
            pfield, _ , outer_coords = spherical_scatter_plane_wave(x_grid, k, radius)
            pfield = np.squeeze(pfield)*random_phase(1, seed = self._seed)

            self._rm, self._pm = x_grid, pfield[:,np.newaxis]
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])
            self._tm = np.zeros(self._pm.shape)

        elif 'synthesized_diffuse' in self.measurement:
            from src.dictionaries import \
                    sinc_kernel_dictionary
            self.rdim = np.array(self.rdim)
            x, y, z = np.arange(self.rdim[0,0],self.rdim[0,1],self.dx),\
                      np.arange(self.rdim[1,0],self.rdim[1,1],self.dx),\
                      0# np.arange(self.rdim[2,0],self.rdim[2,1],self.dx)
            yy, zz, xx = np.meshgrid(y,z,x) # order to separate data easier
            x_grid = np.array([z for z in zip(xx.ravel(),yy.ravel(),zz.ravel())])
            iorder = np.argsort(x_grid@np.array([1e3,1,1e6]), axis=0)
            self._rm = x_grid[iorder,:]
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])
            self._pm = np.empty((np.prod(self.shp),len(self.f),self.nof_apertures),dtype=complex)
            for ii,ff in enumerate(self.f):
                self._pm[:,ii,:] = \
                    sinc_kernel_dictionary(self.shp, 2*np.pi/c0*ff*self.dx, self.nof_apertures) \
                + 1j*sinc_kernel_dictionary(self.shp, 2*np.pi/c0*ff*self.dx, self.nof_apertures)
            self._pm *=57 # approx mean of 011
            # print(np.mean(np.linalg.norm(self._pm,axis=0).ravel()))
            self.aperture_idx = 0

        elif '011' in self.measurement:
            from src._read_data import read_hdf
            if 'h5' in self.measurement:
                filepath = self.measurement
            else:
                filepath = "./../z_data/room011.h5"
            try:
                data = read_hdf(filepath)
            except:
                data = read_hdf('/work1/manha/z_data/room011.h5')
            fidx = select_freq_idx(data['Measurements_Training']['frequency'][:,0])

            if 'single' in self.measurement:
                self._pm, self._rm = get_measurement(data['Measurements_Training'],fidx)
                # self._tm = np.zeros(np.size(self._pm))
                # self._tm = data['Measurements_Training']['timestamp'][iorder]
                self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])

            else: ## load arrays
                rm, iorder = get_position(data['Measurements_Training'])
                self._rm = np.empty((len(iorder),3,7))
                self._pm = np.empty((len(iorder),len(fidx),7),dtype=complex)
                # self._tm = np.empty((len(iorder),3,7))

                self._pm[:,:,0], self._rm[:,:,0] = get_measurement( 
                        data['Measurements_Training'],fidx)
                # self._tm[:,:,0] = data['Measurements_Training']['timestamp']

                for iim in range(1,self.nof_apertures):
                    # round precision to 5 mm grid and transform to mm->m
                    self._pm[:,:,iim], self._rm[:,:,iim] = get_measurement(
                            data['Measurements_P{}'.format(iim)],fidx)
                    # self._tm[:,:,iim] = data['Measurements_P{}'.format(iim)]['timestamp']
                    self._aperturesize = self._pm[:,:,iim].shape[0]*np.ones(7)

                self.aperture_idx = 0 # default aperture for fp...

                self._tm = np.zeros(np.size(self._pm))
                self.shp = tuple([len(np.unique(self._rm[:,0,0])), len(np.unique(self._rm[:,1,0]))])
                print(np.mean(np.linalg.norm(self._pm,axis=0).ravel()))

        elif ('019' in self.measurement) or ('classroom' in self.measurement):
            from src._read_data import read_hdf
            if 'h5' in self.measurement:
                filepath = self.measurement
            else:
                filepath = "./data/classroom_frequency_responses.h5"
            try:
                data = read_hdf(filepath)
            except:
                print("!\n! Measurement *.h5 with sound field data not found. Please specify correct path.\n!")

            fidx = select_freq_idx( data['aperture_z1866mm']['frequency'])
            self._pm, self._rm = get_measurement(
                    data['aperture_z1866mm'], fidx, prec_in_mm = 1)
            if 'full' in self.measurement:
                self._rm[1,1] = 1.9
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])
            
            # self._tm = data['aperture_z1866mm']['outside_temperature'] # not measured for official dataset
            # self._tm = self._tm[iorder]
            self._tm = np.zeros(np.size(self._pm))

        if hasattr(self, "b_single_patch"): # check for legacy compatibility
            if self.b_single_patch:
                self._get_single_patch()

        self.fdim = np.array([np.min(self.r,axis=0), np.max(self.r,axis=0)]).T
        dr_2      = np.diff(self.fdim).ravel()/(np.array([*self.shp,2])-1)/2
        self.fdimdelta = self.fdim + np.array([-dr_2, dr_2]).T

        self.areaxy = np.prod((np.max(self.r, axis=0)-np.min(self.r, axis=0))[:2])

    def _reset(self,**kwargs):
        def checkdelattr(dictionary, attribute): 
            if hasattr(dictionary, attribute): 
                delattr(dictionary, attribute)
        checkdelattr(self,'_pm')
        checkdelattr(self,'_rm')
        checkdelattr(self,'shp')
        self.__dict__.update(kwargs)
        self._gen_field()

    def _get_single_patch(self):
        """ selecting single patch from """
        # 0 select patch
        if not hasattr(self,"patch_number"):
            self.patch_number = self._rng.choice(self.pidx.shape[0])
            print("random patch number:",self.patch_number)
        self._pidx = self.pidx[self.patch_number,:,:]
        idx        = self.pidx.ravel()

        # 1 store pm, rm, t in primary variable fields
        self.pm    = self._pm[idx]
        self.um    = self._um[:,idx]
        self.rm    = self._rm.reshape((-1,3))[idx]
        if hasattr(self, '_tm'):
            self.tm    = self._tm[idx]

        # update pidx to range, shp == psize
        self.pad_method = None
        self.shp   = self.psize
        delattr(self, '_pidx')
        _ = self.pidx

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)
        self._update_flag = 1

    @property
    def fmask(self):
        fmask = np.zeros(self.fp.shape[0], dtype=self.fp.dtype)
        fmask[self.sidx] = 1
        return fmask

    @property
    def mask(self):
        return self.fmask.reshape(self.shp)

    @property
    def mic_density(self): 
        return calc_density(self.N, self.f, self.areaxy)

    @property
    def avg_distance(self): 
        from scipy.spatial import Delaunay
        from itertools import combinations
        points = self.r[self.sidx][:,:2]
        if len(points) <= 1: return 0
        if len(points) == 2: return np.linalg.norm(points[0]-points[1])
        # calc delaunay triangulation
        tri = Delaunay(points)

        #plot
        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.triplot(x[:,0],x[:,1],tri.simplices)
        # plt.plot(x[:,0],x[:,1],'o')
        # plt.savefig('delaunay_triangulation.pdf')

        # avg length of all unique edges
        triangles = tri.simplices
        # get all the unique edges 
        all_edges = set([tuple(sorted(edge)) for item in triangles for edge in combinations(item,2)])
        # compute and return the average dist 
        return np.mean([np.linalg.norm(points[edge[0]]-points[edge[1]]) for edge in all_edges])

    @property
    def N(self): 
        _ = self.sidx
        return self.spatial_sampling

    @property
    def prms2(self):
        # mean p_rms^2 [Pa^2]
        return np.mean(np.abs(self.fp)**2)

    @property
    def sprms2(self):
        # mean p_rms^2 at sampling points [Pa^2]
        return np.mean(np.abs(self.fp[self.sidx])**2)

    @property
    def wavelength(self):
        return c0/self.frequency

    @property
    def extent(self):
        """return (xmin,xmax,ymin,ymax)"""
        dims =[[np.min(self.r[:,1]), np.max(self.r[:,1])],
               [np.min(self.r[:,0]), np.max(self.r[:,0])],
               [np.min(self.r[:,2]), np.max(self.r[:,2])]]
        aperture = np.array(dims).ravel()[:4]
        dx_2     = np.diff(aperture[:2])/(self.shp[0]-1)/2
        dy_2     = np.diff(aperture[2:])/(self.shp[1]-1)/2
        return tuple((aperture + np.array([-dx_2, dx_2, -dy_2, dy_2]).T)[0])


    def _cat_patches(self, x):
        # padding only applied for single aperture case!
        def ifpad(x):
            if self.pad_method:
                if 'zero' in self.pad_method:
                    mode = "constant"
                elif 'reflect' in self.pad_method:
                    mode = "reflect"
                x = np.pad(x, self.psize[0]-1, mode = mode)
            return x

        if np.ndim(x) == 1: 
            # single aperture, single frequency
            x = np.reshape(x, self.shp)
            self.pidx; # init patching
            x = ifpad(x)
            return x.ravel()[self.pidx]
        elif np.ndim(x) == 2: 
            xs = x.shape[1:]
            x = ifpad(x.reshape(self.shp)).reshape((-1,*xs))
            if x.shape[1] == len(self.f):
                # single aperture, multiple frequencies
                x_patched = np.array([pp[self.pidx] for pp in x.T])
                x_patched = np.moveaxis(x_patched,0,-1)
                return x_patched
            else: 
                # multiple apertures, single frequency
                x_patched =  np.array([pp[self.pidx] for pp in x])
                return x_patched.reshape((-1, *self.psize))
        else: 
            #multiple apertures, multiple frequencies
            x_patched = np.array([pp[:,self.pidx] for pp in x.T])
            x_patched = np.moveaxis(x_patched,1,-1)
            return x_patched.reshape((-1, *self.psize, len(self.f)))

    @property
    def noise(self):
        if not hasattr(self,'_noise'):
            if self.snr<100:
                plevel = 20*np.log10(np.std(self.fp))
                amplitude = 10**((plevel-self.snr)/20)
                N = np.prod(self.shp)
                noise = self._rng.standard_normal(N)
                phase = self._rng.uniform(0,np.pi,N)
                phase = np.cos(phase)+1j*np.sin(phase)
                self._noise=(noise*amplitude*phase)[:,np.newaxis]
                nlevel = 20*np.log10(np.std(self._noise))
            else:
                self._noise = self.fp*0
        return self._noise

    @property
    def fsp(self): # sampled versions ONLY take the first aperture
        return self.fmask[:,np.newaxis] * (self.fp + self.noise)

    @property
    def fp(self): # bypass self.fp to include all apertures (for learning)
        if hasattr(self,'pm'):
            pm = self.pm
        elif hasattr(self,'_pm'):
            pm = self._pm
        if np.ndim(pm) <= 2:
            return pm
        else:
            return pm[:,:, self.aperture_idx]

    @property
    def sp(self):
        # return self.fsp.reshape(self.shp + (self.fp.shape[-1],)) #MULTIFREQ
        return self.mask * (self.p + self.noise.reshape(self.p.shape))

    @property
    def patches(self):
        if hasattr(self,"pm"):
            fp = self.pm
        else:
            fp = self._pm
        return self._cat_patches(fp).squeeze()

    @property
    def u(self):
        if hasattr(self,"um"):
            fu = self.um
            return fu.reshape(-1,*self.shp)
        elif hasattr(self,"_um"):
            fu = self._um
            return fu.reshape(-1,*self.shp)
        else:
            return np.zeros((3,*self.shp))

    @property
    def IJ(self): # active + reactive intensity
        return self.p[np.newaxis,...]*self.u.conj()/2

    @property
    def spatches(self):
        return self._cat_patches(self.fsp)

    @property
    def fspatchesall(self):
        fmask = np.zeros(self._pm.shape, dtype=self.fp.dtype)
        fmask[self.sidx,:] = 1
        pm = self._pm * fmask
        spatchesall = self._cat_patches(pm)
        fspatchesall = spatchesall.reshape((-1, np.prod(self.psize), len(self.f)))
        return fspatchesall

    @property
    def fpatches(self):
        return self.patches.reshape((-1, np.prod(self.psize), len(self.f)))

    @property
    def fspatches(self):
        return self.spatches.reshape((-1, np.prod(self.psize), len(self.f)))

    @property
    def fspatches_nz(self):
        # returns list of tuples with (measurements, patch_indices) for each patch
        return [(fspatch.compress(fspatch!=0)[:,np.newaxis], np.where(fspatch!=0)[0] )for fspatch in self.fspatches.squeeze()]

    @property
    def p(self):       
        fpshape = self.fp.shape
        if np.prod(self.shp) == fpshape[-1]:
            p = self.fp.reshape(self.shp)
        elif np.prod(self.shp) == fpshape[0]:
            p = self.fp.reshape(tuple(self.shp) + (self.fp.shape[-1],)) # MUTLIFREQ
        return p.squeeze()

    @property
    def padlen(self):
        return self.psize[0]-1 if self.pad_method is not None else 0

    @property
    def paddedshape(self):
        return (self.shp[0]+2*self.padlen, self.shp[1]+2*self.padlen)

    @property
    def pidx(self): # returns patches
        if not hasattr(self,"_pidx"):
            x = self.p
            if self.pad_method is not None:
                x = np.pad(x, self.padlen, mode = "constant")
            _, self._pidx = extract_patches_2d(x, self.psize)
        return self._pidx

    def multifreq_patches(self, target_freq = None):
        if target_freq is None:
            target_freq = self.frequency
        target_psize = self._gen_psize(freq = target_freq)

        mp = list()
        for ii,ff in enumerate(self.f): #loop through frequencies
            ffpsize = self._gen_psize(freq = ff)
            # TODO: Scale before patch extraction for fewer interpolation artifacts
            fp = np.vstack([ 
                extract_patches_2d(self._pm[:,ii,nn].reshape(self.shp), ffpsize)[0] 
                for nn in range(self.nof_apertures)])
            fp = fp.reshape((fp.shape[0],-1))
            fp = scale_patches(fp, self.dx, ff, self.dx, target_freq,
                    self.patch_size_in_lambda,
                    # mode='direct', # not for DL paper
                    )
            mp.append(fp) # scale patches and append

        # fp2 = scale_patches(fp, self.dx, ff, self.dx, target_freq, self.patch_size_in_lambda, mode='direct')
        # plt.figure(4321)
        # plt.subplot(211)
        # plt.imshow(np.abs(fp[0].reshape(target_psize)))
        # plt.subplot(212)
        # plt.imshow(np.abs(fp2[0].reshape(target_psize)))
        # plt.show()

        mp = np.vstack(mp)
        return mp, target_psize

    @property
    def nof_measurements(self):
        return len(self.sidx)

    @property
    def sidx(self):
        if (self.spatial_sampling != 0.0):
            if not hasattr(self,"sample_idx"):
                # print(" > sampling aperture", end = " ")
                from time import time
                tic = time()
                # convert density to absolute number of mics
                if (self.spatial_sampling < 0.0):
                    self.spatial_sampling = int(np.round( 
                        calc_number_of_mics( -self.spatial_sampling, self.f,
                            self.areaxy)))

                # see if min_distance can be kept
                if   self.spatial_sampling / np.prod(self.shp) > .4:
                    self.min_distance = 1e-5
                elif self.spatial_sampling / np.prod(self.shp) > .09:
                    self.min_distance = self.dx + 1e-5
                if not hasattr(self,'_seed'):
                    self._seed = None
                self.sample_idx = sample_surface(self.shp,
                        self.spatial_sampling,
                        self.loss_mode,
                        pos = self.r,
                        min_distance = self.min_distance,
                        seed = self._seed)
                self.spatial_sampling = len(self.sample_idx)
                # print("{:_>26}".format(" in {:.2f}".format(time()-tic)))
            return self.sample_idx
        else:
            return np.arange(self.fp.shape[0])

    @property
    def frequency(self):
        if hasattr(self.f, '__iter__'):
            return self.f[0]
        else:
            return self.f

    @property
    def k(self):
        return 2*np.pi*self.frequency/c0

    @property
    def r(self):
        if hasattr(self,'rm'):
            rm = self.rm
        else:
            rm = self._rm

        if np.ndim(rm) <= 2:
            return rm
        else:
            return rm[:,:,0]

    @property
    def t(self):     
        if hasattr(self,"tm"):
            return self.tm
        elif hasattr(self,"_tm"):
            return self._tm
        else:
            print("time not available")


def scale_patches(patches, dx_current, f_current, dx_target, f_target,
        patch_size_in_lambda, mode='spline'):
    # args: soundfield, patches (nof_features x nof_patches), f_current, f_target
    # TODO: use direct interpolation, as in SoundfieldReconstruction
    side_current = int(np.sqrt(patches.shape[1]))
    wl_current = c0/f_current
    current    = np.arange(float(side_current))
    current   *= dx_current/wl_current # scale in lambda
    wl_target  = c0/f_target
    target     = np.arange(np.ceil(wl_target/dx_target * patch_size_in_lambda)+1)
    target    *= dx_target/wl_target

    if mode == 'direct': # not tested
        scale = 1/np.max(np.diff(current))
        tmp = np.zeros((len(current),len(target)),dtype=patches.dtype)
        pr = np.zeros((patches.shape[0],len(target),len(target)),dtype=patches.dtype)
        for kk,pm in enumerate(patches.reshape((-1,side_current,side_current))):
            for ii,xi in enumerate(current):
                for jj,yj in enumerate(target):
                    tmp[ii,jj] = np.inner(pm[ii,:],np.sinc((yj-current)*scale))
            for jj,yj in enumerate(target):
                for ii,xi in enumerate(target):
                    pr[kk,ii,jj] = np.inner(tmp[:,jj],np.sinc((xi-current)*scale))
        # plt.figure(234)
        # plt.subplot(311)
        # plt.imshow(np.abs(pm))
        # plt.subplot(312)
        # plt.imshow(np.abs(tmp))
        # plt.subplot(313)
        # plt.imshow(np.abs(pr[kk,...]))
        # print(current)
        # print(target)
        return pr.reshape((patches.shape[0],-1))
    else:
        from scipy import interpolate

        nof_patches = patches.shape[0]
        p_scaled = np.empty((nof_patches, int(len(target)**2) ), dtype=patches.dtype)
        for ii in range(nof_patches):

            fr = interpolate.interp2d( current, current, np.real(patches[ii]), kind='quintic')
            p_scaled[ii] = fr(target,target).ravel('F')

            if patches.dtype == complex:
                fi = interpolate.interp2d( current, current, np.imag(patches[ii]), kind='quintic')
                p_scaled[ii] += 1j*fi(target,target).ravel('F')
        return p_scaled


def plane_wave_expand(sfobj, temperature = None):
    from src._inverse_problem import solve_inv_problem
    from src._sphere_sampling import fibonacci_sphere
    k_unit_sphere = fibonacci_sphere(samples = 1000, randomize = False)

    # position dependent temperature demo
    if np.any(temperature):
        c0_temperature_variation = ((sfobj.t + 273)/(sfobj.t[0]+273))**(1/2)
        c0_temperature_variation = np.expand_dims(c0_temperature_variation, 2) @ np.ones((1,3))
    else:
        c0_temperature_variation = 1

    H  = np.exp((1j*sfobj.r * 2*np.pi*sfobj.frequency/ (c0*c0_temperature_variation) ) @ k_unit_sphere.T)
    H /= np.sqrt(len(sfobj.r)) # unit scale
    # gamma , pfield = solve_inv_problem( 'global_lasso', H, sfobj.fp, {'tolerance': 0.1,})
    gamma , pfield = solve_inv_problem( 'global_ridge', H, sfobj.fp, {'tolerance': 0.1,})
    return np.reshape(pfield , sfobj.shp), gamma


def plane_wave_expand_local(sfobj):
    from src._inverse_problem import solve_inv_problem
    from src._sphere_sampling import fibonacci_sphere
    k_unit_sphere = fibonacci_sphere(samples = 100, randomize = False)

    # plane wave expansion local
    ri    = sfobj.r[sfobj.pidx[0].ravel(),:]
    H     = np.exp(1j*ri @ k_unit_sphere.T)\
            /np.sqrt(len(ri))

    gamma , Y = solve_inv_problem(
            'local_ridge_xval', 
            H, 
            sfobj.fpatches.T,
            {   'reg_lambda'          : 0,
                'reg_tol_c'           : 1e-1,
                'n_iter'              : 100,
                'reg_tol_sig2'        : 1e0,
            })
    Y        = Y.T.reshape(-1, *sfobj.psize)
    pfield,_ = reconstruct_from_patches_2d( Y, sfobj.shp[:2], return_var = False)

    if self.pad_method in ['zero','reflect']:
        self.pn = self.psize[0]-1
        pfield = pfield[pn:-pn,pn:-pn]

    return pfield, gamma


def plane_wave_expand_csc(sfobj):
    from src._inverse_problem import solve_inv_problem
    from src._sphere_sampling import fibonacci_sphere
    k_unit_sphere = fibonacci_sphere(samples = 100, randomize = False)

    # plane wave expansion local
    ri    = sfobj.r[sfobj.pidx[0].ravel(),:]
    H     = np.exp(1j*ri @ k_unit_sphere.T)/np.sqrt(len(ri))

    # sporco
    from sporco import util, plot
    p = sfobj.p
    H = H.reshape((*sfobj.psize,-1))

    from sporco.fista import cbpdn
    lmbda = 5e-2
    L = 5e1
    opt = cbpdn.ConvBPDN.Options({
        'Verbose': True, 
        'MaxMainIter': 250,
        'RelStopTol': 2e-4, 
        'L': L, 
        'BackTrack': {'Enabled': True },
        # 'DataType' : complex128,
        })
    problem = cbpdn.ConvBPDN(H, p, lmbda, opt, dimK=0)
    gamma = problem.solve()
    print("ConvBPDN solve time: %.2fs" % problem.timer.elapsed('solve'))
    pfield = problem.reconstruct().squeeze()

    return pfield, problem


def default_soundfield_config(measurement, **kwargs):
    config = dict({
            'frequency'   : c0, # normalize wavelength to 1 unit = 1[m]
            'min_distance': 0.07,
            'measurement' : measurement,
            'pad_method'  : None,
            'patch_size_in_lambda' : 1.0,
            })
    [config.update({key:value}) for key,value in kwargs.items() if key in config.keys()];
    if measurement == 'sim':
        config.update({
            'dx'   : .05,
            'rdim' : [[0,4.41],[0,3.31],[0,2.97]],
            })
    elif 'monoplane' in measurement:
        config.update({
            'dx'   : 1/10,
            'rdim' : [[-1, 4],[-2, 3],[1/10,1/10]],
            'min_distance' : 2/10+.001,
            'seed' : 192384734, # field generation and sampling demo case
            })
    elif 'xyplane' in measurement:
        s2 = np.sqrt(2) # make sure boundaries wrap (wavenumber fixed)
        config.update({
            'dx'   : s2/10,
            'rdim' : [[-1*s2, 4*s2],[-1*s2, 4*s2],[0,0]],
            })
    elif 'plane' in measurement:
        config.update({
            'dx'   : 1/10,
            'rdim' : [[-1, 4],[-1, 4],[0,0]],
            })
    elif 'mono' in measurement:
        monodist =  1/8
        config.update({
            'dx'   : 1/10,
            # 'rdim' : [[-1, 1],[-1, 1],[monodist,monodist]],
            # 'rdim' : [[-.5, 1.5],[-.5,  1.5],[monodist,monodist]],
            # 'rdim' : [[0, 2],[0, 2],[monodist,monodist]],
            # 'dx'   : 1/6,
            # 'rdim' : [[-3, 3],[-3, 3],[monodist, monodist]],
            'rdim' : [[-2, 2],[-2, 2],[monodist, monodist]],
            })
    elif 'synthesized_diffuse' in measurement:
        config.update({
            'dx'   : 1/10,
            'rdim' : [[-3, 3],[-3, 3],[0,0]],
            'nof_apertures' : 1,
            })
        # print('approx 011 training data setup at', config['frequency'])
        # config.update({
            # 'dx'   : .05,  # for DL on synthetic data, copy 011 measurement layout
            # 'rdim' : [[0, .81],[0, .81],[0,0]],
            # 'nof_apertures' : 7,
            # })
    elif 'scatter_sphere' in measurement:
            config.update({'scatter_sphere_radius': 1.0,
                'frequency' : c0,
                'dx'   : r/10,
                'rdim' : [[-2, 2],[-2, 2],[0,0]],
                })
    elif ('011' in measurement) or ('lab' in measurement):
        config.update({
            'frequency' : 600,
            'dx'   : .05,
            'rdim' : [[0,4.41],[0,3.31],[0,2.97]],
            'nof_apertures' : 7,
            })
    elif ('019' in measurement) or ('classroom' in measurement):
        config.update({
            'frequency' : 600,
            'dx'   : .025,
            'rdim' : [[0,9.45],[0,6.63],[0,2.97]],
            })

    # update with kwargs if they match existing keys
    # [config.update({key:value}) for key,value in kwargs.items() if key in config.keys()];
    config.update(kwargs)
    return config


if __name__ == '__main__':
    b_csc            = False
    b_loc            = True
    b_scale_wavelength = True

    # field_opts = default_soundfield_config('monoplane')
    # field_opts = default_soundfield_config('synthesized_diffuse')
    field_opts = default_soundfield_config('classroom')
    # field_opts = default_soundfield_config('011')
    # field_opts.update({
        # 'frequency'   : np.arange(590,610+1e-4,1.),
        # })

    # field_opts       = dict({
        # "measurement" :'019_lecture_room',
        # "dx"          : .025,
        # "rdim"        : [[0,9.45],[0,6.63],[0,2.97]], # room dimensions
        # # "frequency"   : np.array([1000]),
        # "spatial_sampling" : -4.0, # specific cases
        # # "spatial_sampling" : -12.0,
        # "frequency"   : np.array([1000]),
        # # "spatial_sampling" : -6.0, # find nyquist for specific frequencies
        # })

    # field_opts = dict({
        # # "measurement" : '019_lecture_room', # None, #'room011', #
        # # "measurement" : 'sim',
        # # "measurement" : 'plane',
        # "measurement" : 'mono',
        # "dx"          : .025,
        # "rdim"        : [[-1,1],[-1,1],[-1,1]], # room dimensions
        # # "measurement" : 'room011_single',
        # "frequency"   : np.array([889]),
        # })

    sfo = Soundfield(**field_opts)

    ## PLOTTING
    def make_subplot(ii_ax, value, title, **kwargs):
        from src._plot_tools import add_cbar
        ax = fig.add_subplot(gs[ii_ax])
        im = plt.imshow(
            value,
            origin = 'lower', **kwargs)
        if 'extent' in kwargs:
            if b_scale_wavelength:
                ax.set_ylabel('x/y [$\lambda$]')
            elif hasattr(kwargs,'extent'):
                ax.set_ylabel('x/y [m]')
        ax.set_title("{}) ".format(chr(ii_ax+97)) + title, 
               loc='left', usetex = False)
        ax.grid()
        cbar = add_cbar(im,ax)

    fig = plt.figure(1, figsize=(10,7))
    plt.clf()
    ncol = 4
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2+b_csc+b_loc,ncol , figure=fig, 
            hspace = .7, wspace = .5, 
            bottom = .06, top = .95)

    ed = np.max(sfo.r, axis=0) -np.min(sfo.r, axis=0)
    if b_scale_wavelength:
        ed = ed/sfo.wavelength
    extent = tuple(1/2*np.array([-ed[0] ,ed[0] ,-ed[1] ,ed[1] ]))

    def level(x):
        return 20*np.log10(np.abs(x))

    def nmse(ref,x):
        return 10*np.log10(np.mean((np.abs(x-ref)/np.abs(ref))**2))

    def normalize_abs(x):
        return x/np.max(np.abs(x.ravel()))

    # show data
    rowii = 0
    vmin = np.min(level(sfo.p))
    vmax = np.max(level(sfo.p))
    make_subplot(rowii + 0, level(sfo.p), "$p$ analytical", 
            extent = extent, vmin = vmin, vmax = vmax)

    if hasattr(sfo, 't'):
        sfo.pw_p, gamma_pwe = plane_wave_expand(sfo, sfo.t)
    else:
        sfo.pw_p, gamma_pwe = plane_wave_expand(sfo, None)

    rowii += ncol
    make_subplot(rowii + 0, level(sfo.pw_p),"$p$ pwe_global", 
            vmin = vmin, vmax = vmax, extent = extent)
    make_subplot(rowii + 1, level(sfo.p-sfo.pw_p)
            -level(sfo.p),
            "$\epsilon$ pwe_global {:.2f}".format(
                nmse(sfo.p,sfo.pw_p)),
            cmap = plt.cm.binary, extent = extent)

    ax = fig.add_subplot(gs[rowii+2])
    ax.hist(np.abs(gamma_pwe),color='C0', bins = 50, log=False)
    ax.set_title("{}) ".format(chr(rowii+2+97)) +
            "$|\gamma|$ pwe_global", loc='left', usetex=False)


    if b_loc:
        sfo.pw_loc, gamma_loc = plane_wave_expand_local(sfo)

        rowii += ncol
        make_subplot(rowii + 0, level(sfo.pw_loc),"$p$ loc_indep", 
                vmin = vmin, vmax = vmax, extent = extent)
        make_subplot(rowii + 1, level(sfo.p-sfo.pw_loc)
                -level(sfo.p),
                "$\epsilon$ loc_indep {:.2f}".format(
                    nmse(sfo.p,sfo.pw_loc)),
                cmap = plt.cm.binary, extent = extent)

        ax = fig.add_subplot(gs[rowii+2])
        ax.hist(np.abs(gamma_loc).ravel(), color='C1', bins = 50, log=False)
        ax.set_title("{}) ".format(chr(rowii+2+97)) +
                "$|\gamma|$ local_indep", 
                loc='left', usetex=False)

        coefmap = np.sum(abs(gamma_loc), 0)
        coefmap = coefmap.reshape(tuple([a-b+1 for a,b in
            zip(sfo.shp, sfo.psize)]))
        make_subplot(rowii + 3, coefmap, 
                title='$|\gamma_{xy}|$ local_indep', cmap=plt.cm.Blues)

    if b_csc:
        sfo.pw_csc, cc = plane_wave_expand_csc(sfo)
        gamma_cc = cc.getcoef()

        rowii += ncol
        make_subplot(rowii + 0, level(sfo.pw_csc),"$p$ loc_csc", 
                vmin = vmin, vmax = vmax, extent = extent)
        make_subplot(rowii + 1, level(sfo.p-sfo.pw_csc)
                -level(sfo.p),"$\epsilon$ loc_csc {:.2f}".format(
                    nmse(sfo.p,sfo.pw_csc)),
                cmap = plt.cm.binary, extent = extent)

        ax = fig.add_subplot(gs[rowii+2])
        ax.hist(np.abs(gamma_cc).ravel(),color='C2', bins = 50, log=False)
        ax.set_title("{}) ".format(chr(rowii+2+97)) +
                "$|\gamma|$ loc_csc", 
                loc='left', usetex=False)
        make_subplot(rowii + 3, np.sum(abs(gamma_cc), axis=cc.cri.axisM).squeeze(), 
            title='$|\gamma_{xy}|$ loc_csc', cmap=plt.cm.Blues)


    plt.draw()
    plt.show()
