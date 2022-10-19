import numpy as np
import time
import warnings
from enum import Enum, auto
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from copy import deepcopy

from Soundfield import Soundfield, default_soundfield_config, reconstruct_from_patches_2d
from src._const import c0
from src._sphere_sampling import fibonacci_sphere

from src._tools import calc_coherence
from src._inverse_problem import solve_inv_problem
from src._plot_tools import plot_dictionary

from src._sfr_utils_hdf import *

#######
# TODO: update() function depending on set parameters
# vision: decomposition is only basis functions,
# transformation is model. global / pb / csc 
# regularization is specific global: lasso/ridge, pb: group lasso, csc:
# L2L1Grad, ...
# fitting to learn field not required, but errors arise if LF is None

class Decomposition(Enum):
    cpwe = 0
    lpwe = 1
    gpwe = 2

DECOMPOSITION      = ['gpwe', 'lpwe', 'cpwe']
TRAINING           = ['011', '019']
EVALUATION         = ['011', '019', 'monoplane']
REC_FREQ           = [500, 600, 700, 800, 900, 1000, 1250, 1600, 2000]
SPATIAL_SAMPLING   = [ -4., 40, 80, 160, 320,-12., 640,1280,0.0, ]

REC_FREQ = [
        500,  510,  520,  530,  540,  550,  560,  570,  580,  590,  600,
        610,  620,  630,  640,  650,  660,  670,  680,  690,  700,  710,
        720,  730,  740,  750,  760,  770,  780,  790,  800,  810,  820,
        830,  840,  850,  860,  870,  880,  890,  900,  910,  920,  930,
        940,  950,  960,  970,  980,  990, 1000, 1010, 1020, 1030, 1040,
       1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150,
       1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230, 1240, 1250, 1260,
       1270, 1280, 1290, 1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370,
       1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480,
       1490, 1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590,
       1600, 1610, 1620, 1630, 1640, 1650, 1660, 1670, 1680, 1690, 1700,
       1710, 1720, 1730, 1740, 1750, 1760, 1770, 1780, 1790, 1800, 1810,
       1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920,
       1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000,
    ]

# float for spatial_sampling,
# int for N of mics, neg float for density per circle of wavelength diameter

# REG_LAMBDA         = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# REG_MU             = [1e-1, 1e-2, 1e-3, 1e-4]
# REG_TOLERANCE      = [0, 100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

ALLPARAMETERS      = [
        DECOMPOSITION,    TRAINING,
        EVALUATION,       REC_FREQ,
        SPATIAL_SAMPLING,
        # REG_LAMBDA, REG_MU, REG_TOLERANCE,
        # NOF_COMPONENTS, ALPHA, BATCH_SIZE,
        ]

class Decomposition_clear(Enum):
    cpwe = "Convolutional"# plane waves"
    lpwe = "Local independent"# plane waves"
    gpwe = "Global"# "PWE"#"global plane waves"

def decompositions(): return Decomposition
def decompositions_clear(): return Decomposition_clear
def find_decomposition_clear(dec):
    if type(dec) == Decomposition:
        return Decomposition_clear[dec.name].value
    elif type(dec) == str:
        return Decomposition_clear[dec].value

def spatial_coherence(x,y):
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore", RuntimeWarning)
    coh =  calc_coherence(x, y)
    return coh

def assess_sparsity(gamma, num_patches=None, tol=.001, nof_components = False,
        return_std=False):
    nnonzeros = np.mean(np.abs(gamma)>0)*gamma.shape[0]
    if return_std:
        return nnonzeros, np.std(np.abs(gamma)>0)*gamma.shape[0]
    else:
        return nnonzeros
    
    # # approximation_norm = np.linalg.norm(gamma,2,axis=0)
    # approximation_norm = np.linalg.norm(gamma)
    # if gamma.squeeze().ndim == 1:
        # return np.sum(np.abs(gamma) > tol*approximation_norm)
    # elif gamma.ndim == 2: 
        # nonzeros = (np.abs(gamma)> tol*approximation_norm)
        # # nonzeros = (np.abs(gamma)> tol*approximation_norm[np.newaxis,:])
        # return np.mean(np.sum(nonzeros,axis=0))

def mse(x,reference):
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore", RuntimeWarning)
    x = x.ravel()
    y = reference.ravel()
    # averaged magnitude and phase errors, standard
    return 20 * np.log10(np.linalg.norm(x-y))

def nmse(x,reference):
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore", RuntimeWarning)
    x = x.ravel()
    y = reference.ravel()
    # pointwise comparison of magnitude and phase errors 
    # return 10 * np.log10(np.mean((np.abs(x - y) / np.abs(y)) ** 2)) 
    # averaged magnitude and phase errors, standard
    return mse(x,y)-20 * np.log10(np.linalg.norm(y))

def checkdelattr(obj, attribute): 
    """remove attribute if exist, checkdelattr(obj,attr)"""
    if hasattr(obj, attribute):
        delattr(obj, attribute)

def checkmoveattr(obj, attribute, newattr): 
    """if exists, move attribute to , checkmoveattr(obj,attr,newattr)"""
    if hasattr(obj, attribute):
        obj[newattr] = obj[attribute]
        delattr(obj, attribute)

class SoundfieldReconstruction():
    '''
    todo:
    - move functions in
    - make functions lazy, only link to data at the highest layer
    '''
    def __init__(self,
            measured_field = None,
            training_field = None,
            **kwargs
            ):
        ''' TODO: just unpack all args and check if exist'''
        self.measured_field = measured_field
        self.training_field = training_field
        self.reconstructed_field  = None # no reconstruction available yet
        self.__dict__.update(kwargs)

    def title(self):
        if self.mf.spatial_sampling == 0.0: # integer number of mics
            samplingstr =  "{:0>5.0f}".format(np.prod(self.rf.shp))
        elif self.mf.spatial_sampling%1: # integer number of mics
            samplingstr =  "{:_<5.3f}".format(
                self.mf.spatial_sampling).replace('.','p')
        elif (self.mf.spatial_sampling >= 0.0): # lossfactor 0..1
            samplingstr =  "{:0>5.0f}".format(self.mf.spatial_sampling)

        if hasattr(self, 'rec_timestamp'):
            rt = '{:0>5d}'.format(int(self.rec_timestamp%1e5))
        else:
            rt = ''

        return '_'.join([
            "{:_<5}".format(self.decomposition),
            # self.mf.measurement,
            '{:0>4.0f}'.format(self.mf.frequency),
            samplingstr,
            # self.training_field.measurement,
            str(int(self.fit_timestamp%1e8)),
            rt,
            ])

    def _gen_plane_wave_expansion(self, soundfield, 
            particle_velocity = False, 
            force_global = False):
        k_abs = 2*np.pi*soundfield.frequency/c0
        self._ksphere = fibonacci_sphere(samples = self.nof_components)
        self._ksphere[:,2]*=np.sign(self._ksphere[:,2]) # kz pos for planar field
        self.fit_timestamp = int(time.time())
        if not self.decomposition in ['lpwe', 'cpwe'] or force_global:
            funcsize = soundfield.paddedshape
        else:
            funcsize = soundfield.psize

        rx = np.arange(funcsize[0])
        ry = np.arange(funcsize[1])
        rr = np.dstack(np.meshgrid(rx,ry,0))
        rr = rr.reshape((-1,3)) *soundfield.dx
        A     = np.exp(k_abs * 1j*rr @ self._ksphere.T)/np.sqrt(len(rr))

        if particle_velocity:
            A = A[...,np.newaxis]*self._ksphere[np.newaxis,...]/(1.2*c0)
        return A


    def fit(self, field = None):
        if field:
            self.training_field = field
        # TODO: update flag when settings change, like val.frequency
        tic = time.time()

        # prepend _ to avoid storage in hdf
        if self.tf is None:
            self.training_field = Soundfield('mono') #random config
        self._A_learn = self._gen_plane_wave_expansion(self.tf)

        self.fit_timestamp = int(time.time())


    def _plot_dictionary(self, figure = 1, show=False, 
            nof_samples = None, 
            title = None,
            order = None, #['abs','real','phase','imag'],
            grid = None,
            dictionary = None):
        if order is None:
            if np.all(np.imag(self.A) == 0): order = ['real']
            else: order = ['real','imag']
        if dictionary is None:
            dictionary = 'learn'
        if not grid: grid = (len(order),1)
        if not nof_samples: nof_samples = self.nof_components
        if not title: title = Decomposition_clear[self.decomposition].value

        D = self.A
        field = self.training_field
        overlay = None
        if dictionary == 'rec': 
            D = self.A_rec
            field = self.measured_field

        plot_dictionary(D,
                nof_samples, 
                field.psize, 
                figure,
                resolution = field.dx/c0*field.frequency,
                # title = title,
                title = " ".join([title, dictionary,]),
                grid = grid,
                order = order,
                overlay = overlay,
                )

        if show:
            plt.draw()
            plt.show()

    def _gen_A_rec(self):
        '''from the fitted A, generate a suitable reconstruction transfer matrix '''
        self._A_rec = self._gen_plane_wave_expansion(self.measured_field)
        self._Au_rec = self._A_rec[np.newaxis,...]*self._ksphere.T[:,np.newaxis,:]/(1.2*c0)

    @property
    def A(self):
        if (not hasattr(self,'A_learn')) and (not hasattr(self,'_A_learn')):
            self.fit()
        if hasattr(self,'A_learn'):
            return self.A_learn
        else:
            return self._A_learn

    @property
    def Ar(self):
        if (not hasattr(self,'A_rec')) and (not hasattr(self,'_A_rec')):
            self._gen_A_rec()
        if hasattr(self,'A_rec'):
            return self.A_rec
        else:
            return self._A_rec

    @property
    def Aur(self):
        if (not hasattr(self,'A_rec')) and (not hasattr(self,'_A_rec')):
            self._gen_A_rec()
        if hasattr(self,'_Au_rec'):
            return self._Au_rec

    @property
    def coeffs(self):
        if (not hasattr(self,'coefficients')) and (not hasattr(self,'_coefficients')):
            Warning("No coefficients available, call reconstruct() first, return 0")
            return 0
        if hasattr(self,'coefficients'):
            return self.coefficients
        else:
            return self._coefficients

    @property
    def rf(self):
        return self.reconstructed_field

    @property
    def tf(self):
        return self.training_field

    @property
    def mf(self):
        return self.measured_field

    def reconstruct(self, measured_field = None, local_var=False):
        # TODO: make wrapper that adjusts for apertures and frequencies...
        # reconstructing one aperture / frequency at a time and accumulate
        # results before evaluating.
        # keep reconstruction in internal _reconstruct_aperture()
        if measured_field != None: #update field if provided
            self.measured_field = deepcopy(measured_field)
        mf, tf = self.measured_field, self.training_field
        print(''' > Reconstruction using {} + {} '''.format(
            self.decomposition, self.transform))
        tic = time.time()

        _ = self.Ar # needs to be there before copying and re-created in Monte Carlo
        _ = mf.sidx # sample to maintain the copying

        # All padding should move to soundfield.py, 
        # also reconstruction from patches.
        # if (self.decomposition in ['lpwe'] or 'csc' in self.transform) & (mf.b_single_patch is False):
        if ('csc' in self.transform) & (mf.b_single_patch is False):
            mf.pad_method = 'zero'
            checkdelattr(mf,'_pidx')
            _=mf.pidx

        pn = mf.padlen
        self.pad = lambda x: np.pad(x, pn, mode='constant') if pn else x
        self.crop = lambda x: x[...,pn:-pn, pn:-pn] if pn else x

        self.reconstructed_field = deepcopy(self.measured_field) # copy to reconstruction field
        rf = self.reconstructed_field

        if  self.decomposition in ['lpwe',]: # local
            if '011' in mf.measurement:
                rf.pm = np.zeros(mf.pm.shape, dtype = complex)
                for ii in range(mf._pm.shape[-1]):
                    mf.aperture_idx = ii
                    checkdelattr(mf,'sample_idx')
                    print(mf.fspatches.shape)
                    gamma, Y = solve_inv_problem( self.transform, self.Ar, 
                            mf.fspatches[:,:,0].T, self.transform_opts)
                    Y  = Y.T.reshape(-1, *rf.psize)
                    y, y_var = reconstruct_from_patches_2d(Y, rf.paddedshape, return_var = local_var)
                    prec = self.crop(y).ravel() # patches are padded in Soundfield.py
                    rf.pm[:,0,ii] = prec
            else: # 019 classroom, dl paper
                gamma, Y = solve_inv_problem( self.transform, self.Ar, 
                        mf.fspatches[:,:,0].T, self.transform_opts)
                Y        = Y.T.reshape(-1, *rf.psize)
                y, y_var = reconstruct_from_patches_2d(Y, rf.paddedshape, return_var = local_var)
                rf.pm = self.crop(y).ravel()[:,np.newaxis] # patches are padded
            self._Y = np.atleast_3d(Y)
            if y_var is not None:
                rf._pm_var = self.crop(y_var).ravel()

            Yu = -self.Aur.dot(gamma)[[1,0,2]]
            Yu = Yu.reshape(3,*rf.psize,-1)
            Yu = np.moveaxis(Yu,-1,1)
            rf._um = np.stack([self.crop(reconstruct_from_patches_2d(ui, rf.paddedshape)[0]) for ui in Yu])

            # uref = np.stack([mf._cat_patches(um) for um in mf._um])
            # if False: #debug
                # plt.figure(figsize=(8,8))
                # plt.subplot(221)
                # plt.imshow(uref[0,1000].real,origin='lower')
                # plt.subplot(222)
                # plt.imshow(Yu[0,1000].real.squeeze(),origin='lower')
                # plt.subplot(223)
                # plt.imshow(uref[0,1000].imag,origin='lower')
                # plt.subplot(224)
                # plt.imshow(Yu[0,1000].imag.squeeze(),origin='lower')
                # plt.savefig('pvectest.pdf')
                # plt.close()
                # print(
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(mf._cat_patches(mf._um[0]),mf.paddedshape)[0])-mf.u[0]) ,
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(mf.patches,mf.paddedshape)[0])-mf.p) ,
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(uref[0],mf.paddedshape)[0])-mf.u[0]) ,
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(uref[1],mf.paddedshape)[0])-mf.u[1]) ,
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(uref[2],mf.paddedshape)[0])-mf.u[2]) ,
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(Yu[0],mf.paddedshape)[0])-mf.u[0]) ,
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(Yu[1],mf.paddedshape)[0])-mf.u[1]) ,
                    # np.linalg.norm(self.crop(reconstruct_from_patches_2d(Yu[2],mf.paddedshape)[0])-mf.u[2]) ,
                # )
                # rf._um = np.stack([self.crop(reconstruct_from_patches_2d(ui, rf.paddedshape)[0]) for ui in uref])

        elif self.decomposition in ['cpwe']:
            from sporco.admm import cbpdn

            self.pmeas = self.pad(mf.sp)
            mask = np.real(self.pad(mf.mask))

            H = self.Ar

            H /= np.linalg.norm(H, axis=0)

            # H_hires = self.Ar
            H_hires = self._gen_plane_wave_expansion(mf, force_global=True)
            H_hires /= np.linalg.norm(H_hires, axis=0)

            H = H.reshape((*mf.psize,self.nof_components))
            H_hires = H_hires.reshape((*mf.paddedshape,1,1,self.nof_components))
            H_hires = np.swapaxes(H_hires,0,1)

            opt = { 'Verbose'       : self.transform_opts['verbose'], 
                    'FastSolve'     : self.transform_opts['fastsolve'],
                    'MaxMainIter'   : self.transform_opts['n_iter'],
                    'HighMemSolve'  : True, 
                    'RelStopTol'    : self.transform_opts['reg_tolerance'],
                    'AuxVarObj'     : False, 
                    'RelaxParam'    : 0.7,
                    'rho'           : self.transform_opts['rho'], 
                    'AutoRho'       : {'Enabled': False, 'StdResiduals': False},
                }

            if 'csc_grad' in self.transform:
                opt = cbpdn.ConvL2L1Grd.Options(opt)
                # opt.update({'GradWeight' : 'inv_dict'})
                # print(H.shape,H_hires.shape,self.pmeas.shape)
                print(opt)
                if not hasattr(self,'_cbpdn'):
                    self._cbpdn = cbpdn.ConvL2L1Grd(
                            H, self.pmeas, 
                            self.transform_opts['reg_lambda'], 
                            self.transform_opts['reg_mu'], 
                            W=mask, opt=opt, dimN=2,
                            D_hires = H_hires,
                            kvec = self._ksphere,
                            )
                _ = self._cbpdn.solve()
                gamma = self._cbpdn.X
                prec  = self._cbpdn.reconstruct().squeeze()

                # [x] opt 1: fft of transfer function u
                # [ ] opt 2: convolve u filters with coefficient map
                Hu_hires = -H_hires*self._ksphere.T/(1.2*c0)
                Huf = self._cbpdn.fftn(Hu_hires,None,self._cbpdn.cri.axisN)
                Suf = np.sum(Huf*self._cbpdn.Xf, axis=self._cbpdn.cri.axisM)
                um = self._cbpdn.ifftn(Suf, self._cbpdn.cri.Nv,self._cbpdn.cri.axisN).squeeze()
                rf._um = self.crop(np.einsum('xyu->uxy',um))

            elif sum([tt in self.transform for tt in ['csc','cbpdn']]):
                opt = cbpdn.ConvBPDNMaskDcpl.Options(opt)
                self._cbpdn = cbpdn.ConvBPDNMaskDcpl(
                        H, self.pmeas, 
                        self.transform_opts['reg_lambda'], 
                        W=mask, opt=opt, dimN=2,
                        )

            else:
                # no method specified, basis pursuit denoising fallback, no mask applied
                from sporco.admm import bpdn
                opt = { 'Verbose'       : self.transform_opts['verbose'], 
                        'FastSolve'     : self.transform_opts['fastsolve'],
                        'MaxMainIter'   : 10,
                        'RelStopTol'    : self.transform_opts['reg_tolerance'],
                        'AuxVarObj'     : False, 
                        'RelaxParam'    : 0.7,
                        'rho'           : self.transform_opts['rho'], 
                        'AutoRho'       : {'Enabled': False, 'StdResiduals': False},
                    }
                opt = bpdn.BPDN.Options(opt)
                H = H.reshape((-1,self.nof_components))
                self._cbpdn = bpdn.BPDN( H, mf.fpatches.T, self.transform_opts['reg_lambda'], opt=opt,)

            _ = self._cbpdn.solve()
            gamma = self._cbpdn.X.squeeze()
            prec  = self._cbpdn.reconstruct().squeeze()

            rf.pm = self.crop(prec).ravel()[:,np.newaxis]

            if local_var:
                patches = np.einsum('xyw,psw->xyps',gamma,H).reshape((-1,*mf.psize))
                y, y_var = reconstruct_from_patches_2d(patches, mask.shape, return_var = local_var)

                rf.pm_var= y_var


                # cb = self._cbpdn
                # print(np.all(cb.D.squeeze()==H))
                # print(np.all(cb.X.squeeze()==gamma))
                # Xf = cb.fftn(cb.X, None, cb.cri.axisN)
                # Df = cb.fftn(cb.D, cb.cri.Nv, cb.cri.axisN)
                # # Df = cb.fftn(cb.D_hires, cb.cri.Nv, cb.cri.axisN)
                # Sf = np.sum(Df * Xf, axis=cb.cri.axisM)
                # y2 = cb.ifftn(Sf,cb.cri.Nv, cb.cri.axisN).squeeze()
                # print(np.linalg.norm(prec/y2-1))

            self.prec = prec


        elif self.decomposition in ['gpwe']: # global
            if '011' in mf.measurement:
                rf.pm = np.zeros(mf._pm.shape, dtype = complex)
                rf._um = np.zeros((3,*mf._pm.shape), dtype = complex)
                for ii in range(mf._pm.shape[-1]):
                    gamma, Y = solve_inv_problem(self.transform, self.Ar, mf.fsp, self.transform_opts)
                    rf.pm[:,0,ii] = Y.ravel()
                    rf._um[:,:,0,ii] = self.Aur.dot(gamma)
            else:
                gamma, Y = solve_inv_problem(self.transform, self.Ar, mf.fsp, self.transform_opts)
                rf.pm = Y.ravel()[:,np.newaxis]
                # rf.pm = self.Ar.dot(gamma)
                rf._um = self.Aur.dot(gamma)

        else:
            raise ValueError('''No matchin reconstruction method specified for plane
            wave expansion.''')

        self.rec_time = time.time()-tic
        self._coefficients = gamma

        if hasattr(self, 'rec_timestamp'):
            if self.rec_timestamp == int(time.time()):
                time.sleep(1) # delay for unique timestamp
        self.rec_timestamp = int(time.time())
        self.id = self.title()

        # analyse 
        self.assess_reconstruction()

        # reconstructed sound field statistics
        print("\t > global sample nmse: {:.2f} dB ({:.0f} points)".format(
            self.nmse_sample_global, mf.spatial_sampling))
        # print("\t > global nmse: {:.2f} dB (full aperture)".format( self.nmse))
        # print("\t > mean p_rms^2 [true {:.2f} / meas {:.2f} / fit {:.2f} / rec {:.2f}]".format( mf.prms2, mf.sprms2, rf.sprms2, rf.prms2,))
        print("\t > coh: {:.2f}, nmse: {:.2f} dB, avg_nz: {:.2f}".format(
            self.spatial_coherence , self.nmse, self.avg_nonzeros))
        print('{:=>50}'.format(' total sec {:.2f}'.format(self.rec_time)))
        return self.reconstructed_field, self.Ar, self.coeffs, self.rec_time


    def assess_reconstruction(self):

        # spatial coherence
        self.spatial_coherence =  spatial_coherence(self.rf.fp, self.mf.fp)

        # cardinality
        if (hasattr(self,'coefficients')) or (hasattr(self,'_coefficients')):
            self.__dict__.update( { "avg_nonzeros": assess_sparsity(self.coeffs), })
        else:
            print('no coeffs, skip cardinality update')

        # coefficient norm
        self.coeffnorm = np.linalg.norm(self.coeffs)

        # global residual mse of reconstruction at measurements
        self.nmse_sample_global = nmse(self.rf.fp[self.mf.sidx], self.mf.fp[self.mf.sidx])
        # comparing all apertures - direct access to pm hardcoded
        self.nmse =  nmse(self.rf.fp, self.mf.fp)

    def minimize_hdf_storage(self, full = False):
        self.rf._pm = self.rf.pm
        [checkmoveattr(self.rf.__dict__,key,"_"+key) for key in ["pm","um","pm_var","coefficients"]]

    def clear_reconstruction(self, full = False):
        [checkdelattr(self,key) for key in
                ["id","nmse","nmse_local","nmse_sample_global",
                    "nmse_sample_local","avg_nonzeros","spatial_coherence",
                    "coefficients","_cofficients","A_rec", "_A_rec",
                    "rec_timestamp","reconstructed_field",
                    "_cbpdn"]]
        if full:
            [checkdelattr(self,key) for key in ["A_rec","_A_rec"]]


def read_sfr_from_hdf(filename = './autosave', identifiers = None):
    """
    reads from HDF and casts as objects of a reference class
    identifiers can be one identifier, or a list of identifiers for the highest
    level in the hdf file.
    if identifier is none, all objects are loaded and concatenated to a list

    checks for compatibility with old classes and updates (or even repacks) the stored data if needed.
    """
    import h5py as hdf
    from read_mc import analysis

    if (identifiers is not None) and (type(identifiers) is not list):
        identifiers = [identifiers]

    data = read_from_hdf(filename, identifiers) # if none, read all

    if (identifiers is not None) and (len(identifiers) == 1):
        # cast a single reconstruction object with sound fields
        sfr = SoundfieldReconstruction()
        sfr.__dict__ = deepcopy(data)

        mf  = Soundfield(rdim = np.array([[0,1],[0,1],[0,1]]))
        for fieldkey in ['training_field','measured_field','reconstructed_field']:
            mf.__dict__  = data[fieldkey] # copy to reconstruction field
            sfr.__dict__[fieldkey] = deepcopy(mf)

        return sfr

    else:
        # not specified or several, must be a list of fields
        sfrlist = list()
        keylist = list()
        sfr = SoundfieldReconstruction()
        mf  = Soundfield(rdim = np.array([[0,1],[0,1],[0,1]]))

        # sfrefcfg = default_soundfield_config('/work1/manha/z_data/019_lecture_room.h5')
        # sfref = Soundfield(**sfrefcfg)

        repack = 0 # if data sizes changes, repacking is required to free the disk space
        for key in data.keys():
            # try:
                update = 0
                sfr.__dict__ = data[key]

                for fieldkey in [ 'training_field','measured_field', 'reconstructed_field']:
                        try:
                            mf.__dict__  = data[key][fieldkey] # copy to reconstruction field
                            sfr.__dict__[fieldkey] = deepcopy(mf)
                        except: # legacy
                            if fieldkey == 'training_field':
                                try:
                                    mf.__dict__  = data[key]['learn_field'] # legacy
                                except:
                                    pass
                            elif fieldkey == 'measured_field':
                                mf.__dict__  = data[key]['measured_field'] # legacy
                            sfr.__dict__[fieldkey] = deepcopy(mf)
                            # update = 1

                # compatibility
                if hasattr(sfr.mf, 'loss_factor'):# legacy
                    sfr.mf.spatial_sampling = sfr.mf.loss_factor
                    del sfr.mf.loss_factor
                    update = 2
                if hasattr(sfr, 'avg_cardinality'):# legacy
                    sfr.avg_nonzeros = sfr.avg_cardinality
                    del sfr.avg_cardinality
                    update = 2
                if (key != sfr.title()):
                    with hdf.File(filename, 'r+') as f:
                        f.move(key,sfr.title())
                    key = sfr.title()
                if (sfr.id != sfr.title()):
                    analysis(sfr, b_saveplots = False, b_showplots = False) # checks again for keys
                    sfr.id = sfr.title()
                    update = 3
                elif not np.any([hasattr(sfr,key) for key in ['nmse','spatial_coherence','avg_nonzeros']]):
                    sfr.mf._gen_field()
                    sfr.assess_reconstruction()
                    update = 4
                if hasattr(sfr,'coefficients'): # remove overly large coeffss from lpwe, cpwe
                    # if np.prod(sfr.coeffs.shape) > 1e5:
                    delattr(sfr,'coefficients')
                    update, repack  = 5, 1
                if hasattr(sfr, 'mac'): # legacy
                    sfr.spatial_coherence = sfr.mac
                    del sfr.mac
                    update = 6
                # update data
                if update:
                    print("read_sfr_from_hdf(): update group for criteria nr",update)
                    write_to_hdf(sfr, key, filename)

                # sfr.mf._pm = sfref._pm
                # sfr.assess_reconstruction()

                keylist.append(key)
                sfrlist.append(deepcopy(sfr))
            # except:
                # Warning("No proper sound field reconstruction found, skip {}".format(key))
        # [write_to_hdf(sfr,key,filename) for sfr,key in zip(sfrlist,keylist)]

        if repack: 
            """repack (flag set if significant memory can be saved,
            reloads the complete file"""
            _h5repack(filename)

        return sfrlist

def default_reconstruction_config(decomposition, **kwargs):
    config = dict({
        'decomposition' : decomposition,
        })
    if 'gpwe' == decomposition:
        config.update({
            'tag'                     : 'gPWE',
            'nof_components'          : 1000, # samples of k_vectors on the k sphere
            # 'transform'               : 'global_lasso',
            # 'transform'               : 'ridge_cvx_lcurve',
            'transform'               : 'ridge_xval',
            'transform_opts'          : {
                'reg_lambda'          : .01,
                'reg_tolerance'       : 0, # orthogonal base, less noise
                'reg_tol_sig2'        : 1e0,
                },
            })
        config['transform'] = 'global_' + config['transform']
    elif 'lpwe' in decomposition:
        config.update({
            'tag'                     : 'pb-PWE',
            # 'nof_components'          : 64, # 1 lambda
            # 'transform'               : 'ridge_xval',
            'nof_components'          : 100, # 1 lambda
            'transform'               : 'lasso_lars_xval',
            'transform_opts'          : {
                'reg_lambda'          : 0.01,
                'reg_tolerance'       : 1e-3,
                'n_iter'              : 30,
                'reg_tol_sig2'        : 1e0,
                },
            })
    elif 'cpwe' in decomposition:
        config.update({
            'tag'                     : 'csc-PWE',
            'nof_components'          : 100,
            'transform'               : 'csc_grad', # w grad and conv sparsity
            # 'transform'               : 'csc', # conv sparsity
            # 'transform'               : 'bpdn', # local sparsity
            'transform_opts'          : {
                'reg_lambda'      : 1e-5,
                'reg_mu'          : 1e-3, # tuned for 019
                'rho'             : 1e-5,
                'n_iter'          : 500,
                'verbose'         : False,
                'fastsolve'       : True,
                'reg_tolerance'   : 1e-5,
                },
            })

    # update with non-default kwargs
    for key in ['reg_tol_sig2',
            'reg_tolerance'     ,
            'reg_lambda','reg_mu']:
        if key in kwargs:
            config['transform_opts'].update({key:kwargs.pop(key)})
    config.update(kwargs)

    otrans = config.pop('transform_opts')
    return config, otrans

if __name__ == '__main__' :
    from read_mc import analysis

    valopts = default_soundfield_config('classroom')
    # valopts = default_soundfield_config('monoplane')
    valopts.update({
        'frequency'        : 1000,
        # 'loss_mode' : 'grid',
        # 'spatial_sampling' : 0.0, # full sampling
        # 'spatial_sampling' : -4.0, 
        'spatial_sampling' : 100,
        'seed' : 192384734,
        })

    recopts, transopts  = default_reconstruction_config('gpwe')
    # recopts, transopts  = default_reconstruction_config('cpwe')
    # recopts, transopts  = default_reconstruction_config('lpwe')

    # load or generate learning data, load objects
    sfrec_obj  = SoundfieldReconstruction(
            measured_field = Soundfield(**valopts),
            transform_opts = transopts,
            **recopts)

    # reconstruct
    sfrec_obj.reconstruct()

    # post processing
    analysis(sfrec_obj)


