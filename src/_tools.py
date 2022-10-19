import numpy as np


def complex_interlace(A, axis=0):
    """ Interlace real and imagniary parts along specified axis
    """
    shp = list(A.shape)
    A = np.stack((np.real(A), np.imag(A)), axis)
    shp[axis] *= 2
    return A.reshape(tuple(shp), order="F")


def complex_interlaced_rows_matrix(A, axis=0):
    """ Interlace real and imaginary parts in matrix
    parameters:
        A     matrix with values complex interlaced in one dim
        axis  axis along which A is already interlaced, default 0
    """
    return complex_inerlace_2d(complex_delace(A, axis))


def complex_interlace_2d(A, axis=0):
    """ Interlace real and imaginary parts in matrix
    """
    shp = list(A.shape)
    shp[0] *= 2
    shp[1] *= 2
    Ai = np.empty(shp, dtype=A[0].dtype)

    Ai[:, ::2] = complex_interlace(A)
    Ai[:, 1::2] = complex_interlace(1j * A)

    return Ai


def complex_delace(A, axis=0):
    shp = list(A.shape)
    if A.ndim == 1:
        A = A[::2] + 1j * A[1::2]
    elif axis == 0:
        A = A[::2, :] + 1j * A[1::2, :]
    else:
        A = A[:, ::2] + 1j * A[:, 1::2]
    shp[axis] = int(shp[axis] / 2)
    return np.reshape(A, tuple(shp))


def norm_to_unit_abs(x, axis = 0):
    x = np.array(x)
    if len(x.shape) == 1:
        return x / np.max(np.abs(x.flatten()))
    else:
        if   axis == 0:
            return x / np.linalg.norm(x,axis=axis)[np.newaxis,:]
        elif axis == 1:
            return x / np.linalg.norm(x,axis=axis)[:,np.newaxis]


def calc_coherence(x, y = None, limit=None):
    """
    returns gamma_squared = |S_{xy}|^2 (S_{xx} * S_{yy})^{-1}, where S_{xy} is
    the inner product of two vectors x \cdot y^*

    if only a matrix A (M1xM2...xN) is provided, this functions returns a
    coherence between all N sub-vectors of the matrix A
    """
    error_message = """Please provide two equally sized or one single matrix\
        to calculate coherence"""
    if (y is None):
        if (x.ndim >= 1): 
            Ncolumns = x.shape[-1]
            x = norm_to_unit_abs(x.reshape((-1,Ncolumns)),axis=0)
            Coh = np.zeros((Ncolumns,Ncolumns))
            for ii in range(Ncolumns):
                for jj in range(ii,Ncolumns):
                    if limit:
                        if (np.abs(jj-ii) > limit):
                            continue
                    Coh[ii,jj] = np.abs(np.dot(x[:,ii], x[:,jj].conj()))**2
                    Coh[jj,ii] = Coh[ii,jj]
            return Coh
        else:
            raise ValueError(error_message)
    elif (x.shape == y.shape):
        return np.abs(np.dot(x.flatten(), y.flatten().conj())) ** 2 / (
            np.linalg.norm(x) ** 2 * np.linalg.norm(y) ** 2
        )
    else:
        raise ValueError(error_message)


def hann(length):
    """generate 1d hann window of given length (int)"""
    w = np.sin(np.pi * np.arange(length) / (length - 1)) ** 2
    return np.array([w])


def hann2d(dims):
    """generate 2d hann window with given dimensions (list or tuple)"""
    a = hann(dims[0])
    b = hann(dims[1])
    w = a.transpose() @ b
    return w


def apply_hann2d(signal, patch_size):
    """ apply 2d hann window on signal patches of size patch_size"""
    w = hann2d(patch_size)
    return signal * np.tile(w, (signal.shape[0], 1, 1))


def spatial_correlation(position, dictionary, accuracy = .01):
    """
    Condenses spatial correlation for dictionary atoms (column vectors)
    evaluated at position. Takes mean over all atoms and condenses
    positions to radial distance. Returns 1D distance and correlation
    vectors.
    """
    position = position - np.tile(position[0,:], (position.shape[0],1))

    from scipy.spatial import distance
    dist = distance.cdist(position,position).flatten()

    ## round ?
    rd = np.round(dist/accuracy)*accuracy

    rd_unique, rd_idx = np.unique(rd, return_inverse = True)
    # slight backwards correction, to center according to occurances
    rd_unique = np.array([np.mean(dist[rd_idx==ii]) for ii in range(len(rd_unique))])

    # norm dictionary
    dictionary /= np.linalg.norm(dictionary,axis=0)

    # xcov   = dictionary.dot(dictionary.conj().T).flatten()
    # xcmean = np.array([np.mean(xcov[rd_idx == ii]) for ii in range(len(rd_unique))])
    # xcstd  = np.array([np.std(xcov[rd_idx == ii]) for ii in range(len(rd_unique))])
    # inv_norm = dictionary.shape[0]/dictionary.shape[1]
    # xcmean*= inv_norm 
    # xcstd *= inv_norm
    # xcmean = np.empty(rd_unique.shape, dtype = np.complex)
    # for atom in dictionary.T:
        # xcov   = np.einsum('i,j -> ij',atom,atom.conj())
        # xcmean += [np.mean(xcov.flatten()[rd_idx == ii]) for ii in range(len(rd_unique))]
    
    xcov   = np.array([np.outer(atom,atom.conj()).flatten() for atom in dictionary.T]).astype(np.float)
    xcov *=dictionary.shape[0]
    # xcmean = np.array([np.mean(xcov[:,rd_idx == rdi]) for ii in range(len(rd_unique))])
    # xcstd  = np.array([ np.std(xcov[:,rd_idx == rdi]) for ii in range(len(rd_unique))])
    xcovs  = np.empty((xcov.shape[0],len(rd_unique)))
    for ii in range(len(rd_unique)): # collect closeby values
        xcovs[:,ii] = np.array([np.mean(xc[rd_idx == ii]) for xc in xcov])
    xcmean = np.mean(xcovs,axis=0)
    xcstd  = np.std(xcovs,axis=0)

    return rd_unique, xcmean, xcstd, xcovs


import signal # UNIX ONLY
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
