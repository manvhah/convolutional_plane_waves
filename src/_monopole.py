import numpy as np
from ._const import *
"""!@package
Monopole module
"""

def random_phase(N = 1,seed = None, block_reset = False):
    """!
    @param *N: number of samples, default one
    @param *seed: set to force random number seed
    @param *block_reset: block seed set

    @return phases: uniformly distributed random phase vector with unit amplitude
    """
    # could also used np.random.set_state to reset to e.g. initial state
    if not block_reset:
        np.random.seed(seed)
    phases = np.exp(2j*np.pi*np.random.uniform(0,1,N))
    return phases

def monopole(r, amplitude = None, f = None):
    """!
    Sound pressure exerted by a monopole in free air at distance r. Give
    amplitude if desired. Add frequency and complex amplitude to include phase,
    else the absolute pressure is returned.
    pressure is set to zero where r == 0.
    @params r: distance
    @params *amplitude: may be complex
    @params *f: frequency
    @return pressure: complex sound pressure
    """

    if amplitude is None:
        amplitude = 1
    r[r == 0.0] = np.infty

    pressure = amplitude/(4*np.pi*r)
    if f is not None:
        pressure *= np.exp(2j*np.pi*f*r/c0)
    else:
        pressure = np.abs(pressure)
    return pressure


def distance(x, X):
    """!
    Calculate euclidian distance from a point x to a field X
    @param x: point vector of dimension D
    @param X: list of D coordinate vectors
    """
    acc = (X[0] - x[0])**2
    for dim in range(1,len(x)):
        acc += (X[dim] - x[dim])**2
    return np.sqrt(acc)


if __name__ == '__main__':
    """!
    toolbox demo, simulate N monopoles of frequency f in a square free air domain
    """
    lim = 4
    no_monos = 5
    f = 117

    # setup
    x = np.arange(-lim,lim,.05)
    y = np.arange(-lim,lim,.05)
    xx, yy = np.meshgrid(x,y)

    # gen monopoles in free field
    mpx = np.random.uniform(-lim*0.8, lim*0.8, no_monos)
    mpy = np.random.uniform(-lim*0.8, lim*0.8, no_monos)
    pfield = [monopole( distance(mp, np.stack((xx,yy))),
        amplitude = random_phase(), f=f) for mp in zip(mpx,mpy)]
    pfield = np.sum(pfield,0)

    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(11,4))
    plt.clf()
    plt.subplot(121)
    plotfield(x,y, pfield, np.stack((mpx,mpy)))
    plt.title('p')
    plt.subplot(122)
    pfield = 20*np.log10(np.abs(pfield))
    plotfield(x,y, pfield, np.stack((mpx,mpy)))
    plt.title('SPL rel 1 Pa')

