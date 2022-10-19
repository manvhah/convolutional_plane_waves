import numpy as np
from SoundfieldReconstruction import *

allparameters = [
        'decomposition', 'training',
        'evaluation',    'rec_freq',
        'spatial_sampling',
        # 'reg_lambda','reg_mu', 'reg_tolerance',
        # 'nof_components', 'alpha', 'batch_size',
        ]

### CONFIGTOOLS for labelling, monte carlo simulations etc
def binscale(List):
    return int(np.ceil(np.log2(len(List))))

def gen_config(config_index, **kwargs):
    """kwargs for rec config"""
    decomposition, training, \
            evaluation, rec_freq, spatial_sampling \
            = index_2_config(config_index)
            # reg_lambda, reg_mu, reg_tolerance \
            # nof_components, alpha, batch_size, \

    print ("\n> ", config_index, decomposition, 
            evaluation, rec_freq, spatial_sampling,)
            # reg_lambda, reg_mu, reg_tolerance,
            # nof_components, alpha, batch_size,

    otrain = default_soundfield_config(training, **kwargs)

    kwargs.update({
            "spatial_sampling": spatial_sampling,
            "loss_mode": 'sample',
            "frequency": rec_freq,
            })
    oval   = default_soundfield_config(evaluation, **kwargs)

    orec, otrans   = default_reconstruction_config(
            decomposition,
            # reg_lambda = reg_lambda,
            # reg_mu = reg_mu,
            # reg_tolerance = reg_tolerance,
            # nof_components = nof_components,
            # alpha = alpha,
            # batch_size = batch_size,
            **kwargs)

    return otrain, oval, orec, otrans, config_index

def index_2_config(config_index):
    if config_index == -1:
        config_index, _= config_2_index_design(-1)
    cfg = list()
    bshift = 0
    for PAR in ALLPARAMETERS:
        cfg.append(PAR[(config_index >> bshift) % (2**binscale(PAR))])
        bshift += binscale(PAR)
    return tuple(cfg)

def config_2_index_design(*args, **kwargs):
    if ('max' in args) or (-1 in args): # return maximum
        cfgmax = 0
        bshift = 0
        for PAR in ALLPARAMETERS:
            cfgmax += (len(PAR)-1)<<bshift
            bshift += binscale(PAR)
        return cfgmax, {}
    else:
        matches = list()
        for par, PAR in zip(allparameters, ALLPARAMETERS): # fill list
            match = list()
            if par in kwargs.keys(): # scan kwargs
                match = [idx for idx, val in enumerate(PAR) if val in kwargs[par]]
                kwargs.pop(par)
            if len(match) == 0:      # scan args
                match = [idx for idx, val in enumerate(PAR) if val in args]
            if len(match) == 0:      # full test design on this par
                match = np.arange(len(PAR))
            matches.append(match)

        from itertools import product
        index_dec = list(product(*matches))

        index_bin = list()
        for cfg in index_dec:
            bindex = 0
            bshift = 0
            for ii, PAR in enumerate(ALLPARAMETERS):
                bindex += cfg[ii] << bshift
                bshift += binscale(PAR)
            index_bin.append(bindex)

        return index_bin, kwargs
