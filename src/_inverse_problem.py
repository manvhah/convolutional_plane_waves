import numpy as np
from ._tools import (
    complex_interlace,
    complex_interlace_2d,
    complex_delace,
    timeout,
)
import warnings
import time

import cvxpy as cp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ridge_regression, RidgeCV
from sklearn.linear_model import LassoLars, LassoLarsCV, lars_path, LassoLarsIC

from multiprocessing import cpu_count
from pebble import ProcessPool
from functools import partial

import platform
if platform.system() == 'Darwin': 
    DEBUG = True
    MAX_NOF_CPUS = 2
elif 'arch' in platform.release(): 
    DEBUG = False
    MAX_NOF_CPUS = 12
else: 
    DEBUG = False
    MAX_NOF_CPUS = 16

TIMEOUT_CVX = 300
TIMEOUT_PER_WORKER_CVX = 30
TIMEOUT_PER_WORKER_LOCAL = 100
CHUNK_SIZE = 20

def _loss_fn(X, Y, w):
    return cp.norm2(cp.matmul(X, w) - Y) ** 2

def _objective_fn(X, Y, w, lambd, p):
    return _loss_fn(X, Y, w) + lambd * cp.pnorm(w, p=p) ** p

def _nmse(X, Y, w):
    return (1.0 / X.shape[0]) * _loss_fn(X, Y, w).value

def lcurve_func(yl, Al):
    from lcurve_functions import csvd, l_curve
    n0, n1 = Al.shape
    if n0 > n1: # select max number of basis funcs
        ridx = np.random.choice(n0,n1)
        Al = Al[ridx,:]
        yl = yl[ridx]

    # if n0 > n1
    U, S, V = csvd(Al)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lambd = l_curve(U, S, yl, debug = False)
    return lambd

def _run_cvx(b_vector, A, reg_tol, reg_lambda, pnorm, lcurve, **kwargs_solver):
    """
    run cvx regression, filter zero components
    """
    w = cp.Variable(A.shape[1], complex=True)
    lambd = cp.Parameter(nonneg=True, value=reg_lambda)
    midx = b_vector.nonzero()[0]

    if reg_tol:
        opttype = 'constraint'
        problem = cp.Problem(
            cp.Minimize(cp.norm(w, pnorm)),
            constraints=[
                _loss_fn(A[midx, :], b_vector[midx], w)
                <= reg_tol * midx.size
            ],
        )
    else:
        if (pnorm ==2) & lcurve: 
            # regularization paraemters from l-curve not reported back!
            lambd = cp.Parameter(nonneg=True, value=lcurve_func(b_vector, A))
        opttype = 'penalty'
        problem = cp.Problem(cp.Minimize(
            _objective_fn(A[midx, :], b_vector[midx], w, lambd, pnorm)
            ),
        )

    problem.solve(**kwargs_solver)  # solving the problem
    # if problem.status in ["infeasible", "unbounded"]:
        # print(problem.status)
    # else:
        # print("Optimal value: %s" % problem.value)
    # print('\ncvx finished.',opttype,' applied. results:', 
            # '\n\t > mse\t', problem._value,
            # '\n\t > constraint\t', reg_tol * midx.size,
            # '\n\t > nmse\t\t', 10*np.log10(_nmse(A[midx,:],b_vector[midx],w.value)),
            # '\n\t > penalty\t', reg_lambda*cp.pnorm(w, pnorm).value,
            # len(midx)
            # )
    return w.value, lambd.value, problem.status


def complex_cvx(A, y, reg_tol, reg_lambda, reg_norm, lcurve=False):
    """
    cvx optimization wrapper for complex data,

    parameters:

        A           transfer matrix
        y           data to transform
        reg_tol     tolerance for regularization
        reg_lambda  regularization parameter
        reg_norm    norm order for regularization, int: 1 for lasso, 2 for ridge
        lcurve      for reg_norm ==2, use l-curve?

    filters empty columns in data y
    parallelizing along columns if y is 2D,
    """

    if np.prod(y.shape) > y.shape[0]:  # iterate along axis 0
        x = np.zeros((A.shape[1], y.shape[1]), dtype=np.complex128)
        nz_cols = np.sum(np.abs(y), axis=0) > 0
        y = np.compress(nz_cols, y, axis=1)

        ## parallel
        pfunc = partial(
            _run_cvx,
            A = A,
            reg_tol = reg_tol,
            reg_lambda = reg_lambda,
            pnorm = reg_norm,
            lcurve = lcurve,
            verbose = False,
        )
        if DEBUG: ## sequential
            tmp = [_run_cvx(b_vec, A, reg_tol, reg_lambda, reg_norm, 
                    lcurve = lcurve, verbose = False,)
                    for ii, b_vec in enumerate(y.T)]
        else: ## parallel
            nof_workers = np.min([cpu_count(), MAX_NOF_CPUS, y.shape[0]])
            with ProcessPool(max_workers=nof_workers) as pool:
                tmp = pool.map(pfunc, y.T, 
                        chunksize = 50,
                        timeout = TIMEOUT_PER_WORKER_CVX)
            tmp = [tt for tt in tmp.result()]


        converged = np.array([(tt[2] == 'optimal') for tt in tmp])
        nz_cols[nz_cols] &= converged

        if lcurve:
            reg_lambda = np.mean([tt[1] for tt in tmp])

        tmp = np.array([tt[0] for tt in tmp],dtype=object)
        x[:, nz_cols] = np.vstack(tmp[converged]).T
        return x, reg_lambda
    else:
        with timeout(seconds=TIMEOUT_CVX):
            x, reg_lambda, _ =  _run_cvx(
                y.squeeze(), A, reg_tol, reg_lambda, 
                pnorm = reg_norm, 
                lcurve = lcurve,
                verbose = False,
            )
        return x, reg_lambda


def _run_lasso_lars(A, y, alpha, midx = None, xval = False, debug = False):
    # regularization = reg_lambda # default 1
    # alpha = float(regularization) / (A.shape[1]/2)  # account for scaling
    if np.any(midx):
        y = y[midx]
        A = A[midx,:]

    larsopts = {
            "precompute"    : 'auto',
            "verbose"       : False,
            "positive"      : False,
            "fit_intercept" : False,
            "normalize"     : False,
            }

    if not xval: # solve for given alpha
        lasso_lars = LassoLars(alpha = alpha, fit_path = False, **larsopts)
    else:
        if y.shape[0] < 3: return np.zeros(A.shape[1])
        lasso_lars = LassoLarsCV(cv=y.shape[0], n_jobs = MAX_NOF_CPUS, **larsopts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso_lars.fit(A, np.squeeze(y))
    coefs =  lasso_lars.coef_.T

    if debug: # regularizaiton & testing
        idx = 23
        import matplotlib.pyplot as plt
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # #############################################################################
        # LassoLars with path
        lasso_lars = make_pipeline(StandardScaler(with_mean=False),LassoLars(alpha = 0,
            fit_path = True, **larsopts))
        lasso_lars.fit(A, y[:,idx], Xy=cov)
        plt.figure(123)
        plt.clf()
        plt.plot(lasso_lars[-1].alphas_,lasso_lars[-1].coef_path_.T);

        # #############################################################################
        # lars_path
        #larspath2
        print("Computing regularization path using the LARS ...")
        _, _, coefs = lars_path( A, y[:,idx], Xy=cov[:,idx], method='lasso', verbose=True)
        xx = np.sum(np.abs(coefs.T), axis=1)
        xx /= xx[-1]
        plt.plot(xx, coefs.T)
        ymin, ymax = plt.ylim()
        plt.figure(124)
        plt.clf()
        plt.vlines(xx, ymin, ymax, linestyle='dashed')
        plt.xlabel('|coef| / max|coef|')
        plt.ylabel('Coefficients')
        plt.title('LASSO Path')
        plt.axis('tight')

        # #############################################################################
        # Lasso & LassoLars crossvalidation
        EPSILON = 1e-4
        print("Computing regularization path , cross validation...")
        model = LassoCV(cv=10).fit(A, y[:,idx])
        plt.figure(125)
        plt.clf()
        plt.subplot(211)
        plt.semilogx(model.alphas_ + EPSILON, model.mse_path_, ':')
        plt.semilogx(model.alphas_ + EPSILON, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
        plt.axvline(model.alpha_, linestyle='--', color='k', label='alpha CV')
        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: Coordinate descent')
        plt.axis('tight')
        model = LassoLarsCV(cv=10).fit(A, y[:,idx])
        plt.subplot(212)
        plt.semilogx(model.cv_alphas_ + EPSILON, model.mse_path_, ':')
        plt.semilogx(model.cv_alphas_ + EPSILON, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
        plt.axvline(model.alpha_, linestyle='--', color='k', label='alpha CV')
        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: Lars')
        plt.axis('tight')

        # #############################################################################
        # LassoLarsIC: least angle regression with BIC/AIC criterion

        def plot_ic_criterion(model, name, color):
            criterion_ = model.criterion_
            plt.semilogx(model.alphas_ + EPSILON, criterion_, '--', color=color,
                         linewidth=3, label='%s criterion' % name)
            plt.axvline(model.alpha_ + EPSILON, color=color, linewidth=3,
                        label='alpha: %s estimate' % name)
            plt.xlabel(r'$\alpha$')
            plt.ylabel('criterion')

        model_bic = LassoLarsIC(criterion='bic')
        model_aic = LassoLarsIC(criterion='aic')
        bic, aic = list(), list()
        for ii in range(y.shape[1]):
            model_bic.fit(A, y[:,ii])
            bic.append(model_bic.alpha_)
            model_aic.fit(A, y[:,ii])
            aic.append(model_aic.alpha_)
        alpha_bic_ = model_bic.alpha_
        alpha_aic_ = model_aic.alpha_
        plt.figure(126)
        plt.clf()
        plot_ic_criterion(model_aic, 'AIC', 'b')
        plot_ic_criterion(model_bic, 'BIC', 'r')
        plt.legend()
        plt.title('Information-criterion for model selection')

        plt.show()

    return coefs


def _irun_lasso_lars(ii, midx, A, y, alpha, xval=False):
    return  _run_lasso_lars( A, y[:,ii,np.newaxis], # Re and Im always at the same midx
            alpha, midx = midx[ii], xval=xval, debug = False,)

def _iridge_regression(ii, midx, A, y, alpha, cv=False, **kwargs):
    if cv:
        reg = RidgeCV(alphas=np.logspace(-5,2,15))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reg.fit( A, y[:,ii], sample_weight = midx[ii], **kwargs)
        return reg.coef_
    else:
        return ridge_regression(
            A, y[:,ii], alpha[ii], sample_weight = midx[ii], **kwargs)


def solve_inv_problem(method, A, y, opts=dict()):
    """
    Solving inverse problem y = A.dot(x) with given regularization method.

    parameters:
        method  'ridge', 'lasso', 'omp' or 'group_lasso'.
                lasso   takes complex numbers, but is slower
                omp     complex omp, pickled from sklearn

                use for example 'group_lasso' for efficient lasso on complex data

        A       MxN transfer matrix

        y       MxP complex observed data, batches along 0-axis

        optimization settings, dict
                modified and updated with defaults (if required) inplace

                defaults:
                    {
                        "l1_reg": 1e-12,
                        "n_iter": 100,
                        "reg_tol_sig2": 1,
                        "reg_tolerance": 0,
                        "reg_lambda": 1e-2,
                    }

    returns:
        x       NxP complex weightings
        y_hat   transformed data
    """
    default_opts = dict(
        {
            "l1_reg": 1e-12,
            "n_iter": 100,
            "reg_tol_sig2": 1,
            "reg_tolerance": 0,
            "reg_lambda": 1e-2,
        }
    )
    # make sure the original dictionary is returned updated
    _ = [
        opts.update({elem: default_opts[elem]})
        for elem in default_opts
        if not elem in opts.keys()
    ]

    print("{:_<26}".format(" > " + method + " "), end="_")
    tic = time.time()


    if "lasso_lars" in method:
        uw = set(opts) - set(
            ["reg_tolerance", "l1_reg", "reg_lambda", "n_iter"]
        )
        for uwk in uw:
            del opts[uwk]

        # condition data
        if y.ndim < 2:
            y = np.expand_dims(y, 1)

        nonzero_cols = np.sum(np.abs(y), axis=0) > 0

        Aci = np.real(complex_interlace_2d(A))
        yci = complex_interlace(y, 0)

        if (y.shape[1] == 1) or (y.nonzero()[0].size == y.size):  
            # full data set or single vector
            xci = _run_lasso_lars( Aci, yci, 
                    midx = np.squeeze(yci != 0),
                    alpha = opts['reg_lambda'],
                    xval = ('xv' in method),)
        else: 
            # several, possibly sparsely sampled targets
            xci = np.zeros((Aci.shape[1], yci.shape[1]), dtype=y.dtype)
            nonzero_cols = np.sum(np.abs(yci), axis=0) > 0
            yci = np.compress(nonzero_cols, yci, axis=1)
            num_patches = yci.shape[1]

            pfunc = partial(_irun_lasso_lars, 
                    midx = (yci != 0).T,
                    A = Aci, 
                    y = yci, 
                    alpha = opts['reg_lambda'],
                    xval = ('xv' in method))

            if ('xv' in method) or DEBUG: ## sequential
                tmp = [pfunc(ii) for ii in range(num_patches)]
            else: ## parallel
                nof_workers = np.min([cpu_count(), MAX_NOF_CPUS, num_patches])
                with ProcessPool(max_workers=nof_workers) as p:
                    tmp = p.map(pfunc, range(num_patches), 
                            chunksize = CHUNK_SIZE,
                            timeout = TIMEOUT_PER_WORKER_LOCAL)
                tmp = [tt for tt in tmp.result()]
            xci[:, nonzero_cols] = np.squeeze(tmp).T

        x = complex_delace(xci)
        y_hat = A.dot(x)

    elif ("ridge" in method) and not ("cvx" in method):
        uw = set(opts) - set(["reg_lambda"])#, "reg_tolerance", "reg_tol_sig2"])
        for uwk in uw:
            del opts[uwk]
        solver_opts = {}

        if y.ndim < 2:
            y = np.expand_dims(y, 1)

        Aci = np.real(complex_interlace_2d(A))
        yci = complex_interlace(y,0)

        xci = np.zeros((Aci.shape[1], yci.shape[1]), dtype=y.dtype)
        nonzero_cols = np.sum(np.abs(yci), axis=0) > 0
        yci = np.compress(nonzero_cols, yci, axis=1)
        num_patches = yci.shape[1]

        alpha = opts['reg_lambda']*np.ones(yci.shape[1])

        if (y.shape[1] == 1) or (y.nonzero()[0].size == y.size):
            # full data set or single vector

            if ("xval" in method):
                reg = RidgeCV(alphas=np.logspace(-4,2,25),
                        alpha_per_target=True,
                        store_cv_values = True)
                if (y.shape[1] == 1):
                    midx = np.squeeze(yci != 0,1)
                    reg.fit(Aci[midx,:], yci[midx,:])
                else:
                    reg.fit(Aci,yci)
                alpha = np.mean(reg.alpha_)
                # print("cross validation, alpha selected:", reg.alpha_)
                # print("xval values", [np.round(np.linalg.norm(rcv)) for rcv in reg.cv_values_.T],)
                opts.update({"reg_lambda":np.mean(alpha)})
                xci = reg.coef_.T
            else:
                if ("lc" in method):
                    alpha = lcurve_func(y, Aci)
                    opts.update({"reg_lambda":alpha})
                    print("lcurve alpha: ",alpha)
                midx = np.squeeze(yci != 0,1) # new
                xci = ridge_regression( Aci[midx,:], yci[midx,:], alpha, **solver_opts).T

        else: # several, possibly sparsely sampled targets
            if ("lc" in method):
                if ("xval" in method):
                    print("\nWarning! LCURVE SPECIFIED. OVERWRITING CV!\n")
                nof_workers = np.min([cpu_count(), MAX_NOF_CPUS, num_patches])
                pfunc = partial(lcurve_func, Al = Aci)
                with ProcessPool(max_workers=nof_workers) as p:
                    tmp = p.map(pfunc, yci.T, 
                                chunksize = CHUNK_SIZE,
                                timeout = TIMEOUT_PER_WORKER_LOCAL)
                alpha = np.array([tt for tt in tmp.result()])
                # alpha = [lcurve_func(yy, Aci) for yy in yci.T]
                opts.update({"reg_lambda":np.mean(alpha)})
                print("lcurve mean alpha: ",np.mean(alpha))

            pfunc = partial(_iridge_regression, 
                    midx = (yci != 0).T,
                    A = Aci, y = yci, 
                    alpha = alpha,
                    cv = ('xval' in method),
                    **solver_opts)

            if DEBUG or ('xv' in method): ## sequential
                tmp = [pfunc(ii) for ii in range(num_patches)]
            # if not ('xv' in method): ## parallel
            else:
                nof_workers = np.min([cpu_count(), MAX_NOF_CPUS, num_patches])
                with ProcessPool(max_workers=nof_workers) as p:
                    tmp = p.map(pfunc, np.arange(num_patches), 
                                chunksize = CHUNK_SIZE,
                                timeout = TIMEOUT_PER_WORKER_LOCAL)
            tmp = [tt for tt in tmp.result()]
            xci[:, nonzero_cols] = np.squeeze(tmp).T

        x = complex_delace(xci)
        y_hat = A.dot(x)

    else: # solve problem using cvx
        uw = set(opts) - set(["reg_lambda", "reg_tolerance", "reg_tol_sig2"])
        for uwk in uw:
            del opts[uwk]

        if "lasso" in method:
            opts.update({"reg_norm": 1})
        else: # if "ridge" in method:
            opts.update({"reg_norm": 2})

        x, reg_lambda = complex_cvx(
            A,
            y,
            opts["reg_tolerance"] * opts["reg_tol_sig2"],
            opts["reg_lambda"],
            opts["reg_norm"],
            ("lc" in method), # regularization paraemters from l-curve?
        )
        opts.update({"reg_lambda":reg_lambda})
        # print(opts)
        if x is None:
            x = np.zeros((A.shape[1],),dtype=np.complex)
        y_hat = A.dot(x)

    print("{:_>23}".format(" in {:.2f}".format(time.time() - tic)))
    return x, y_hat
