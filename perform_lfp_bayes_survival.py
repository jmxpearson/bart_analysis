# run Bayesian survival analysis for all subjects

import numpy as np
import pandas as pd
import pystan
import pickle

def make_init_fun(M, Ntypes):
    # returns function that returns inits for model parameters
    # needed because Cauchy dists result in huge outliers
    return lambda: ({'tau_global': 0.1 * np.abs(np.random.randn()),
                     'tau_local': 0.1 * np.abs(np.random.randn(M)),
                     'alpha_raw': 0.01 * 2 * (np.random.rand() - 0.5),
                     'beta_raw': 0.1 * np.random.randn(M),
                     'mu': 0.1 * np.random.randn(),
                     'mm': 2 + 0.1 * np.abs(np.random.rand(Ntypes)),
                     'ss': 5 + 0.1 * np.abs(np.random.rand(Ntypes)),
                    })

def run_stan(dtup):
    # fit stan model, save output
    # dtup is a (subject id, dataset) pair
    fname = '.'.join([str(x) for x in dtup])
    dat = pd.read_csv('data/' + fname + '.lfpsurvdata.csv', index_col=0)

    # extract some variables
    event = dat.event
    time = dat['rel_time']
    ttype = dat['ttype']
    X = dat.drop(['event', 'rel_time', 'ttype'], axis=1)
    dt = X.index[1] - X.index[0]
    Nobs, M = X.shape
    Ntypes = len(ttype.unique())

    # get data in shape for feeding to Stan
    survival_dat = ({'X': X,
                     'Nobs': Nobs,
                     'Ntypes': Ntypes,
                     'ttype': ttype,
                     'M': M,
                     'event': event,
                     'time': time,
                     'dt': dt})

    # build Stan model
    sm = pystan.StanModel(file='survival.stan')

    # sample
    fit = sm.sampling(data=survival_dat, iter=2000, thin=5, chains=4,
                      seed=12345,
                      pars=['tau_global', 'tau_local', 'beta', 'mu', 'mm', 'ss'],
                      init=make_init_fun(M, Ntypes))

    # save output
    # BOTH model and samples need to be pickled
    with open('data/' + fname + '.stan_model', 'wb') as f:
        pickle.dump(sm, f)
    with open('data/' + fname + '.stan_samples', 'wb') as f:
        pickle.dump(fit, f)

    return None

subjs = pd.read_csv('data/lfp_channel_file.csv', header=None).drop(2, axis=1).drop_duplicates()

for dtup in [(18, 1)]:
# for _, dtup in subjs.iterrows():
    print("Subject {}, Set {}".format(dtup[0], dtup[1]))
    run_stan(dtup)
