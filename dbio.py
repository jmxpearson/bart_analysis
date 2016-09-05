"""
Contains files for reading data from the database.
"""

import numpy as np
import pandas as pd
import h5py
import pandas.io.pytables as pdtbl
import physutils

def make_path(*tup):
    abbr = ['p', 'd', 'c', 'u'][:len(tup)]
    nstrs = list(map(str, tup))
    pieces = [a + b for a,b in zip(abbr, nstrs)]
    return '/'.join(pieces)

def fetch(dbname, node, *args):
    """
    Given a node ('lfp', 'spikes', 'events', 'censor'), and
    a tuple of (patient, dataset, channel, unit), retrieves data.
    """
    target = node + '/' + make_path(*args)
    return pd.read_hdf(dbname, target)

def fetch_metadata(dbname, node, *args):
    # have to get metadata this way because we didn't store it via pandas
    # (which currently has no support for dataframe metadata)
    target = node + '/' + make_path(*args)
    fobj = h5py.File(dbname, 'r')
    attdict = dict(fobj[target].attrs)
    fobj.close()
    return attdict

def fetch_all_such(dbname, node, *args, **kwargs):
    """
    Given an incomplete specification in args, get all datasets consistent
    with it. Return a single dataframe.
    If database keys are precomputed to save time, they can be specified.
    """
    if 'keys' in kwargs:
        keys = kwargs['keys']
    else:
        keys = list(pdtbl.HDFStore(dbname).keys())

    glob = node + '/' + make_path(*args)
    # do really simple regex matching
    matches = [k for k in keys if glob in k]

    if matches:
        parts = []
        for m in matches:
            parts.append(fetch(dbname, m))
        excl = pd.concat(parts)
    else:
        excl = pd.DataFrame([])

    return excl

def get_censor(dbname, taxis, *args):
    """
    Convenience function for retrieving censoring intervals from the db
    and converting to logical arrays, one entry for each time point.
    args = patient, dataset, channel
    Assumes timestamp range equal to that of lfp.
    """
    
    # make sure tindex is of type float64
    ##### should be able to remove when pandas is upgraded to 0.13, which 
    ##### allows for float64 indices
    taxis = pd.Series(taxis.values.astype('float64'))

    # get censoring intervals, group by channel
    censors = fetch_all_such(dbname, 'censor', *args)
    if not censors.empty:
        censors = censors.groupby('channel')
    else:
        return censors
    
    # arrange start and stop times into linear sequence
    # (the extra pair of braces around the return value is to prevent
    # pandas from converting the time array to a series when apply gets only
    # a single return value (i.e., when censors has only a single group))
    if censors.ngroups > 1:
        flatfun = lambda x: x[['start', 'stop']].values.ravel()
    else:
        flatfun = lambda x: [x[['start', 'stop']].values.ravel()]
    censbins = censors.apply(flatfun)
    
    # append 0 and inf to bins
    censbins = censbins.apply(lambda x: np.append([0], x))
    censbins = censbins.apply(lambda x: np.append(x, np.inf))
    # bin times in taxis; censored bins will have even indices
    binnum = censbins.apply(lambda x: np.digitize(taxis, x))
    binnum = binnum.apply(lambda x: x % 2 == 0)

    excludes = pd.concat([pd.Series(b) for b in binnum], axis=1)
    excludes.columns = binnum.index  # channel names
    excludes.index = taxis
    
    return excludes

def censor_lfp(lfpset):
    return get_censor(lfpset.meta['dbname'], lfpset.dataframe.index, 
        *lfpset.meta['tuple'])
   
def censor_spikes(df, dbname, dtup):
    excludes = get_censor(dbname, df.index, *dtup)
    if not excludes.empty:
        excludes = excludes[excludes.columns.intersection(
            df.columns)]
        excl_vec = np.any(excludes.values, axis=1)
        newdf = df.copy()
        newdf[excl_vec] = np.nan
        return newdf
    else:
        return df

def load_spikes(dbname, dtup):
    spks = fetch(dbname, 'spikes', *dtup)

    spkbin = physutils.binspikes(spks, 0.05)  # use 50 ms bins

    return censor_spikes(spkbin, dbname, dtup)

def fetch_LFP(dbname, *tup):
    """ 
    Given a database and a tuple (tup), return an LFPset object.
    """

    lfp = fetch(dbname, 'lfp', *tup)
    lfp = lfp.set_index(['time', 'channel'])
    lfp = lfp['voltage']
    lfp = lfp.unstack()
    
    dt = lfp.index[1] - lfp.index[0]
    sr = 1. / dt
    meta = {'dbname': dbname, 'tuple': tup, 'sr': sr}

    return physutils.LFPset(lfp, meta)    

def fetch_all_such_LFP(dbname, *tup, **kwargs):
    """ 
    Given a database and a tuple (tup), return an LFPset object.
    """

    lfp = fetch_all_such(dbname, 'lfp', *tup, **kwargs)
    lfp = lfp.set_index(['time', 'channel'])
    lfp = lfp['voltage']
    lfp = lfp.unstack()
    
    dt = lfp.index[1] - lfp.index[0]
    sr = 1. / dt
    meta = {'dbname': dbname, 'tuple': tup, 'sr': sr}

    return physutils.LFPset(lfp, meta)
