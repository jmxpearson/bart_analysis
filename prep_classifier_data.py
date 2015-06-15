import numpy as np
import pandas as pd
import physutils
import hephys.dbio as dbio
import warnings
import os

def within_range(test_value, anchor_list, radius_tuple):
    # return true when test_value is not within a radius tuple
    # of any value in anchor_list 
    # NOTE: both elements of radiust tuple must be positive!
    if radius_tuple < (0, 0):
        wrnstr = """Both elements of the exclusion radius must be positive.
        Answers may not mean what you think."""
        warnings.warn(wrnstr)

    dist = test_value - np.array(anchor_list)
    within_range = np.logical_and(dist > -radius_tuple[0],
        dist < radius_tuple[1]) 
    return np.any(within_range)

# define some useful numbers
np.random.seed(12345)

# open data file
dbname = os.path.expanduser('data/bart.hdf5')

# first, get a list of lfp channels
setlist = pd.read_hdf(dbname, '/meta/lfplist')

# group by (patient, dataset) entries:
groups = setlist.groupby(['patient', 'dataset'])

# iterate over groups
for name, grp in groups:

    allchans = []

    # iterate over channels within groups
    for ind, series in grp.iterrows():

        dtup = tuple(series.values)

        print dtup

        # read in data
        print 'Reading LFP...'
        lfp = dbio.fetch_LFP(dbname, *dtup)

        # de-mean
        print 'Removing mean...'
        lfp = lfp.demean()

        # break out by frequency bands
        print 'Filtering by frequency...'
        filters = ['delta', 'theta', 'alpha', 'beta']
        banded = lfp.bandlimit(filters)

        # decimate down to 40 Hz
        print 'Decimating...'
        banded = banded.decimate(5)

        # get instantaneous power
        print 'Calculating power...'
        banded = banded.instpwr()

        # handle censoring
        print 'Censoring...'
        banded = banded.censor()

        # standardize per channel
        print 'Standardizing regressors...'
        banded = banded.zscore()


        # append to whole dataset
        allchans.append(banded.dataframe)

    # concatenate data from all channels
    print 'Merging channels...'
    groupdata = pd.concat(allchans, axis=1)
    groupdata = physutils.LFPset(groupdata, banded.meta)

    # specify peri-event times
    dt = 1. / np.array(banded.meta['sr']).round(3)  # dt in ms
    Tpre = 2  # time relative to event to start
    Tpost = 1.5  # time following event to exclude

    # grab events (successful stops = true positives for training)
    print 'Fetching events (true positives)...'
    evt = dbio.fetch(dbname, 'events', *dtup[:2])['banked'].dropna()
    evt = np.around(evt / dt) * dt  # round to nearest dt
    # extend with nearby times
    truepos = (pd.DataFrame(evt.values, columns=['time']))
    truepos['outcome'] = 1

    # grab random timepoints (true negatives in training set)
    print 'Generating true negatives...'
    maxT = np.max(groupdata.index.values)
    # make some candidate random times
    Nrand = truepos.shape[0]  # number to keep
    Ncand = Nrand * 10  # number of candidates to generate
    candidates = np.random.rand(Ncand) * (maxT - Tpre) + Tpre
    candidates = np.around(candidates / dt) * dt  # round to nearest dt
    candidates = np.unique(candidates)
    np.random.shuffle(candidates)
    rand_times = filter(lambda x: ~within_range(x, truepos['time'], 
        (Tpre, Tpost)), candidates)[:Nrand]
    trueneg = pd.DataFrame(rand_times, columns=['time'])
    trueneg['outcome'] = 0

    # concatenate all training events
    allevt = pd.concat([truepos, trueneg])
    allevt = allevt.set_index('time')

    # get running average estimate of power at each timepoint of interest
    print 'Grabbing data for each event...'
    meanpwr = pd.rolling_mean(groupdata.dataframe, 
        np.ceil(Tpre / dt), min_periods=1)
    meanpwr.index = np.around(meanpwr / dt) * dt  # round index to nearest dt
    tset = pd.concat([allevt, meanpwr], axis=1, join='inner')
    tset = tset.dropna()  # can't send glmnet any row with a NaN

    # write out
    print 'Writing out...'
    outdir = os.path.join(os.getcwd(), 'data/')
    outfile = outdir + str(dtup[0]) + '.' + str(dtup[1]) + '.lfpglmdata.csv'

    tset.to_csv(outfile)