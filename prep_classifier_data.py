
import numpy as np
import pandas as pd
import physutils
from . import dbio
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
np.random.seed(12346)

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

        print(dtup)

        # read in data
        print('Reading LFP...')
        lfp = dbio.fetch_LFP(dbname, *dtup)

        # de-mean
        print('Removing mean...')
        lfp = lfp.demean()

        # break out by frequency bands
        print('Filtering by frequency...')
        filters = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        banded = lfp.bandlimit(filters)

        # decimate down to 40 Hz
        print('Decimating...')
        banded = banded.decimate(5)

        # get instantaneous power
        print('Calculating power...')
        banded = banded.instpwr()

        # handle censoring
        print('Censoring...')
        banded = banded.censor()

        # standardize per channel
        print('Standardizing regressors...')
        banded = banded.rzscore()

        # append to whole dataset
        allchans.append(banded.dataframe)

    # concatenate data from all channels
    print('Merging channels...')
    groupdata = pd.concat(allchans, axis=1)
    groupdata = physutils.LFPset(groupdata, banded.meta)

    # specify peri-event times
    dt = 1. / np.array(banded.meta['sr']).round(3)  # dt in ms
    Tpre = 2  # time relative to event to start
    Tpost = 1.5  # time following event to exclude

    # grab events (successful stops = true positives for training)
    print('Fetching events (true positives)...')
    evt = dbio.fetch(dbname, 'events', *name)
    stops = evt['banked'].dropna()
    pops = evt['popped'].dropna()
    starts = evt['start inflating']
    if 'is_control' in evt.columns:
        stops_free = evt.query('is_control == False')['banked'].dropna()
        stops_control = evt.query('is_control == True')['banked'].dropna()
        stops_rewarded = evt.query('trial_type != 4')['banked'].dropna()
        stops_unrewarded = evt.query('trial_type == 4')['banked'].dropna()
    else:
        stops_free = stops
        stops_rewarded = stops

    truepos = pd.DataFrame(stops_free.values, columns=['time'])
    truepos['outcome'] = 1

    # grab random timepoints (true negatives in training set)
    print('Generating true negatives...')
    maxT = lfp.index[-1]
    Nrand = truepos.shape[0]  # number to generate: same as number of true positives
    Ncand = 2000  # number to filter down to Nrand
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
    allevt['time'] = np.around(allevt['time'] / dt) * dt
    allevt = allevt.set_index('time')

    # get running average estimate of power at each timepoint of interest
    print('Grabbing data for each event...')
    meanpwr = pd.rolling_mean(groupdata.dataframe,
        np.ceil(Tpre / dt), min_periods=1)
    meanpwr.index = np.around(meanpwr.index / dt) * dt  # round index to nearest dt

    # make interaction terms and squares
    int_terms = []
    for i in range(len(meanpwr.columns)):
        for j in range(i + 1):
            if i == j:
                col = meanpwr.iloc[:, i] ** 2
                band, chan = col.name.split('.')
                col.name = "{}.{}.{}.{}".format(band, chan, band, chan)
            else:
                icol = meanpwr.iloc[:, i]
                jcol = meanpwr.iloc[:, j]
                col = icol * jcol
                iband, ichan = icol.name.split('.')
                jband, jchan = jcol.name.split('.')
                col.name = "{}.{}.{}.{}".format(iband, ichan, jband, jchan)

            int_terms.append(col)

    tset = pd.concat([allevt, meanpwr] + int_terms, axis=1, join='inner')
    tset = tset.dropna()  # can't send glmnet any row with a NaN

    # write out
    print('Writing out...')
    outdir = os.path.join(os.getcwd(), 'data/')
    outfile = outdir + str(dtup[0]) + '.' + str(dtup[1]) + '.lfpglmdata.csv'

    tset.to_csv(outfile)
