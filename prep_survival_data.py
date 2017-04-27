from __future__ import division
import numpy as np
import pandas as pd
import physutils
import dbio
import warnings
import os

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

    dtup = name
    print(dtup)

    print('Reading LFP...')
    lfp = dbio.fetch_all_such_LFP(dbname, *dtup)

    print('Removing mean...')
    if lfp.shape[1] > 1:
        lfp = lfp.demean_global()

    lfp = lfp.demean()

    print('Filtering by frequency...')
    filters = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    banded = lfp.bandlimit(filters)

    # decimate down to 40 Hz
    print('Decimating...')
    banded = banded.decimate(5).decimate(4)

    # get instantaneous power
    print('Calculating power...')
    banded = banded.instpwr()

    # handle censoring
    # print('Censoring...')
    # banded = banded.censor()

    # standardize per channel
    print('Standardizing regressors...')
    banded = banded.rzscore()

    # grab events (successful stops = true positives for training)
    print('Processing events...')
    evt = dbio.fetch(dbname, 'events', *dtup)
    cols = ['banked', 'popped', 'start inflating', 'trial_type']

    if 'is_control' in evt.columns:
        evt_tmp = evt.query('is_control == False')[cols]
    else:
        evt_tmp = evt.loc[:, cols]

    # add a binary column (1 = voluntary stop)
    evt_tmp['event'] = np.isnan(evt_tmp['popped']).astype('int')

    # add a column for stop time (regardless of cause)
    evt_tmp['stop'] = evt.loc[:, ['banked', 'popped']].mean(axis=1)

    # drop unneeded columns
    evt_tmp = evt_tmp.drop(['banked', 'popped'], axis=1)
    evt_tmp = evt_tmp.rename(columns={'start inflating': 'start'})

    # remove unneeded time points
    chunks = []
    dt = banded.index[1] - banded.index[0]
    for trial, row in evt_tmp.iterrows():
        start, stop = row['start'], row['stop']
        this_chunk = banded.loc[start:stop].copy()
        if not this_chunk.empty:
            this_chunk['event'] = 0  # no event until the last bin
            this_chunk.iloc[-1, this_chunk.columns.get_loc('event')] = int(row['event'])  # set last bin correctly
            this_chunk['ttype'] = int(row['trial_type'])
            this_chunk['rel_time'] = this_chunk.index - this_chunk.index[0] + dt

            chunks.append(this_chunk)

    # concatenate chunks, make non-power events their own series
    meanpwr = pd.concat(chunks)
    event = meanpwr['event']
    ttype = pd.get_dummies(meanpwr['ttype'])
    time_in_trial = meanpwr['rel_time']
    ttype.columns = ['ttype' + str(idx) for idx in ttype.columns]
    ttype = ttype.drop('ttype1', axis=1)
    meanpwr = meanpwr.drop(['event', 'ttype', 'rel_time'], axis=1)

    # standardize
    meanpwr = meanpwr.apply(lambda x: (x - x.mean())/x.std())

    # make interaction terms and squares
    int_terms = []
    # for i in range(len(meanpwr.columns)):
    #     for j in range(i + 1):
    #         if i == j:
    #             col = meanpwr.iloc[:, i] ** 2
    #             band, chan = col.name.split('.')
    #             col.name = "{}.{}.{}.{}".format(band, chan, band, chan)
    #         else:
    #             icol = meanpwr.iloc[:, i]
    #             jcol = meanpwr.iloc[:, j]
    #             col = icol * jcol
    #             iband, ichan = icol.name.split('.')
    #             jband, jchan = jcol.name.split('.')
    #             col.name = "{}.{}.{}.{}".format(iband, ichan, jband, jchan)
    #
    #         col = (col - col.mean())/col.std()
    #         int_terms.append(col)


    trainset = pd.concat([event, time_in_trial, ttype, meanpwr] + int_terms, axis=1, join='inner')
    trainset = trainset.dropna()  # can't send glmnet any row with a NaN

    # write out
    print('Writing out...')
    outdir = os.path.join(os.getcwd(), 'data/')
    outfile = outdir + str(dtup[0]) + '.' + str(dtup[1]) + '.lfpsurvdata.csv'

    trainset.to_csv(outfile)
