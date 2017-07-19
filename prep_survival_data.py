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

    # get instantaneous power
    print('Calculating power...')
    banded = banded.instpwr()

    print('Smoothing...')
    Tsmooth = 0.5  # smoothing window (s)
    meta = banded.meta
    wsmooth = int(Tsmooth * meta['sr'])
    rmean = banded.rolling(wsmooth, min_periods=1, win_type='boxcar').mean()
    smoothed = physutils.LFPset(rmean, meta)

    # downsample to 10 Hz
    print('Decimating...')
    smoothed = smoothed.decimate(5).decimate(4)

    # handle censoring
    print('Censoring...')
    smoothed = smoothed.censor()

    # log power and standardize per channel
    print('Standardizing regressors...')
    meta = smoothed.meta
    smoothed = physutils.LFPset(np.log(smoothed), meta).rzscore()

    # mean across channels by band (i.e., geometric mean)
    # print('Mean across channels...')
    # chan_cols = [col for col in smoothed.columns if '.' in col]
    # band = {col: col.split('.')[0] for col in chan_cols}
    # rest = smoothed[smoothed.columns.difference(chan_cols)]
    # chan_means = smoothed.groupby(by=band, axis=1).mean()
    # smoothed = pd.concat([rest, chan_means], axis=1)

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
    dt = 1 / meta['sr']
    for trial, row in evt_tmp.iterrows():
        start, stop = row['start'], row['stop']
        this_chunk = smoothed.loc[start:stop].copy()
        if not this_chunk.empty:
            this_chunk['trial'] = trial
            this_chunk['event'] = 0  # no event until the last bin
            this_chunk.iloc[-1, this_chunk.columns.get_loc('event')] = int(row['event'])  # set last bin correctly
            this_chunk['ttype'] = int(row['trial_type'])
            this_chunk['rel_time'] = this_chunk.index - this_chunk.index[0] + dt

            chunks.append(this_chunk)

    # concatenate chunks, make non-power events their own series
    meanpwr = pd.concat(chunks)
    event = meanpwr['event']
    # ttype = pd.get_dummies(meanpwr['ttype'])
    ttype = meanpwr['ttype']
    time_in_trial = meanpwr['rel_time']
    trial = meanpwr['trial']
    meanpwr = meanpwr.drop(['event', 'ttype', 'rel_time', 'trial'], axis=1)

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


    trainset = pd.concat([event, time_in_trial, ttype, trial, meanpwr] + int_terms, axis=1, join='inner')
    trainset = trainset.dropna()  # can't send glmnet any row with a NaN

    # write out
    print('Writing out...')
    outdir = os.path.join(os.getcwd(), 'data/')
    outfile = outdir + str(dtup[0]) + '.' + str(dtup[1]) + '.lfpsurvdata.csv'

    trainset.to_csv(outfile)
