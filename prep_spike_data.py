"""
For each patient, dataset, unit, prepare a dataframe to be used to fit a 
generalized linear model.
"""
import numpy as np
import os
import pandas as pd
import hephys.dbio as dbio

def set_intervals_to_true(df, starts, stops):
    pairs = zip(starts, stops)
    for p in pairs:
        df[slice(*p)] = 1
    return df

def make_regressor_is_in_trial(taxis, events):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='is_in_trial')
    starts = events['trial_start']
    stops = events['trial_over']
    return set_intervals_to_true(reg, starts, stops)

def make_regressor_is_control_trial(taxis, events):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='is_control_trial')
    if 'is_control' in events.columns:
        control_trials = events.query('is_control == 1')
        starts = control_trials['trial_start']
        stops = control_trials['trial_over']
        return set_intervals_to_true(reg, starts, stops)
    else:
        return reg

def make_regressor_is_outcome(taxis, events):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='is_outcome')
    starts = events['outcome']
    stops = events['trial_over']
    return set_intervals_to_true(reg, starts, stops)

def make_regressor_is_inflating(taxis, events):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='is_inflating')
    starts = events['start inflating']
    stops = events[['stop inflating', 'popped']].sum(axis=1)
    return set_intervals_to_true(reg, starts, stops)

def make_regressor_elapsed_time(taxis, events):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='elapsed_time')
    starts = events['start inflating']
    stops = events[['stop inflating', 'popped']].sum(axis=1)
    pairs = zip(starts, stops)
    for p in pairs:
        slc = slice(*p)
        time = reg[slc].index.values
        # log on this scale is linear on firing rate scale
        reg[slc] = np.log(time - time[0] + 1)  
    return reg

def make_regressor_is_banked(taxis, events):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='banked')
    trialset = events[['outcome', 'trial_over', 'banked']].dropna()
    starts = trialset['outcome']
    stops = trialset['trial_over']
    return set_intervals_to_true(reg, starts, stops)

def make_regressor_is_popped(taxis, events):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='popped')
    trialset = events[['outcome', 'trial_over', 'popped']].dropna()
    starts = trialset['outcome']
    stops = trialset['trial_over']
    return set_intervals_to_true(reg, starts, stops)

def make_regressor_trial_type(taxis, events, trial_type):
    # takes a dataframe of events and a time axis and returns a binary series
    reg = pd.Series(0, index=taxis, name='trial_type' + str(trial_type))
    evred = events[events['trial_type'] == trial_type]  # restrict trials
    starts = evred['trial_start']
    stops = evred['trial_over']
    pairs = zip(starts, stops)
    for p in pairs:
        reg[slice(*p)] = 1
    return reg

def make_regressor_frame(spikes, events):
    # get unique trial types
    unique_trial_types = np.unique(events['trial_type'])

    # make a regressor call for each trial type
    # the j=j is needed because of Python's late binding
    ttype_regressors = [lambda x, y, j=j: make_regressor_trial_type(x, y, j) for j in unique_trial_types]

    regressor_list = [make_regressor_is_in_trial, make_regressor_is_control_trial, make_regressor_is_outcome, make_regressor_is_inflating, make_regressor_elapsed_time, make_regressor_is_banked, make_regressor_is_popped] + ttype_regressors

    return pd.concat([f(spikes.index, events) for f in regressor_list], axis=1)

if __name__ == '__main__':
    # set a random seed
    np.random.seed(12345)

    # name of database to use
    dbname = os.path.join(os.getcwd(), 'data/bart.hdf5')

    # first, get a list of lfp channels
    setlist = pd.read_hdf(dbname, '/meta/spklist')

    for idx, row in setlist.iterrows():
        dtup = tuple(row)
        print dtup    

        spks = dbio.load_spikes(dbname, dtup)

        evt = dbio.fetch(dbname, 'events', *dtup[0:2])

        regressors = make_regressor_frame(spks, evt)

        # make spikes the first column in dataframe
        df = pd.concat([spks, regressors], axis=1)

        # write out
        outdir = os.path.join(os.getcwd(), 'data/')
        outfile = outdir + '.'.join(map(str, dtup)) + '.spkglmdata.csv'

        df.to_csv(outfile)