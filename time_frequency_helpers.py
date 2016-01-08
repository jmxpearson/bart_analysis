import numpy as np
import pandas as pd
import physutils as phys
import physutils.bootstrap as boot
import physutils.tf as tf
import matplotlib.pyplot as plt
import dbio
from scipy.interpolate import interp1d

def load_and_preprocess(dbname, dtup):
    """
    Load and preprocess LFP data.
    """
    # load data
    lfp = dbio.fetch_all_such_LFP(dbname, *dtup)

    # handle FHC recordings
    standard_sr = 200.
    if lfp.meta['sr'] != standard_sr:
        dt = 1/standard_sr
        T0, Tf = lfp.index[0], lfp.index[-1]
        tnew = np.arange(T0, Tf, dt)
        f = lambda x: interp1d(lfp.index, x)(tnew)
        new_lfp = pd.DataFrame(np.apply_along_axis(f, 0, lfp.dataframe.values),
                              index=tnew, columns=lfp.columns)
        lfp = phys.LFPset(new_lfp)

    # censor and robustly zscore
    lfpmz = lfp.censor().rzscore()

    return lfpmz

def avg_time_freq_arrays(dataframe, times, Tpre, Tpost,
                         expand=1.0, method='wav', normfun=None, **kwargs):
    """
    Stolen and modified from physutils.bootstrap.py.
    Splits a dataframe around index values in iterable times and
    returns a time-frequency matrix for each event, averaged across
    dataframe channels (columns).
    """
    if method == 'wav':
        callback = tf.continuous_wavelet
    else:
        callback = tf.spectrogram

    dT = Tpost - Tpre
    Tpre_x = Tpre - expand * dT
    Tpost_x = Tpost + expand * dT

    nchan = dataframe.shape[1]
    spectra = None

    for chan, series in dataframe.iteritems():
        # get time-frequency matrix for each event
        this_spec0, taxis, faxis = tf._per_event_time_frequency(series,
            callback, times[0], Tpre_x, Tpost_x, complete_only=False, **kwargs)
        this_spec1, taxis, faxis = tf._per_event_time_frequency(series,
            callback, times[1], Tpre_x, Tpost_x, complete_only=False, **kwargs)

        this_spectra = this_spec0 + this_spec1

        if spectra is None:
            spectra = this_spectra[:]
        else:
            for idx, ts in enumerate(this_spectra):
                spectra[idx] += ts


    # normalize
    if normfun:
        spectra = normfun(spectra)

    # convert from dataframes to ndarrays
    spectra = [s.values/nchan for s in spectra]

    # make a dataframe containing all times, labeled by event type
    labels0 = np.zeros((len(times[0]),))
    labels1 = np.ones((len(times[1]),))
    alllabels = np.concatenate((labels0, labels1))

    # remove trials with nans
    sfinal, lfinal = zip(*[(s, l) for (s, l) in zip(spectra, alllabels)
                          if not np.any(np.isnan(s))])

    return np.array(sfinal), np.array(lfinal).astype('int'), taxis, faxis

def trials_to_clusters(spectra, alllabels, thresh, taxis, Tpre, Tpost,
                       niter=1000, pval=0.05,
                       doplot=True, diff_fun=boot.F_stat,
                       mass_fun=boot.log_F_stat):
    """
    Takes an iterable of time-frequency matrices and labels and performs
    bootstrap resampling to determine significant clusters in the average
    time-frequency plot. (Stolen and modified from physutils/bootstrap.py)
    """
    thlo = thresh[0]
    thhi = thresh[1]

    # now loop
    cluster_masses = []
    for ind in np.arange(niter):
        labels = np.random.permutation(alllabels)

        # find clusters based on diff_fun
        pos = boot.make_thresholded_diff(spectra, labels, hi=thhi, diff_fun=diff_fun)
        neg = boot.make_thresholded_diff(spectra, labels, lo=thlo, diff_fun=diff_fun)

        # label clusters
        posclus = boot.label_clusters(pos)
        negclus = boot.label_clusters(neg)

        # calculate mass map
        mass_map = mass_fun(spectra, labels)

        # mask mass map based on clusters
        pos_mass = np.ma.masked_array(data=mass_map, mask=pos.mask)
        neg_mass = np.ma.masked_array(data=mass_map, mask=neg.mask)

        # get all masses for clusters other than cluster 0 (= background)
        cluster_masses = np.concatenate([
            cluster_masses,
            boot.get_cluster_masses(pos_mass, posclus)[1:],
            boot.get_cluster_masses(neg_mass, negclus)[1:]
            ])

    # extract cluster size thresholds based on null distribution
    cluster_masses = np.sort(cluster_masses)
    plo = pval / 2.0
    phi = 1 - plo
    Nlo = np.floor(cluster_masses.size * plo).astype('int')
    Nhi = np.ceil(cluster_masses.size * phi).astype('int')
    Clo = cluster_masses[Nlo]
    Chi = cluster_masses[Nhi]

    # get significance-masked array for statistic image
    truelabels = alllabels
    signif = boot.threshold_clusters(spectra, truelabels, lo=thlo,
        hi=thhi, keeplo=Clo, keephi=Chi, diff_fun=diff_fun,
        mass_fun=mass_fun)

    # make contrast image
    img0 = np.nanmean(spectra[truelabels == 0, :, :], axis=0)
    img1 = np.nanmean(spectra[truelabels == 1, :, :], axis=0)
    contrast = (img0 / img1)

    # use mask from statistic map to mask original data
    mcontrast = np.ma.masked_array(data=contrast, mask=signif.mask)
    to_return = np.logical_and(taxis >= Tpre, taxis < Tpost)

    return mcontrast[to_return], taxis[to_return]

def get_spectra_and_labels(dbname, tuplist, event_labels, Tpre, Tpost, freqs, normfun):
    spectra_list = []
    labels_list = []

    for dtup in tuplist:
        print dtup

        lfp = load_and_preprocess(dbname, dtup)

        # get events
        evt = dbio.fetch(dbname, 'events', *dtup)
        evtdict = {}
        evtdict['stops'] = evt['banked'].dropna()
        evtdict['pops'] = evt['popped'].dropna()
        evtdict['starts'] = evt['start inflating']
        if 'is_control' in evt.columns:
            evtdict['stops_free'] = evt.query('is_control == False')['banked'].dropna()
            evtdict['stops_control'] = evt.query('is_control == True')['banked'].dropna()
            evtdict['stops_rewarded'] = evt.query('trial_type != 4')['banked'].dropna()
            evtdict['stops_unrewarded'] = evt.query('trial_type == 4')['banked'].dropna()
        else:
            evtdict['stops_free'] = evtdict['stops']
            evtdict['stops_control'] = None
            evtdict['stops_rewarded'] = evtdict['stops']
            evtdict['stops_unrewarded'] = None

        if evtdict[event_labels[0]] is None:
            print "Dataset {} has no events of type {}".format(dtup, event_labels[0])
        elif evtdict[event_labels[1]] is None:
            print "Dataset {} has no events of type {}".format(dtup, event_labels[1])
        else:
            this_spectra, this_labels, taxis, faxis = avg_time_freq_arrays(lfp,
                                [evtdict[event_labels[0]], evtdict[event_labels[1]]],
                                Tpre, Tpost,
                                method='wav', normfun=normfun, freqs=freqs)

            spectra_list.append(this_spectra)
            labels_list.append(this_labels)

    spectra = np.concatenate(spectra_list)
    labels = np.concatenate(labels_list)

    return spectra, labels, taxis, faxis

def make_plot(contrast, taxis, faxis, **kwargs):
    """
    Given a time-frequence contrast matrix and time and frequency axes,
    make a plot.
    """
    dfcontrast = pd.DataFrame(contrast, index=taxis, columns=faxis)

    dbvals = 10 * np.log10(contrast.data)
    if 'clim' in kwargs:
        color_lims = kwargs['clim']
        kwargs.pop('clim', None)
    else:
        color_lims = (np.amin(dbvals), np.amax(dbvals))
    fig = tf.plot_time_frequency(dfcontrast, clim=color_lims, **kwargs)

    return dfcontrast, fig

def significant_time_frequency(dbname, tuplist, event_names, Tpre, Tpost,
                               freqs, thresh, normfun=None, niter=1000,
                               **kwargs):
    spectra, labels, taxis, faxis = get_spectra_and_labels(dbname, tuplist,
                                    event_names, Tpre, Tpost, freqs, normfun)

    contrast, taxis = trials_to_clusters(spectra, labels, thresh, taxis, Tpre, Tpost, niter=niter)

    dfcontrast, fig = make_plot(contrast, taxis, faxis, doplot=True, **kwargs)

    return dfcontrast, fig
