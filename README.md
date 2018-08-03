# bart_analysis
Analysis code and figure generation for intracranial recording experiment using the balloon analogue risk task (BART).

## Data

Data are available [here](http://dx.doi.org/10.5061/dryad.54tp8q5). The data are in a single HDF5 file with top-level directories

- `meta`: contains data for censoring, events, and valid lfp and spike channels. This is replicated in csv files in the `data` subdirectory of the current repository.

- `censor`: contains pairs of times between which data should be discarded based on artifacts.

- `spikes`: contains spike timestamps.

- `lfp`: contains local field potential (LFP) data.

Within the last three directories, the data are organized hierarchically by patient, dataset, channel, and unit (in the case of spikes) and by patient/dataset/unit for LFP and censoring.

## Analysis:
Analyses in the paper each correspond to one or more iPython notebooks. Other code is contained in scripts, as detailed below.

- __Behavior Analysis__: Behavioral analyses are presented in `behavior_analysis.ipynb`.

- __Spike Analysis__: Analysis of spike data is in `spike_analysis.ipynb`. This code calls several other pieces, including:
    - `prep_spike_data.py`: makes regressors for the firing rate model. This is only run if `reprep_data` is set to `True` at the top of the notebook. If the variable is `False`, the code expects to find `data/spkfitdata`.
    - `perform_glm_analysis.R`: runs the elastic net GLM on data for each unit. Outputs csv files of results in the form <code>&lt;<var>patient</var>&gt;.&lt;<var>dataset</var>&gt;.&lt;<var>channel</var>&gt;.&lt;<var>unit</var>&gt;.spkglmdata.csv</code>.
    - `analyze_spike_glm_output.R`: Takes csv files output by the glm analysis and produces tables of coefficients and plots.
    - `helpers.R` and `setup_env.R` contain helper functions, package loads, and relevant constants.

- __LFP Analysis__: Analysis of local field potential data is contained in the following notebooks:
    - `plot_channel_traces.ipynb`: Plots power in different frequency bands, by channel and normed across channels, aligned to trial start and stop, for a given dataset.
    - `plot_aggregate_channel_traces.ipynb`: Same as `plot_channel_traces`, but aggregates and averages over all datasets.
    - `plot_LFP_channel_raster.ipynb`: Plot power for a single channel, single frequency band for each trial in a rastergram-type plot.
    - `time_frequency_mean_across_channels.ipynb`: Perform time-frequency analysis on the mean across channels for a given dataset.
    - `contrast_motor_start_vs_stop.ipynb`: Perform a time-frequency analysis of the contrast between identical motor movements at trial start and trial stop.
    - `contrast_reward_vs_no_reward.ipynb`: Perform a time-frequency analysis of the contrasts between rewarded (stop) and unrewarded (pop and control) trials.
    - `LFP_classifier_data.ipynb`: Fit a sparse logistic classifier to LFP data for a single dataset. (A similar analysis can be performed in batch using `prep_classifier_data.py`, `perform_glm_analysis.R`, and `analyze_lfp_glm_output.R`.)

- __Old Analyses__: These notebooks are extra analyses not in the finished manuscript. They may be interesting but were largely null results and are not actively maintained.
    - `LFP_wavelet_classifier.ipynb`: Fit a sparse logistic classifier based on wavelet features instead of frequency bands. __Requires PyWavelets.__
    - `time_frequency_phase_analysis.ipynb` and `time_frequency_phase_amplitude_coupling.ipynb`: Phase and phase-amplitude coupling analysis for the LFP data.
    - `plot_channel_traces_by_risk.ipynb`: Plots LFP power across frequency bands, broken out by risk level.
    - `channel_correlation_clustering.ipynb`: Attempt to cluster channels together based on correlations among them.
    - `test_integration_activity.ipynb`: Performs a median split on LFP activity near trial stop and start and tests for statistically significant increases in power over time.


## Dependencies

### R
- glmnet
- caret
- ggplot2
- plyr
- reshape

### Python
- numpy, scipy, matplotlib, pandas
- seaborn (for plotting)
- rpy2 (for R &harr; Python connection)
- physutils (available [here](https://github.com/jmxpearson/physutils))
- PyWavelets (only for `LFP_wavelet_classifier.ipynb`)
