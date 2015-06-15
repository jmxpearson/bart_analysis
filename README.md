# bart_analysis
Analysis code and figure generation for intracranial recording experiment using the balloon analogue risk task (BART).

## Data

Data are available [here](TBD). The data are in a single HDF5 file with top-level directories

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
    - `plot_channel_traces_by_factor.ipynb`: Plots LFP power across frequency bands, broken out by risk level.
    - `test_integration_activity.ipynb`: Performs a median split on LFP activity near trial stop and start and tests for statistically significant increases in power over time.
    - `plot_LFP_channel_raster.ipynb`: Plot power for a single channel, single frequency band for each trial in a rastergram-type plot.


## Dependencies

### R
- glmnet
- caret
- ggplot2
- plyr
- reshape

### Python
- numpy, scipy, matplotlib, pandas
- rpy2 (for R &harr; Python connection)
- physutils (available [here](https://github.com/jmxpearson/physutils))
