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
- __Behavior Analysis__:
- __Spike Analysis__:

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
