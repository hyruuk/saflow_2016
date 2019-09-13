from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne.io import read_raw_ctf, read_raw_fif
from matplotlib.pyplot import plot as plt
from matplotlib.pyplot import close as closefig
import mne
import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
import time



filepath = '/home/karim/DATA/DATAmeg_gradCPT/20190820/SA11_SAflow-yharel_20190820_01.ds'

report = mne.Report(verbose=True)

raw_data = read_raw_ctf(filepath, preload=True)
picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude='bads')

fig = raw_data.plot(show=False);
report.add_figs_to_section(fig, captions='Time series', section='Raw data')
closefig(fig)

fig = raw_data.plot_psd(average=False, picks=picks, show=False);
report.add_figs_to_section(fig, captions='PSD', section='Raw data')
closefig(fig)


## Filtering
high_cutoff = 200
low_cutoff = 0.5
raw_data.filter(low_cutoff, high_cutoff, fir_design="firwin")
raw_data.notch_filter(np.arange(60, high_cutoff+1, 60), picks=picks, filter_length='auto',phase='zero', fir_design="firwin")

fig = raw_data.plot_psd(average=False, picks=picks, show=False);
report.add_figs_to_section(fig, captions='PSD', section='Filtered data')
closefig(fig)

## ICA
ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
fig = ica.plot_sources(raw_data, show=False);
report.add_figs_to_section(fig, captions='Independent Components', section='ICA')
closefig(fig)
fmax = 40. ## correlation threshold for ICA components (maybe increase to 40. ?)

## FIND ECG COMPONENTS
ecg_epochs = create_ecg_epochs(raw_data, ch_name='EEG059')
ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name='EEG059')

fig = ica.plot_scores(ecg_scores, ecg_inds, show=False);
report.add_figs_to_section(fig, captions='Correlation with ECG (EEG059)', section='ICA - ECG')
closefig(fig)

fig = list()
fig = ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': fmax}, image_args={'sigma': 1.}, show=False);
for figure in fig:
    report.add_figs_to_section(figure, captions='Detected components', section='ICA - ECG')
    closefig(figure)


## FIND EOG COMPONENTS
eog_epochs = create_eog_epochs(raw_data, ch_name='EEG057')
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name='EEG057')
fig = ica.plot_scores(eog_scores, eog_inds, show=False);
report.add_figs_to_section(fig, captions='Correlation with EOG (EEG057)', section='ICA - EOG')
closefig(fig)

fig = list()
fig = ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': fmax}, image_args={'sigma': 1.}, show=False);
for figure in fig:
    report.add_figs_to_section(figure, captions='Detected components', section='ICA - EOG')
    closefig(figure)

## EXCLUDE COMPONENTS
ica.exclude = ecg_inds
ica.apply(raw_data)
ica.exclude = eog_inds
ica.apply(raw_data)

fig = raw_data.plot(show=False); # Plot the clean signal.
report.add_figs_to_section(fig, captions='After filtering + ICA', section='Raw data')
closefig(fig)

## SAVE PREPROCESSED FILE
time.sleep(30)
raw_data.save(filepath + '_preprocessed_raw.fif.gz', overwrite=True)
report.save('report_basic.html')
