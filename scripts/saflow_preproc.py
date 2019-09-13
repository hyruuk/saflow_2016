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


filepath = '/home/hyruuk/data_temp/inscapesMEG/P03_NEUROMOD-jboyle_20190529_01.ds'
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('multipage_pdf.pdf') as pdf:
    raw_data = read_raw_ctf(filepath, preload=True)
    picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude='bads')

    fig = raw_data.plot();
    pdf.savefig(fig)
    closefig(fig)

    fig = raw_data.plot_psd(average=False, picks=picks);
    pdf.savefig(fig)
    closefig(fig)


    ## Filtering
    high_cutoff = 200
    low_cutoff = 0.5
    raw_data.filter(low_cutoff, high_cutoff, fir_design="firwin")
    raw_data.notch_filter(np.arange(60, high_cutoff+1, 60), picks=picks, filter_length='auto',phase='zero', fir_design="firwin")

    fig = raw_data.plot_psd(average=False, picks=picks);
    pdf.savefig(fig)
    closefig(fig)

    ## ICA
    ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
    fig = ica.plot_sources(raw_data);
    pdf.savefig(fig)
    closefig(fig)
    fmax = 40. ## correlation threshold for ICA components (maybe increase to 40. ?)

    ## FIND ECG COMPONENTS
    ecg_epochs = create_ecg_epochs(raw_data, ch_name='EEG059')
    ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name='EEG059')

    fig = ica.plot_scores(ecg_scores, ecg_inds);
    pdf.savefig(fig)
    closefig(fig)

    fig = ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': fmax}, image_args={'sigma': 1.});
    pdf.savefig(fig)
    closefig(fig)

    ## FIND EOG COMPONENTS
    eog_epochs = create_eog_epochs(raw_data, ch_name='EEG057')
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name='EEG057')
    fig = ica.plot_scores(eog_scores, eog_inds);
    pdf.savefig(fig)
    closefig(fig)

    fig = ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': fmax}, image_args={'sigma': 1.});
    pdf.savefig(fig)
    closefig(fig)

    ## EXCLUDE COMPONENTS
    ica.exclude = ecg_inds
    ica.apply(raw_data)
    ica.exclude = eog_inds
    ica.apply(raw_data)

    fig = raw_data.plot(); # Plot the clean signal.
    pdf.savefig(fig)
    closefig(fig)

    ## SAVE PREPROCESSED FILE
    time.sleep(30)
    raw_data.save(filepath + '_preprocessed.fif.gz', overwrite=True)
