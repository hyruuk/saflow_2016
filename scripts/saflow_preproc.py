from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne.io import read_raw_ctf, read_raw_fif
from matplotlib.pyplot import plot as plt
from matplotlib.pyplot import close as closefig
import mne
import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
import os.path as op
import time

BIDS_PATH = '/storage/Yann/saflow_DATA/saflow_bids'
SUBJ_LIST = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
BLOCS_LIST = ['1', '2', '3', '4', '5', '6', '7', '8']
REPORTS_PATH = op.join(BIDS_PATH, 'preproc_reports')

# create report path
try:
	os.mkdir(REPORTS_PATH)
except:
	print('Report path already exists.')


def find_rawfile(subj, bloc, BIDS_PATH):
	filepath = '/sub-{}/ses-recording/meg/'.format(subj)
	files = os.listdir(BIDS_PATH + filepath)
	for file in files:
		if file[-8] == bloc:
			filename = file
	return filepath, filename


def saflow_preproc(filepath, savepath, reportpath):
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
	fig = raw_data.plot_psd(average=False, picks=picks, fmax=120, show=False);
	report.add_figs_to_section(fig, captions='PSD', section='Filtered data')
	closefig(fig)
	## ICA
	ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
	fig = ica.plot_sources(raw_data, show=False);
	report.add_figs_to_section(fig, captions='Independent Components', section='ICA')
	closefig(fig)
	## FIND ECG COMPONENTS
	ecg_threshold = 0.50
	ecg_epochs = create_ecg_epochs(raw_data, ch_name='EEG059')
	ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name='EEG059', method='ctps', threshold=ecg_threshold)
	fig = ica.plot_scores(ecg_scores, ecg_inds, show=False);
	report.add_figs_to_section(fig, captions='Correlation with ECG (EEG059)', section='ICA - ECG')
	closefig(fig)
	fig = list()
	fig = ica.plot_properties(ecg_epochs, picks=ecg_inds, image_args={'sigma': 1.}, show=False);
	for i, figure in enumerate(fig):
		report.add_figs_to_section(figure, captions='Detected component ' + str(i), section='ICA - ECG')
		closefig(figure)
	## FIND EOG COMPONENTS
	eog_threshold = 4
	eog_epochs = create_eog_epochs(raw_data, ch_name='EEG057')
	eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name='EEG057', threshold=eog_threshold)
	fig = ica.plot_scores(eog_scores, eog_inds, show=False);
	report.add_figs_to_section(fig, captions='Correlation with EOG (EEG057)', section='ICA - EOG')
	closefig(fig)
	fig = list()
	fig = ica.plot_properties(eog_epochs, picks=eog_inds, image_args={'sigma': 1.}, show=False);
	for i, figure in enumerate(fig):
		report.add_figs_to_section(figure, captions='Detected component ' + str(i), section='ICA - EOG')
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
	report.save(reportpath, open_browser=False);
	try:
		raw_data.save(savepath, overwrite=False)
	except:
		print('File already exists')

if __name__ == "__main__":
	for subj in SUBJ_LIST:
		for bloc in BLOCS_LIST:
			filepath, filename = find_rawfile(subj, bloc, BIDS_PATH)
			save_pattern =  op.join(BIDS_PATH + filepath, filename[:-3] + '_preproc_raw.fif.gz')
			report_pattern = op.join(REPORTS_PATH, filename[:-3] + '_preproc_report.html')
			full_filepath = BIDS_PATH + filepath + filename
			saflow_preproc(full_filepath, save_pattern, report_pattern)
