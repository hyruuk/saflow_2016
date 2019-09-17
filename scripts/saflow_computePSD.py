from saflow_utils import compute_PSD
import mne
import os
import numpy as np
from scipy.io import savemat


folderpath = '/storage/Yann/saflow_DATA/'
PREPROC_PATH = folderpath + 'saflow_preproc/'
LOGS_DIR = "/home/karim/pCloudDrive/science/saflow/gradCPT/gradCPT_share_Mac_PC/gradCPT_share_Mac_PC/saflow_data/"
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'
EPOCHS_DIR = folderpath + 'saflow_epoched/'
PSDS_DIR = folderpath + 'saflow_PSD/'

FREQS = [ [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']

subj_list = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
blocs_list = ['1','2','3', '4', '5', '6']



if __name__ == "__main__":
	### OPEN SEGMENTED FILES AND COMPUTE PSDS
	for subj in subj_list:
	    for bloc in blocs_list:
	        for zone in ['IN', 'OUT']:
	            data = mne.read_epochs(EPOCHS_DIR + 'SA' + subj + '_' + bloc + '_' + zone + '_epo.fif.gz')
	            psds = compute_PSD(data, data.info['sfreq'], epochs_length = 0.8)
	            psds = np.mean(psds, axis=2)
	            for i, freq_name in enumerate(FREQS_NAMES):
	                PSD_save = psds[i]
	                savemat(PSDS_DIR + 'SA' + subj + '_' + bloc + '_' + zone + '_' + freq_name + '.mat', {'PSD': PSD_save})

