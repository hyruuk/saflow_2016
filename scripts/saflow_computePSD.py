from saflow_utils import compute_PSD
import mne
import os
import numpy as np
from scipy.io import savemat


FOLDERPATH = '/storage/Yann/saflow_DATA/'
EPOCHS_DIR = FOLDERPATH + 'saflow_epoched/'
PSDS_DIR = FOLDERPATH + 'saflow_PSD/'

FREQS = [ [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']

#SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
#BLOCS_LIST = ['1','2','3', '4', '5', '6']
SUBJ_LIST = ['13']
BLOCS_LIST = ['5', '6']


if __name__ == "__main__":
	### OPEN SEGMENTED FILES AND COMPUTE PSDS
	for subj in SUBJ_LIST:
	    for bloc in BLOCS_LIST:
	        for zone in ['IN', 'OUT']:
	            data = mne.read_epochs(EPOCHS_DIR + 'SA' + subj + '_' + bloc + '_' + zone + '_epo.fif.gz')
	            psds = compute_PSD(data, data.info['sfreq'], epochs_length = 0.8, f=FREQS)
	            psds = np.mean(psds, axis=2)
	            for i, freq_name in enumerate(FREQS_NAMES):
	                PSD_save = psds[i]
	                savemat(PSDS_DIR + 'SA' + subj + '_' + bloc + '_' + zone + '_' + freq_name + '.mat', {'PSD': PSD_save})

