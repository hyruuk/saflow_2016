import mne
import numpy as np
from saflow_utils import get_SAflow_bids, compute_PSD
from saflow_params import FOLDERPATH, IMG_DIR, FREQS, SUBJ_LIST, BLOCS_LIST, ZONE2575_CONDS
from brainpipe import feature
from scipy.io import savemat


### OPEN SEGMENTED FILES AND COMPUTE PSDS
if __name__ == "__main__":
    conditions = ZONE2575_CONDS
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            for zone in conditions:
                SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='epo', cond=zone)
                data = mne.read_epochs(SAflow_bidspath)
                psds = compute_PSD(data, data.info['sfreq'], epochs_length = 0.8, f=FREQS)
                psds = np.mean(psds, axis=2) # average PSDs in time across epochs
                PSD_bidsname, PSD_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='PSD', cond=zone)
                savemat(PSD_bidspath, {'PSD': psds})
