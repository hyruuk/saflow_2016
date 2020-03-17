import mne
import numpy as np
from utils import get_SAflow_bids
from neuro import compute_TFR
from saflow_params import FOLDERPATH, IMG_DIR, FREQS, SUBJ_LIST, BLOCS_LIST
from brainpipe import feature
from scipy.io import savemat


### OPEN SEGMENTED FILES AND COMPUTE PSDS
if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='epo', cond=None)
            data = mne.read_epochs(SAflow_bidspath)
            TFR = compute_TFR(data)
            TFR_bidsname, TFR_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='TFR', cond=None)
            TFR.save(TFR_bidspath)
