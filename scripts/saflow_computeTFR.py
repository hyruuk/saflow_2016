import mne
import numpy as np
from utils import get_SAflow_bids
from neuro import compute_TFR
from saflow_params import FOLDERPATH, IMG_DIR, FREQS, SUBJ_LIST, BLOCS_LIST
from brainpipe import feature
from scipy.io import savemat
import pdb

#SUBJ_LIST = ['04', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
#SUBJ_LIST = ['04']
### OPEN SEGMENTED FILES AND COMPUTE PSDS
if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='1600epo', cond=None)
            #print('baba{}'.format(SAflow_bidspath))
            #pdb.set_trace()
            data = mne.read_epochs(SAflow_bidspath)
            TFR = compute_TFR(data.copy().resample(500, npad='auto'), baseline=False) ##### NOTE THAT THE TFR IS COMPUTED ON RESAMPLED SIGNAL
            TFR_bidsname, TFR_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='1600TFRnobl', cond=None)
            TFR.save(TFR_bidspath, overwrite=True)
            del TFR
