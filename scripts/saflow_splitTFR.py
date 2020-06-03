##### OPEN PREPROC FILES AND SEGMENT THEM
from neuro import split_TFR
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH
from utils import get_SAflow_bids
import mne

if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            TFR_fname, TFR_fpath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='1600TFR')
            TFR = split_TFR(TFR_fpath, subj, bloc, by='VTC', lobound=None, hibound=None, stage='1600TFR', filt_order=3, filt_cutoff=0.1)
            TFR[0].save(TFR_fpath, overwrite=True)
