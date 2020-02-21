##### OPEN PREPROC FILES AND SEGMENT THEM
from utils import get_SAflow_bids
from neuro import segment_files
from saflow_params import FOLDERPATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST
import pickle


if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            preproc_path, preproc_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='preproc_raw', cond=None)
            epoched_path, epoched_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='1600epo', cond=None)
            ARlog_path, ARlog_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='ARlog', cond=None)
            epochs_clean, AR_log = segment_files(preproc_filename, tmin=-0.4, tmax=1.2)
            epochs_clean.save(epoched_filename)
            with open(ARlog_filename, 'wb') as fp:
                pickle.dump(AR_log, fp)
