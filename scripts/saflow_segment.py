##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow_utils import segment_files, get_SAflow_bids
from saflow_params import FOLDERPATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST


if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            preproc_path, preproc_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='preproc_raw', cond=None)
            epoched_path, epoched_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='epo', cond=None)
            epochs_clean = segment_files(preproc_filename)
            epochs_clean.save(epoched_filename)
