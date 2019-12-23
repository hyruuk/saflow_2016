##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow_utils import segment_files_INvsOUT, get_SAflow_bids
from saflow_params import FOLDERPATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST

if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            INepochs, OUTepochs = segment_files_INvsOUT(LOGS_DIR, subj, bloc, lobound=0.25, hibound=0.75)
            INpath, INfilename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond='IN25')
            OUTpath, OUTfilename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond='OUT75')
            INepochs.save(INfilename)
            OUTepochs.save(OUTfilename)
