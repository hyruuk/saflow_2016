##### OPEN PREPROC FILES AND SEGMENT THEM
from neuro import load_VTC_data
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH, LOGS_DIR
from scipy.io import savemat
import pickle

if __name__ == "__main__":
    VTC_alldata = load_VTC_data(FOLDERPATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST)

    #VTC_dict = {'PSD_alldata': PSD_alldata}

    with open(FEAT_PATH + 'VTC', 'wb') as fp:
        pickle.dump(VTC_alldata, fp)
