##### OPEN PREPROC FILES AND SEGMENT THEM
from neuro import split_PSD_data
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH
from scipy.io import savemat
import pickle

if __name__ == "__main__":
    PSD_alldata = split_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, by='VTC', lobound=0.15, hibound=0.85, stage='PSD')
    PSD_dict = {'PSD_alldata': PSD_alldata}

    with open(FEAT_PATH + 'PSD_VTC1585', 'wb') as fp:
        pickle.dump(PSD_alldata, fp)
