##### OPEN PREPROC FILES AND SEGMENT THEM
from neuro import split_PSD_data
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH
from scipy.io import savemat
import pickle

if __name__ == "__main__":
    PSD_alldata = split_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, by='VTC', lobound=None, hibound=None, stage='PSD', filt_order=3, filt_cutoff=0.1)
    PSD_dict = {'PSD_alldata': PSD_alldata}

    with open(FEAT_PATH + 'PSD_VTCo3c01', 'wb') as fp:
        pickle.dump(PSD_alldata, fp)
