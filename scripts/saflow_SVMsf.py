from scipy.io import loadmat, savemat
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from pathlib import Path
import argparse
import os
from saflow_utils import load_PSD_data, get_SAflow_bids
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--channel",
    default=0,
    type=int,
    help="Channels to compute",
)

args = parser.parse_args()

RESULTS_PATH = '/storage/Yann/saflow_DATA/saflow_bids/ML_results/single_feat/SVM_L1SO_RndSrch_10it'



### ML single subject classification of IN vs OUT epochs
# - single-features
# - CV k-fold (maybe 10 ?)
# - LDA, RF, kNN ?
def prepare_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FREQ, CHAN=None):
    '''
    Returns X, y and groups arrays from SAflow data for sklearn classification.
    FOLDERPATH is the base BIDS path
    SUBJ_LIST and BLOCS_LIST are lists of strings
    FREQ is an integer
    CHAN is an int or a list of int
    '''
    PSD_data = load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST)

    # retain desired CHAN(s)
    for i, cond in enumerate(PSD_data):
        for j, subj in enumerate(cond):
            if CHAN != None:
                PSD_data[i][j] = PSD_data[i][j][FREQ,CHAN,:]
            else:
                PSD_data[i][j] = PSD_data[i][j][FREQ,:,:]

    X_list = []
    y_list = []
    groups_list = []
    for i, cond in enumerate(PSD_data):
        for j, subj in enumerate(cond):
            X_list.append(subj)
            if i == 0:
                y_list.append(np.zeros(len(subj)))
            elif i == 1:
                y_list.append(np.ones(len(subj)))
            groups_list.append(np.ones(len(subj))*j)
    X = np.concatenate((X_list), axis=0).reshape(-1, 1)
    y = np.concatenate((y_list), axis=0)
    groups = np.concatenate((groups_list), axis=0)

    return X, y, groups

def classif_and_save(X,y,groups, FREQ, CHAN, SAVEPATH):
    if Path(SAVEPATH).is_file():
        print(SAVEPATH + ' already exists.')
        return

    cv = LeaveOneGroupOut()
    inner_cv = LeaveOneGroupOut()
    p_grid = {}
    p_grid['gamma']= [1e-3, 1e-4]
    p_grid['C']= [1, 10, 100, 1000]
    n_iter_search=10
    svc = SVC()
    Rnd_Srch = RandomizedSearchCV(svc, param_distributions=p_grid,
                                   n_iter=n_iter_search, cv=inner_cv)
    clf = make_pipeline(StandardScaler(),Rnd_Srch)
    save = classification(clf, cv, X, y, groups=groups, perm=100, n_jobs=8)
    print('Done')
    print('DA : ' + str(save['acc_score']))
    print('p value : ' + str(save['acc_pvalue']))
    savemat(SAVEPATH, save)

if __name__ == "__main__":
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
        print('Results folder created at : {}'.format(RESULTS_PATH))
    else:
        print('{} already exists.'.format(RESULTS_PATH))

    CHAN = args.channel
    for FREQ, FREQ_NAME in enumerate(FREQS_NAMES):
        for CHAN in range(270):
            X, y, groups = prepare_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FREQ, CHAN)
            print('Computing chan {} in {} band :'.format(CHAN, FREQ_NAME))
            SAVEPATH = '{}/classif_{}_{}.mat'.format(RESULTS_PATH, FREQ_NAME, CHAN)
            print(groups)
            classif_and_save(X,y,groups, FREQ, CHAN, SAVEPATH)

#### Résultat on veut : elec * freq X trials(IN+OUT) = 1890 X N_trials_tot
