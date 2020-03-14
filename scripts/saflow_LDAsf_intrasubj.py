from scipy.io import loadmat, savemat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from pathlib import Path
import argparse
import os
from neuro import load_PSD_data
from utils import get_SAflow_bids
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, ZONE2575_CONDS, ZONE_CONDS
from joblib import Parallel, delayed
from itertools import product
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--features",
    default='PSD_VTC',
    type=str,
    help="Channels to compute",
)

args = parser.parse_args()


### ML single subject classification of IN vs OUT epochs
# - single-features
# - CV k-fold (maybe 10 ?)
# - LDA, RF, kNN ?
def prepare_data(PSD_data, SUBJ, FREQ, CHAN=None):
    '''
    Returns X, y and groups arrays from SAflow data for sklearn classification.
    FOLDERPATH is the base BIDS path
    SUBJ_LIST and BLOCS_LIST are lists of strings
    FREQ is an integer
    CHAN is an int or a list of int
    '''
    new_PSD_data = [[]]*len(PSD_data)
    # retain desired CHAN(s)
    for i, cond in enumerate(PSD_data):
        if CHAN != None:
            new_PSD_data[i] = PSD_data[i][SUBJ][FREQ,CHAN,:]
        else:
            new_PSD_data[i] = PSD_data[i][SUBJ][FREQ,:,:]
    X_list = []
    y_list = []
    groups_list = []
    for i, cond in enumerate(new_PSD_data):
        X_list.append(cond)
        if i == 0:
            y_list.append(np.zeros(len(cond)))
        elif i == 1:
            y_list.append(np.ones(len(cond)))
    X = np.concatenate((X_list), axis=0).reshape(-1, 1)
    y = np.concatenate((y_list), axis=0)

    return X, y

def classif_intrasubj(X,y, FREQ, CHAN, SAVEPATH):
    if Path(SAVEPATH).is_file():
        print(SAVEPATH + ' already exists.')
        return
    cv = ShuffleSplit(test_size=0.1, n_splits=10)
    #cv = LeaveOneGroupOut()sur une liste ?
    clf = LinearDiscriminantAnalysis()
    print(y.shape)
    results = classification(clf, cv, X.reshape(-1, 1), y, groups=None, perm=1001, n_jobs=-1)
    print('Done')
    print('DA : ' + str(save['acc_score']))
    print('p value : ' + str(save['acc_pvalue']))
    return results

def LDAsf(SUBJ, CHAN, FREQ, FEAT_FILE, RESULTS_PATH):
    with open(FEAT_FILE, 'rb') as fp:
        PSD_data = pickle.load(fp)
    X, y = prepare_data(PSD_data, SUBJ, FREQ, CHAN)
    print('Computing chan {} in {} band :'.format(CHAN, FREQS_NAMES[FREQ]))
    SAVEPATH = '{}/classif_sub-{}_{}_{}.mat'.format(RESULTS_PATH, SUBJ_LIST[SUBJ], FREQS_NAMES[FREQ], CHAN)
    results = classif_intrasubj(X,y,FREQ, CHAN, SAVEPATH)
    savemat(SAVEPATH, results)

FEAT_PATH = '../features/'
FEAT_FILE = FEAT_PATH + args.features
RESULTS_PATH = '../results/single_feat/LDA_singlesubj_L1SO_' + args.features

if __name__ == "__main__":
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
        print('Results folder created at : {}'.format(RESULTS_PATH))
    else:
        print('{} already exists.'.format(RESULTS_PATH))

    Parallel(n_jobs=-1)(
        delayed(LDAsf)(SUBJ, CHAN, FREQ, FEAT_FILE, RESULTS_PATH) for CHAN, FREQ, SUBJ in product(range(270), range(len(FREQS_NAMES)), range(len(SUBJ_LIST)))
    )

#### Résultat on veut : elec * freq X trials(IN+OUT) = 1890 X N_trials_tot
