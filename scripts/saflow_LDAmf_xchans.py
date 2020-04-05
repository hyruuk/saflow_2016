from scipy.io import loadmat, savemat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from pathlib import Path
import argparse
import os
from utils import get_SAflow_bids
from neuro import load_PSD_data
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, FEAT_PATH
import pickle
import time
from joblib import Parallel, delayed
from itertools import product


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
def prepare_data(PSD_data, FREQ, CHAN=None):
    '''
    Returns X, y and groups arrays from SAflow data for sklearn classification.
    FREQ is an integer
    CHAN is an int or a list of int
    '''
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
            if CHAN != None:
                n_trials_subj = len(subj)
            else:
                n_trials_subj = subj.shape[1]
            if i == 0:
                y_list.append(np.zeros(n_trials_subj))
            elif i == 1:
                y_list.append(np.ones(n_trials_subj))
            groups_list.append(np.ones(n_trials_subj)*j)
    if CHAN != None:
        X = np.concatenate((X_list), axis=0).reshape(-1, 1)
    else:
        X = X_list[0]
        for i in range(1,len(X_list)):
            X = np.hstack((X, X_list[i]))
    y = np.concatenate((y_list), axis=0)
    groups = np.concatenate((groups_list), axis=0)
    return X, y, groups

def classif_singlefeat(X,y,groups):
    cv = LeaveOneGroupOut()
    clf = LinearDiscriminantAnalysis()
    results = classification(clf, cv, X, y, groups=groups, perm=1001, n_jobs=-1)
    print('Done')
    print('DA : ' + str(results['acc_score']))
    print('p value : ' + str(results['acc_pvalue']))
    return results

def LDAmf(FREQ, FEAT_FILE, RESULTS_PATH):
    with open(FEAT_FILE, 'rb') as fp:
        PSD_data = pickle.load(fp)
    X, y, groups = prepare_data(PSD_data, FREQ)
    print('Computing MF chans in {} band :'.format(FREQS_NAMES[FREQ]))
    results = classif_singlefeat(X.T,y,groups)
    print(results)
    SAVEPATH = '{}/classif_{}_mf.mat'.format(RESULTS_PATH, FREQS_NAMES[FREQ])
    savemat(SAVEPATH, results)

FEAT_PATH = '/home/hyruuk/pCloudDrive/science/saflow/features/'
FEAT_FILE = FEAT_PATH + args.features
RESULTS_PATH = '../results/multi_feat/LDAmf_L1SO_' + args.features


if __name__ == "__main__":
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
        print('Results folder created at : {}'.format(RESULTS_PATH))
    else:
        print('{} already exists.'.format(RESULTS_PATH))

    Parallel(n_jobs=-1)(
        delayed(LDAmf)(FREQ, FEAT_FILE, RESULTS_PATH) for FREQ in range(len(FREQS_NAMES))
    )
