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


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--features",
    default='PSD_VTC',
    type=int,
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
    for i, cond in enumerate(PSD_dloadloadload_PSD_dataload_PSD_data_PSD_dataload_PSD_data_PSD_dataata):
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

def classif_singlefeat(X,y,groups, FREQ, CHAN):
    cv = LeaveOneGroupOut()
    clf = LinearDiscriminantAnalysis()
    results = classification(clf, cv, X, y, groups=groups, perm=1000, n_jobs=-1)
    print('Done')
    print('DA : ' + str(results['acc_score']))
    print('p value : ' + str(results['acc_pvalue']))
    return results

def LDAsf(CHAN, FREQ, FEAT_FILE, RESULTS_PATH):
    with open(FEAT_FILE, 'rb') as fp:
        PSD_data = pickle.load(fp)
    X, y, groups = prepare_data(PSD_data, FREQ, CHAN)
    print('Computing chan {} in {} band :'.format(CHAN, FREQS_NAMES[FREQ]))
    results = classif_singlefeat(X,y,groups, FREQ, CHAN)
    SAVEPATH = '{}/classif_{}_{}.mat'.format(RESULTS_PATH, FREQS_NAMES[FREQ], CHAN)
    savemat(SAVEPATH, results)

FEAT_PATH = '../features/'
FEAT_FILE = FEAT_PATH + args.features
RESULTS_PATH = '../results/single_feat/LDA_L1SO' + args.features

if __name__ == "__main__":
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
        print('Results folder created at : {}'.format(RESULTS_PATH))
    else:
        print('{} already exists.'.format(RESULTS_PATH))

    Parallel(n_jobs=-1)(
        delayed(LDAsf)(CHAN, FREQ, FEAT_FILE, RESULTS_PATH) for CHAN, FREQ in product(range(270), range(len(FREQS_NAMES)))
    )
