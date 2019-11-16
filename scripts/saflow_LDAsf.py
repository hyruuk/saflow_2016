from scipy.io import loadmat, savemat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlneurotools.ml import classification
from pathlib import Path


SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
BLOCS_LIST = ['1','2','3', '4', '5', '6']
#FREQ = 'alpha'
FREQ_BANDS = ['theta','alpha','beta','gamma1','gamma2','gamma3']
#CHAN = 0
DPATH = '/storage/Yann/saflow_DATA/saflow_PSD'
RESULTS_PATH = '/storage/Yann/saflow_DATA/LDA_results'


### ML single subject classification of IN vs OUT epochs
# - single-features
# - CV k-fold (maybe 10 ?)
# - LDA, RF, kNN ?
def prepare_data(DPATH, SUBJ_LIST, BLOCS_LIST, FREQ, CHAN):
    prepared_data = []
    prepared_y = []
    prepared_groups = []
    i_group = 0
    for zone in ['IN','OUT']:
        for subj in SUBJ_LIST:
            for bloc in BLOCS_LIST:
                # Open datafile (n_chan X n_epochs matrix)
                filename = 'SA{}_{}_{}_{}.mat'.format(subj, bloc, zone, FREQ)
                fpath = DPATH + '/' + filename
                data = loadmat(fpath)['PSD']

                # Extract the features
                prepared_data.append(data)

                # Create y label
                if zone == 'IN':
                    trials_y = np.array(data.shape[1]*[1])
                else:
                    trials_y = np.array(data.shape[1]*[0])
                prepared_y.append(trials_y)

                # Create groups
                trials_group = np.array(data.shape[1]*[i_group])
                prepared_groups.append(trials_group)

            i_group += 1

    X = np.concatenate(prepared_data, axis=1)
    y = np.concatenate(prepared_y, axis=0)
    groups = np.concatenate(prepared_groups, axis=0)

    if CHAN != None:
        X = X[CHAN,:]

    X = X.reshape(-1,1)
    return X, y, groups

def combine_features(Xs):
    X = np.concatenate(Xs, axis=1)
    return X

def classif_and_save(X,y,groups, FREQ, CHAN):
    savepath = '{}/LDA_{}_{}.mat'.format(RESULTS_PATH, FREQ, CHAN)
    if Path(savepath).is_file():
        print(savepath + 'already exists.')
        return

    cv = StratifiedShuffleSplit(10)
    clf = LinearDiscriminantAnalysis()
    save = classification(clf, cv, X, y, groups=groups, perm=1000, n_jobs=4)

    print('Done')
    print('DA : ' + str(save['acc_score']))
    print('p value : ' + str(save['acc_pvalue']))
    savemat(savepath, save)



for FREQ in FREQ_BANDS:
    for CHAN in range(270):
        X, y, groups = prepare_data(DPATH, SUBJ_LIST, BLOCS_LIST, FREQ, CHAN)
        print('Computing chan {} in {} band.')
        classif_and_save(X,y,groups, FREQ, CHAN)



#### Résultat on veut : elec * freq X trials(IN+OUT) = 1890 X N_trials_tot
