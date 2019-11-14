from scipy.io import loadmat, savemat
import numpy as np


SUBJ_LIST = ['04', '05']#, '06', '07', '08', '09', '10', '11', '12', '13']
BLOCS_LIST = ['1']#,'2','3', '4', '5', '6']
FREQ = 'alpha'
CHAN = [0,1,2,3]
DPATH = '/home/hyruuk/GitHub/saflow/Temp_DATA'

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

    X = np.concatenate(prepared_data, axis=1).T
    y = np.concatenate(prepared_y, axis=0)
    groups = np.concatenate(prepared_groups, axis=0)

    if CHAN != None:
        X = X[:,CHAN]

    return X, y, groups

X, y, groups = prepare_data(DPATH, SUBJ_LIST, BLOCS_LIST, FREQ, CHAN)
print(X.shape)



#### Résultat on veut : elec * freq X trials(IN+OUT) = 1890 X N_trials_tot
