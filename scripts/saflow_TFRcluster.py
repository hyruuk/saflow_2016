from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH
from utils import get_SAflow_bids
import mne
import numpy as np
import os
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt

TFR1_fpath = FEAT_PATH + '1600TFR_IN50_c01.hd5'
TFR2_fpath = FEAT_PATH + '1600TFR_OUT50_c01.hd5'
cond1 = 'IN'
cond2 = 'OUT'

if __name__ == "__main__":
    #
    if not(os.path.isfile(TFR1_fpath)):
        alldata_cond1 = []
        alldata_cond2 = []
        for subj in SUBJ_LIST:
            subjdata_cond1 = []
            subjdata_cond2 = []
            for bloc in BLOCS_LIST:
                TFR_fname, TFR_fpath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='1600TFR')
                TFR = mne.time_frequency.read_tfrs(TFR_fpath)[0]
                # averaging trials
                subjdata_cond1.append(TFR[cond1].average().data)
                subjdata_cond2.append(TFR[cond2].average().data)
            subjdata_cond1 = np.array(subjdata_cond1)
            subjdata_cond2 = np.array(subjdata_cond2)
            # averaging blocs
            alldata_cond1.append(np.average(subjdata_cond1, axis=0))
            alldata_cond2.append(np.average(subjdata_cond2, axis=0))

        alldata_cond1 = np.array(alldata_cond1)
        TFR_cond1 = TFR.copy()
        TFR_cond1.data = alldata_cond1
        TFR_cond1.save(TFR1_fpath, overwrite=True)

        alldata_cond2 = np.array(alldata_cond2)
        TFR_cond2 = TFR.copy()
        TFR_cond2.data = alldata_cond2
        TFR_cond2.save(TFR2_fpath, overwrite=True)
        del TFR
    else:
        TFR_cond1 =  mne.time_frequency.read_tfrs(TFR1_fpath)[0]
        TFR_cond2 =  mne.time_frequency.read_tfrs(TFR2_fpath)[0]
    epochs_power_1 = TFR_cond1.data
    epochs_power_2 = TFR_cond2.data
    threshold = 6.0
    T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([epochs_power_1, epochs_power_2], n_permutations=100, threshold=threshold, tail=0)
    epochs_times = mne.read_epochs(get_SAflow_bids(FOLDERPATH, subj='04', run=3, stage='1600epo')[1])
    times = 1e3 * epochs_times.times


    plt.figure()
    # Create new stats image with only significant clusters
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    fmin = 2
    fmax = 150
    n_bins=30
    freqs = np.logspace(*np.log10([fmin, fmax]), num=n_bins)
    plt.imshow(T_obs,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', cmap='gray')
    plt.imshow(T_obs_plot,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', cmap='RdBu_r')

    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Induced power (%s)' % ch_name)

    plt.show()
