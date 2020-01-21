from scipy.io import loadmat
import numpy as np
from mne.io import read_raw_ctf
from saflow_utils import array_topoplot, get_ch_pos, create_pval_mask
from saflow_params import IMG_DIR
import itertools
from mlneurotools.stats import compute_pval

RESULTS_PATH = '/storage/Yann/saflow_DATA/saflow_bids/ML_results/single_feat/LDA_L1SO_2575'
FREQ_BANDS = ['theta','alpha','lobeta', 'hibeta', 'gamma1','gamma2','gamma3']

if __name__ == "__main__":
    ##### obtain ch_pos
    filename = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
    raw = read_raw_ctf(filename, preload=False, verbose=False)
    ch_xy = get_ch_pos(raw)
    raw.close()
    all_acc = []
    all_pval = []
    all_masks = []
    for FREQ in FREQ_BANDS:
        freq_acc = []
        freq_pval = []
        freq_perms_acc = []
        for CHAN in range(270):
            savepath = '{}/classif_{}_{}.mat'.format(RESULTS_PATH, FREQ, CHAN)
            data_acc = loadmat(savepath)['acc_score']
            data_pval = loadmat(savepath)['acc_pvalue']
            data_perms_acc = loadmat(savepath)['acc_pscores']
            freq_acc.append(data_acc)
            freq_pval.append(data_pval)
            freq_perms_acc.append(data_perms_acc)
        freq_perms = list(itertools.chain.from_iterable(freq_perms_acc))
        corrected_pval = []
        for acc in freq_acc:
            corrected_pval.append(compute_pval(acc, freq_perms))
        corrected_pval = np.array(corrected_pval)
        pval_mask = create_pval_mask(corrected_pval, alpha=0.05)
        all_acc.append(np.array(freq_acc).squeeze())
        all_pval.append(np.array(freq_pval).squeeze())
        all_masks.append(pval_mask)



    toplot = all_acc
    vmax = np.max(np.max(np.asarray(toplot)))
    vmin = np.min(np.min(np.asarray(toplot)))
    array_topoplot(toplot, ch_xy, showtitle=True, titles=FREQ_BANDS, savefig=True, figpath=IMG_DIR + 'LDA_L1SO_2575_A05.png' ,vmin=vmin, vmax=vmax, with_mask=True, masks=all_masks)
