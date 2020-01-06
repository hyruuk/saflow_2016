from scipy.io import loadmat
import numpy as np
from mne.io import read_raw_ctf
from saflow_utils import array_topoplot, get_ch_pos, create_pval_mask
from saflow_params import IMG_DIR, SUBJ_LIST

RESULTS_PATH = '/storage/Yann/saflow_DATA/saflow_bids/ML_results/single_feat/LDAsf_intrasubj_K10_2575'
FREQ_BANDS = ['theta','alpha','lobeta', 'hibeta', 'gamma1','gamma2','gamma3']
MODEL = 'LDA'
NPERM = 1001
CV = '10FOLD'
COND = 2575


if __name__ == "__main__":
    ##### obtain ch_pos
    filename = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
    raw = read_raw_ctf(filename, preload=False, verbose=False)
    ch_xy = get_ch_pos(raw)
    raw.close()
    for SUBJ in SUBJ_LIST:
        all_acc = []
        all_pval = []
        all_masks = []
        for FREQ in FREQ_BANDS:
            freq_acc = []
            freq_pval = []
            for CHAN in range(270):
                savepath = '{}/classif_sub-{}_{}_{}.mat'.format(RESULTS_PATH, SUBJ, FREQ, CHAN)
                data_acc = loadmat(savepath)['acc_score']
                #print(loadmat(savepath)['acc_pscores'].shape)

                data_pval = loadmat(savepath)['acc_pvalue']
                freq_acc.append(data_acc)
                freq_pval.append(data_pval)
            pval_mask = create_pval_mask(np.array(freq_pval).squeeze(), alpha=0.05)
            all_acc.append(np.array(freq_acc).squeeze())
            all_pval.append(np.array(freq_pval).squeeze())
            all_masks.append(pval_mask)

        toplot = all_acc
        vmax = np.max(np.max(np.asarray(toplot)))
        vmin = np.min(np.min(np.asarray(toplot)))
        array_topoplot(toplot, ch_xy, show=False, showtitle=True, titles=FREQ_BANDS, savefig=True, figpath=IMG_DIR + '{}_{}perm_{}_sub-{}_{}.png'.format(MODEL, NPERM, CV, SUBJ, COND) ,vmin=vmin, vmax=vmax, with_mask=True, masks=all_masks)
