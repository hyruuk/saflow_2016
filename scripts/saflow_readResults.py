from scipy.io import loadmat
import numpy as np
from mne.io import read_raw_ctf
from saflow_utils import array_topoplot, get_ch_pos, create_pval_mask
from saflow_params import IMG_DIR
import itertools
from mlneurotools.stats import compute_pval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--condition",
    default='LDA_L1SO_2575',
    type=str,
    help="Defines classifier and condition set in the form CLASS_XVAL_SPLIT(optionnal)",
)
parser.add_argument(
    "-a",
    "--alpha",
    default=0.05,
    type=float,
    help="Desired alpha threshold",
)


args = parser.parse_args()

CONDITION = args.condition
ALPHA = args.alpha

RESULTS_PATH = '/storage/Yann/saflow_DATA/results/single_feat/' + CONDITION
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
        pval_mask = create_pval_mask(corrected_pval, alpha=ALPHA)
        all_acc.append(np.array(freq_acc).squeeze())
        all_pval.append(np.array(freq_pval).squeeze())
        all_masks.append(pval_mask)



    toplot = all_acc
    vmax = np.max(np.max(np.asarray(toplot)))
    vmin = np.min(np.min(np.asarray(toplot)))
    array_topoplot(toplot, ch_xy, showtitle=True, titles=FREQ_BANDS, savefig=True, figpath=IMG_DIR + '{}_A{}.png'.format(CONDITION, str(ALPHA)[2:]) ,vmin=vmin, vmax=vmax, with_mask=True, masks=all_masks)
