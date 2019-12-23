from saflow_utils import array_topoplot, create_pval_mask, load_PSD_data
from saflow_params import FOLDERPATH, IMG_DIR, FREQS_NAMES, SUBJ_LIST, BLOCS_LIST, ZONE_CONDS, ZONE2575_CONDS
from scipy.io import loadmat
import mne
from hytools.meg_utils import get_ch_pos
import numpy as np
from mlneurotools.stats import ttest_perm
import matplotlib.pyplot as plt



FOLDERPATH = '/storage/Yann/saflow_DATA/saflow_bids/'
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'
FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
BLOCS_LIST = ['2','3', '4', '5', '6', '7']


### OPEN PSDS AND CREATE TOPOPLOTS

#### ALL SUBJ TOPOPLOT
if __name__ == "__main__":
    # get ch x and y coordinates
    ch_file = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
    raw = mne.io.read_raw_ctf(ch_file, verbose=False)
    ch_xy = get_ch_pos(raw)
    raw.close()

    # load PSD data
    PSD_alldata = load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, ZONE2575_CONDS)

    # average across trials
    for cond in range(len(PSD_alldata)):
        for subj in range(len(PSD_alldata[0])):
            PSD_alldata[cond][subj] = np.mean(PSD_alldata[cond][subj], axis=2)
    PSD_alldata = np.array(PSD_alldata)

    # compute t_tests
    power_diff = []
    masks = []
    tvalues = []
    pvalues = []
    for i, freq in enumerate(FREQS_NAMES):
        tvals, pvals = ttest_perm(PSD_alldata[0][i,:,:], PSD_alldata[1][i,:,:], # cond1 = IN, cond2 = OUT
        n_perm=0,
        n_jobs=6,
        correction='maxstat',
        paired=True,
        two_tailed=True)
        tvalues.append(tvals)
        pvalues.append(pvals)
        masks.append(create_pval_mask(pvals, alpha=0.05))

    # plot
    toplot = tvalues
    vmax = np.max(np.max(abs(np.asarray(toplot))))
    vmin = -vmax
    fig = array_topoplot(toplot,
                    ch_xy,
                    showtitle=True,
                    titles=FREQS_NAMES,
                    savefig=False,
                    figpath=IMG_DIR + 'IN25vsOUT75_tvals_12subj_A05_maxstat.png',
                    vmin=vmin,
                    vmax=vmax,
                    cmap='coolwarm',
                    with_mask=True,
                    masks=masks)
    plt.close(fig=fig)
