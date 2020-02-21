from saflow_params import FOLDERPATH, IMG_DIR, FREQS_NAMES, SUBJ_LIST, BLOCS_LIST, ZONE_CONDS, ZONE2575_CONDS
from scipy.io import loadmat
import mne
from hytools.meg_utils import get_ch_pos
import numpy as np
from mlneurotools.stats import ttest_perm
import matplotlib.pyplot as plt
from neuro import split_PSD_data, load_PSD_data
from utils import create_pval_mask, array_topoplot
from saflow_params import FOLDERPATH, IMG_DIR, FREQS_NAMES, SUBJ_LIST, BLOCS_LIST, FEAT_PATH
import pickle


ALPHA = 0.001

### OPEN PSDS AND CREATE TOPOPLOTS
#### ALL SUBJ TOPOPLOT
if __name__ == "__main__":
    # get ch x and y coordinates
    ch_file = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
    raw = mne.io.read_raw_ctf(ch_file, verbose=False)
    ch_xy = get_ch_pos(raw)
    raw.close()

    # load data and create pd Dataframe
    PSD_alldata = load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, time_avg=True)
    with open(FEAT_PATH + 'VTC', 'rb') as fp:
        VTC_alldata = pickle.load(fp)
    # PSD is [subj][bloc][freq][chan][trial]
    # VTC is [subj][bloc][trial]

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
    for chan, freq in enumerate(FREQS_NAMES):
        tvals, pvals = ttest_perm(PSD_alldata[0][chan,:,:], PSD_alldata[1][chan,:,:], # cond1 = IN, cond2 = OUT
        n_perm=0,
        n_jobs=6,
        correction='maxstat',
        paired=True,
        two_tailed=True)
        tvalues.append(tvals)
        pvalues.append(pvals)
        masks.append(create_pval_mask(pvals, alpha=ALPHA))

    # plot
    toplot = tvalues
    vmax = np.max(np.max(abs(np.asarray(toplot))))
    vmin = -vmax
    fig = array_topoplot(toplot,
                    ch_xy,
                    showtitle=True,
                    titles=FREQS_NAMES,
                    savefig=True,
                    figpath=IMG_DIR + 'INvsOUT1585_tvals_12subj_A{}_maxstat.png'.format(str(ALPHA)[2:]),
                    vmin=vmin,
                    vmax=vmax,
                    cmap='coolwarm',
                    with_mask=True,
                    masks=masks)
    plt.close(fig=fig)
