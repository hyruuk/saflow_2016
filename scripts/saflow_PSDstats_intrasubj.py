from saflow_utils import array_topoplot, create_pval_mask, load_PSD_data
from saflow_params import FOLDERPATH, IMG_DIR, FREQS_NAMES, SUBJ_LIST, BLOCS_LIST, ZONE_CONDS, ZONE2575_CONDS
from scipy.io import loadmat
import mne
from hytools.meg_utils import get_ch_pos
import numpy as np
from mlneurotools.stats import ttest_perm
import matplotlib.pyplot as plt
import argparse


FOLDERPATH = '/storage/Yann/saflow_DATA/saflow_bids/'
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'
FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
BLOCS_LIST = ['2','3', '4', '5', '6', '7']


### OPEN PSDS AND CREATE TOPOPLOTS

#### ALL SUBJ TOPOPLOT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--subject",
        default=None,
        type=int,
        help="Subject to compute",
    )

    args = parser.parse_args()

    # get ch x and y coordinates
    ch_file = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
    raw = mne.io.read_raw_ctf(ch_file, verbose=False)
    ch_xy = get_ch_pos(raw)
    raw.close()

    # load PSD data
    PSD_alldata = load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, ZONE2575_CONDS)

    # compute t_tests
    arg_subj = args.subject
    if arg_subj != None:
        s_idx = arg_subj
        subj = SUBJ_LIST[arg_subj]
        power_diff = []
        masks = []
        tvalues = []
        pvalues = []
        max_trials = np.min((PSD_alldata[0][s_idx].shape[2], PSD_alldata[1][s_idx].shape[2]))
        for i, freq in enumerate(FREQS_NAMES):

            tvals, pvals = ttest_perm(PSD_alldata[0][s_idx][i,:,:max_trials].T, PSD_alldata[1][s_idx][i,:,:max_trials].T, # cond1 = IN, cond2 = OUT
            n_perm=1001,
            n_jobs=6,
            correction='maxstat',
            paired=False,
            two_tailed=True)
            tvalues.append(tvals)
            pvalues.append(pvals)
            masks.append(create_pval_mask(pvals, alpha=0.05))

        # plot
        toplot = tvalues
        vmax = np.max(np.max(abs(np.asarray(toplot))))
        vmin = -vmax
        array_topoplot(toplot,
                        ch_xy,
                        showtitle=True,
                        titles=FREQS_NAMES,
                        savefig=True,
                        figpath=IMG_DIR + 'IN25vsOUT75_tvals_sub-{}_A05_maxstat.png'.format(subj),
                        vmin=vmin,
                        vmax=vmax,
                        cmap='coolwarm',
                        with_mask=True,
                        masks=masks,
                        show=False)
    else:
        for s_idx, subj in enumerate(SUBJ_LIST):
            power_diff = []
            masks = []
            tvalues = []
            pvalues = []
            max_trials = np.min((PSD_alldata[0][s_idx].shape[2], PSD_alldata[1][s_idx].shape[2]))
            for i, freq in enumerate(FREQS_NAMES):

                tvals, pvals = ttest_perm(PSD_alldata[0][s_idx][i,:,:max_trials].T, PSD_alldata[1][s_idx][i,:,:max_trials].T, # cond1 = IN, cond2 = OUT
                n_perm=1001,
                n_jobs=6,
                correction=None,
                paired=False,
                two_tailed=True)
                tvalues.append(tvals)
                pvalues.append(pvals)
                masks.append(create_pval_mask(pvals, alpha=0.05))

            # plot
            toplot = tvalues
            vmax = np.max(np.max(abs(np.asarray(toplot))))
            vmin = -vmax
            array_topoplot(toplot,
                            ch_xy,
                            showtitle=True,
                            titles=FREQS_NAMES,
                            savefig=True,
                            figpath=IMG_DIR + 'IN25vsOUT75_tvals_sub-{}_A05_uncorr.png'.format(subj),
                            vmin=vmin,
                            vmax=vmax,
                            cmap='coolwarm',
                            with_mask=True,
                            masks=masks,
                            show=False)
