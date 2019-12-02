from saflow_utils import array_topoplot, create_pval_mask, load_PSD_data
from scipy.io import loadmat
import mne
from hytools.meg_utils import get_ch_pos
import numpy as np
from mlneurotools.stats import ttest_perm



FOLDERPATH = '/storage/Yann/saflow_DATA/'
PSDS_DIR = FOLDERPATH + 'saflow_PSD/'
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'
FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
BLOCS_LIST = ['1','2','3', '4', '5', '6']


### OPEN PSDS AND CREATE TOPOPLOTS

#### ALL SUBJ TOPOPLOT
if __name__ == "__main__":
    ch_file = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
    raw = mne.io.read_raw_ctf(ch_file)
    ch_xy = get_ch_pos(raw)
    PSD_alldata = load_PSD_data(SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, PSDS_DIR)
    power_diff = []
    masks = []
    tvalues = []
    pvalues = []

    # get PSDS averaged across trials (1 value per subj per sensor)
    for i, freq in enumerate(FREQS_NAMES):
        data_IN = []
        data_OUT = []
        for subj, _ in enumerate(SUBJ_LIST):
            dat = PSD_alldata[i][0][subj]
            data_IN.append(np.mean(dat, axis=1))
            dat = PSD_alldata[i][1][subj]
            data_OUT.append(np.mean(dat, axis=1))

        data_IN = np.array(data_IN)
        data_OUT = np.array(data_OUT)
        data_IN_avg = np.mean(data_IN, axis=0)
        data_OUT_avg = np.mean(data_OUT, axis=0)
        power_diff.append((data_IN_avg - data_OUT_avg)/data_OUT_avg) ### IN - OUT / OUT
        #tvals, pvals = ttest_perm(alldata[0].T, alldata[1].T, n_perm=1000, n_jobs=6)


        #tvalues.append(tvals)

        #pvalues.append(pvals)
        #print(pvals)
        #masks.append(create_pval_mask(pvals, alpha=0.1))

    toplot = power_diff
    vmax = np.max(np.max(abs(np.asarray(toplot))))
    vmin = -vmax
    array_topoplot(toplot, ch_xy, showtitle=True, titles=FREQS_NAMES, savefig=False, figpath=IMG_DIR + 'IN_vs_OUT_PSD_autoreject.png', vmin=vmin, vmax=vmax)#, with_mask=True, masks=masks)
