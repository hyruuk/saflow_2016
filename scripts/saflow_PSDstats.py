from saflow_utils import array_topoplot, create_pval_mask
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
#SUBJ_LIST = ['14']
#BLOCS_LIST = ['5', '6']
BLOCS_LIST = ['1','2','3', '4', '5', '6']


### OPEN PSDS AND CREATE TOPOPLOTS

#### ALL SUBJ TOPOPLOT
if __name__ == "__main__":
    ch_file = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
    raw = mne.io.read_raw_ctf(ch_file)
    ch_xy = get_ch_pos(raw)
    toplot = []
    tvalues = []
    pvalues = []
    masks = []
    for i, freq_name in enumerate(FREQS_NAMES):
        alldata = []
        for zone in ['IN', 'OUT']: # 0 = IN, 1 = OUT
            zonedata = []
            for subj in SUBJ_LIST:
                subjdata = []
                for bloc in BLOCS_LIST:
                    mat = loadmat(PSDS_DIR + 'SA' + subj + '_' + bloc + '_' + zone + '_' + freq_name + '.mat')
                    data = mat['PSD']
                    if subjdata == []:
                        subjdata = data
                    else:
                        subjdata = np.hstack((subjdata, data))
                #subjdata = np.mean(subjdata, axis=1) # average blocs
                if zonedata == []:
                    zonedata = subjdata
                else:
                    zonedata = np.hstack((zonedata, subjdata))
            alldata.append(np.asarray(zonedata)[:,:6500])
        #alldata = np.array(alldata)
        print(alldata[1].shape)
        #0/0
        #alldata_avg = np.mean(alldata, axis=1) # average subjects
        alldata_avg = []
        for alldat in alldata:
            alldata_avg.append(np.mean(alldat, axis=1))

        toplot.append((alldata_avg[0] - alldata_avg[1])/alldata_avg[1]) ### IN - OUT / OUT
        tvals, pvals = ttest_perm(alldata[0].T, alldata[1].T, n_perm=1000, n_jobs=6)

        tvalues.append(tvals)
        pvalues.append(pvals)
        print(pvals)
        masks.append(create_pval_mask(pvals, alpha=0.1))




    ## Plot functions
    # Create masks with pvalues

    vmax = np.max(np.max(abs(np.asarray(tvalues))))
    vmin = -vmax
    array_topoplot(tvalues, ch_xy, showtitle=True, titles=FREQS_NAMES, savefig=False, figpath=IMG_DIR + 'IN_vs_OUT_PSD_autoreject.png', vmin=vmin, vmax=vmax, with_mask=True, masks=masks)
