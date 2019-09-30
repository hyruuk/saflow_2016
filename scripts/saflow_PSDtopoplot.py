from saflow_utils import array_topoplot
from scipy.io import loadmat
import mne
from hytools.meg_utils import get_ch_pos
import numpy as np
from mlneurotools.stats import relative_perm, ttest_perm



FOLDERPATH = '/storage/Yann/saflow_DATA/'
PSDS_DIR = FOLDERPATH + 'saflow_PSD/'
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'


FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
#SUBJ_LIST = ['14']
#BLOCS_LIST = ['5', '6']
BLOCS_LIST = ['1','2','3', '4', '5', '6']



alpha = 0.05

    
### OPEN PSDS AND CREATE TOPOPLOTS

#### ALL SUBJ TOPOPLOT
if __name__ == "__main__":
    ch_file = '/home/karim/pCloudDrive/science/saflow/DATAmeg_gradCPT/20190411/SA04_SAflow-yharel_20190411_02.ds'
    raw = mne.io.read_raw_ctf(ch_file)
    ch_xy = get_ch_pos(raw)
    toplot = []
    toplot_pval = []
    toplot_masks = []
    for i, freq_name in enumerate(FREQS_NAMES):
        alldata = []
        for zone in ['IN', 'OUT']:
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
                subjdata = np.mean(subjdata, axis=1) # average blocs
                zonedata.append(subjdata)
            alldata.append(np.asarray(zonedata))
        alldata = np.asarray(alldata)
        tvals, pvals = ttest_perm(np.asarray(alldata[0]), np.asarray(alldata[1]), n_perm=1001, two_tailed=False, correction=None, paired=True, n_jobs=4)
        mask = pvals < alpha
        print(np.min(pvals))
        alldata_avg = np.mean(alldata, axis=1) # average subjects
        toplot.append((alldata_avg[0] - alldata_avg[1])/alldata_avg[1]) ### IN - OUT / OUT
        toplot_pval.append(pvals)
        toplot_masks.append(mask)




    toplot = [i * 100 for i in toplot]
    vmax = np.max(np.max(abs(np.asarray(toplot))))
    vmin = - vmax
    array_topoplot(toplot, ch_xy, showtitle=True, titles=FREQS_NAMES, savefig=True, figpath=IMG_DIR + 'INvsOUT_PSD_autoreject_05uncorr.png', vmin=vmin, vmax=vmax, cmap='seismic', with_mask=True, masks=toplot_masks)

    #### multiplier valeurs par 100
    #### gérer la colorbar