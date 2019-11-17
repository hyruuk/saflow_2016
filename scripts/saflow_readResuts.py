from scipy.io import loadmat
from hytools.meg_utils import array_topoplot, get_ch_pos
import numpy as np
from mne.io import read_raw_ctf

RESULTS_PATH = '/storage/Yann/saflow_DATA/LDA_results'
FREQ_BANDS = ['theta','alpha','lobeta', 'hibeta', 'gamma1','gamma2','gamma3']

if __name__ == "__main__":
	##### obtain ch_pos
	filename = '/storage/Yann/saflow_DATA/alldata/SA04_SAflow-yharel_20190411_01.ds'
	raw = read_raw_ctf(filename, preload=False)
	ch_xy = get_ch_pos(raw)
	raw.close()
	all_acc = []
	all_pval = []
	for FREQ in FREQ_BANDS:
    	freq_acc = []
    	freq_pval = []
    	for CHAN in range(270):
        	savepath = '{}/LDA_{}_{}.mat'.format(RESULTS_PATH, FREQ, CHAN)
        	data_acc = loadmat(savepath)['acc_score']
        	data_pval = loadmat(savepath)['acc_pvalue']
        	freq_acc.append(data_acc)
        	freq_pval.append(data_pval)
    	all_acc.append(np.array(freq_acc).squeeze)
    	all_pval.append(np.array(freq_pval).squeeze)
	print(len(all_acc))
	print(len(all_pval))
	array_topoplot(all_acc, ch_xy, showtitle=True, titles=FREQ_BANDS, savefig=False)
