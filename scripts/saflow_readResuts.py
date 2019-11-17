from scipy.io import loadmat
from hytools.meg_utils import array_topoplot, get_ch_pos

RESULTS_PATH = '/storage/Yann/saflow_DATA/LDA_results'

if __name__ == "__main__":
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
        all_acc.append(np.array(freq_acc))
        all_pval.append(np.array(freq_pval))
    all_acc = np.array(all_acc)
    all_pval = np.array(all_pval)
    print(all_acc.shape)
    print(all_pval.shape)
