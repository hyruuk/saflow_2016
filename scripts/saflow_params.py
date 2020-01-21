import os.path as op

FOLDERPATH = '/storage/Yann/saflow_DATA/saflow_bids/'
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'
LOGS_DIR = '/home/karim/pCloudDrive/science/saflow/gradCPT/gradCPT_share_Mac_PC/gradCPT_share_Mac_PC/saflow_data/'
REPORTS_PATH = op.join(FOLDERPATH, 'preproc_reports')
FEAT_PATH = FOLDERPATH + 'features/'
FREQS = [ [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
BLOCS_LIST = ['2','3', '4', '5', '6', '7']
ZONE_CONDS = ['IN', 'OUT']
ZONE2575_CONDS = ['IN25', 'OUT75']
