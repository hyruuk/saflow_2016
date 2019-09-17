##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow_utils import find_logfile, find_preprocfile, split_events_by_VTC
import mne
import os
from mne.io import read_raw_fif
import numpy as np
from autoreject import AutoReject



folderpath = '/storage/Yann/saflow_DATA/'
PREPROC_PATH = folderpath + 'saflow_preproc/'
LOGS_DIR = "/home/karim/pCloudDrive/science/saflow/gradCPT/gradCPT_share_Mac_PC/gradCPT_share_Mac_PC/saflow_data/"
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'
EPOCHS_DIR = folderpath + 'saflow_epoched/'
PSDS_DIR = folderpath + 'saflow_PSD/'

FREQS = [ [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
FREQS_NAMES = ['theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']

subj_list = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
blocs_list = ['1','2','3', '4', '5', '6']


if __name__ == "__main__":
    for subj in subj_list:
        for bloc in blocs_list:
            ### Find logfile to extract VTC
            log_file = find_logfile(subj,bloc,os.listdir(LOGS_DIR))
            VTC, INbounds, OUTbounds, INzone, OUTzone = get_VTC_from_file(LOGS_DIR + log_file, lobound=None, hibound=None)
            ### Load pre-processed datafile
            preproc_file = find_preprocfile(subj, bloc, os.listdir(PREPROC_PATH))
            raw = read_raw_fif(PREPROC_PATH + preproc_file, preload=True)
            picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False)

            ### Set some constants for epoching

            baseline = (None, 0.0)
            reject = {'mag': 4e-12}
            tmin, tmax = 0, 0.8

            ### Find events, split them by IN/OUT and start epoching
            events = mne.find_events(raw, min_duration=2/raw.info['sfreq'])
            INevents, OUTevents = split_events_by_VTC(INzone, OUTzone, events)
            try:
                event_id = {'Freq': 21, 'Rare': 31}
                INepochs = mne.Epochs(raw, events=INevents, event_id=event_id, tmin=tmin,
                                tmax=tmax, baseline=baseline, reject=None, picks=picks, preload=True)
            except:
                event_id = {'Freq': 21}
                INepochs = mne.Epochs(raw, events=INevents, event_id=event_id, tmin=tmin,
                                tmax=tmax, baseline=baseline, reject=None, picks=picks, preload=True)
                
            try:
                event_id = {'Freq': 21, 'Rare': 31}
                OUTepochs = mne.Epochs(raw, events=OUTevents, event_id=event_id, tmin=tmin,
                                tmax=tmax, baseline=baseline, reject=None, picks=picks, preload=True)
            except:
                event_id = {'Freq': 21}
                OUTepochs = mne.Epochs(raw, events=OUTevents, event_id=event_id, tmin=tmin,
                                tmax=tmax, baseline=baseline, reject=None, picks=picks, preload=True)
                

            ### Autoreject detects, rejects and interpolate artifacted epochs
            ar = AutoReject()
            INepochs_clean = ar.fit_transform(INepochs)
            OUTepochs_clean = ar.fit_transform(OUTepochs)
            
            INepochs_clean.save(EPOCHS_DIR + 'SA' + subj + '_' + bloc + '_IN_epo.fif.gz')
            OUTepochs_clean.save(EPOCHS_DIR + 'SA' + subj + '_' + bloc + '_OUT_epo.fif.gz')
