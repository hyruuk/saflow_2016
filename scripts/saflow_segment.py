##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow_utils import find_logfile, find_preprocfile, split_events_by_VTC
import mne
import os
from mne.io import read_raw_fif
import numpy as np
from autoreject import AutoReject
from hytools.vtc_utils import get_VTC_from_file



FOLDERPATH = '/storage/Yann/saflow_DATA/'
PREPROC_PATH = FOLDERPATH + 'saflow_preproc/'
EPOCHS_DIR = FOLDERPATH + 'saflow_epoched_noAR/'
LOGS_DIR = "/home/karim/pCloudDrive/science/saflow/gradCPT/gradCPT_share_Mac_PC/gradCPT_share_Mac_PC/saflow_data/"


SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
#SUBJ_LIST = ['14']
#BLOCS_LIST = ['5', '6']
BLOCS_LIST = ['1','2','3', '4', '5', '6']


if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
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
