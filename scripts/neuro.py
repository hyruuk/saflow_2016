import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from autoreject import AutoReject
from scipy.io import loadmat, savemat
from brainpipe import feature
from saflow_params import FOLDERPATH, LOGS_DIR
from mne.io import read_raw_fif, read_raw_ctf
from hytools.meg_utils import get_ch_pos
from utils import get_SAflow_bids
from behav import find_logfile, get_VTC_from_file
import random
from matplotlib.pyplot import close

def find_rawfile(subj, bloc, BIDS_PATH):
    filepath = '/sub-{}/ses-recording/meg/'.format(subj)
    files = os.listdir(BIDS_PATH + filepath)
    for file in files:
        if file[-8] == bloc:
            filename = file
    return filepath, filename

def saflow_preproc(filepath, savepath, reportpath):
    report = mne.Report(verbose=True)
    raw = read_raw_ctf(filepath, preload=True)
    raw_data = raw.copy().apply_gradient_compensation(grade=3) #required for source reconstruction
    picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude='bads')
    fig = raw_data.plot(show=False);
    report.add_figs_to_section(fig, captions='Time series', section='Raw data')
    close(fig)
    fig = raw_data.plot_psd(average=False, picks=picks, show=False);
    report.add_figs_to_section(fig, captions='PSD', section='Raw data')
    close(fig)

    ## Filtering
    high_cutoff = 200
    low_cutoff = 0.5
    raw_data.filter(low_cutoff, high_cutoff, fir_design="firwin")
    raw_data.notch_filter(np.arange(60, high_cutoff+1, 60), picks=picks, filter_length='auto',phase='zero', fir_design="firwin")
    fig = raw_data.plot_psd(average=False, picks=picks, fmax=120, show=False);
    report.add_figs_to_section(fig, captions='PSD', section='Filtered data')
    close(fig)

    ## ICA
    ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
    fig = ica.plot_sources(raw_data, show=False);
    report.add_figs_to_section(fig, captions='Independent Components', section='ICA')
    close(fig)

    ## FIND ECG COMPONENTS
    ecg_threshold = 0.50
    ecg_epochs = create_ecg_epochs(raw_data, ch_name='EEG059')
    ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name='EEG059', method='ctps', threshold=ecg_threshold)
    fig = ica.plot_scores(ecg_scores, ecg_inds, show=False);
    report.add_figs_to_section(fig, captions='Correlation with ECG (EEG059)', section='ICA - ECG')
    close(fig)
    fig = list()
    try:
        fig = ica.plot_properties(ecg_epochs, picks=ecg_inds, image_args={'sigma': 1.}, show=False);
        for i, figure in enumerate(fig):
            report.add_figs_to_section(figure, captions='Detected component ' + str(i), section='ICA - ECG')
            close(figure)
    except:
        print('No component to remove')

    ## FIND EOG COMPONENTS
    eog_threshold = 4
    eog_epochs = create_eog_epochs(raw_data, ch_name='EEG057')
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name='EEG057', threshold=eog_threshold)
    fig = ica.plot_scores(eog_scores, eog_inds, show=False);
    report.add_figs_to_section(fig, captions='Correlation with EOG (EEG057)', section='ICA - EOG')
    close(fig)
    fig = list()
    try:
        fig = ica.plot_properties(eog_epochs, picks=eog_inds, image_args={'sigma': 1.}, show=False);
        for i, figure in enumerate(fig):
            report.add_figs_to_section(figure, captions='Detected component ' + str(i), section='ICA - EOG')
            close(figure)
    except:
        print('No component to remove')

    ## EXCLUDE COMPONENTS
    ica.exclude = ecg_inds
    ica.apply(raw_data)
    ica.exclude = eog_inds
    ica.apply(raw_data)
    fig = raw_data.plot(show=False); # Plot the clean signal.
    report.add_figs_to_section(fig, captions='After filtering + ICA', section='Raw data')
    close(fig)
    ## SAVE PREPROCESSED FILE
    report.save(reportpath, open_browser=False, overwrite=True);
    raw_data.save(savepath, overwrite=False)

    return raw_data

def segment_files(bids_filepath, tmin=0, tmax=0.8):
    raw = read_raw_fif(bids_filepath, preload=True)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=True, eeg=False, eog=False, stim=False)
    ### Set some constants for epoching
    baseline = None #(None, -0.05)
    #reject = {'mag': 4e-12}
    try:
        events = mne.find_events(raw, min_duration=1/raw.info['sfreq'], verbose=False)
    except ValueError:
        events = mne.find_events(raw, min_duration=2/raw.info['sfreq'], verbose=False)
    event_id = {'Freq': 21, 'Rare': 31}
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, baseline=baseline, reject=None, picks=picks, preload=True)
    ar = AutoReject(n_jobs=-1)
    epochs_clean, autoreject_log = ar.fit_transform(epochs, return_log=True)
    return epochs_clean, autoreject_log


def remove_errors(logfile, events):
    '''
    Takes as input the raw events, including both stimuli and responses.
    Outputs an event vector containing only the correct trials, and indices of their position in the list
    of all stims
    '''

    # load RT vector
    data = loadmat(logfile)
    df_response = pd.DataFrame(data['response'])
    RT_array= np.asarray(df_response.loc[:,4])

    # get events vector with only stimuli events
    new_events = []
    for ev in events:
        if ev[2] != 99:
            new_events.append(ev)
    events = np.array(new_events)

    # check for each trial if it has a response or not
    events_noerr = []

    events_comerr = []
    events_omerr = []

    for idx, event in enumerate(events):
        if event[2] == 21:
            if RT_array[idx] != 0:
                events_noerr.append(event)
            else:
                events_omerr.append(event)
        if event[2] == 31:
            if RT_array[idx] == 0:
                events_noerr.append(event)
            else:
                events_comerr.append(event)
    events_noerr = np.array(events_noerr)
    events_comerr = np.array(events_comerr)
    events_omerr = np.array(events_omerr)

    return events_noerr, events_comerr, events_omerr

def trim_events(events_noerr, events_artrej):
    '''
    This function compares the events vectors of correct epochs (events_noerr)
    and of kept epochs after auto-reject (events_artrej).
    Returns a list of intersecting epochs, + their idx in the clean epochs vector
    '''
    events_trimmed = []
    idx_trimmed = []
    for idx, event in enumerate(events_artrej):
        if event[0] in events_noerr[:,0]:
            events_trimmed.append(event)
            idx_trimmed.append(idx)

    events_trimmed = np.array(events_trimmed)
    idx_trimmed = np.array(idx_trimmed)

    print('N events in clean epochs : {}'.format(len(events_artrej)))
    print('N events in correct epochs : {}'.format(len(events_noerr)))
    print('N events in intersection : {}'.format(len(idx_trimmed)))
    return events_trimmed, idx_trimmed



def trim_INOUT_idx(INidx, OUTidx, events_trimmed, events):
    '''
    With INidx_trimmed refering to indices in events_artrej
    '''
    # get events vector with only stimuli events
    new_events = []
    for ev in events:
        if ev[2] != 99:
            new_events.append(ev)
    events = np.array(new_events)

    INidx_trimmed = []
    OUTidx_trimmed = []
    # compare trimmed events with all_events, and store corresponding indices
    for idx, ev in enumerate(events):
        for idx_trim, ev_trim in enumerate(events_trimmed):
            if ev[0] == ev_trim[0]:
                if idx in INidx:
                    INidx_trimmed.append(idx_trim)
                if idx in OUTidx:
                    OUTidx_trimmed.append(idx_trim)
    INidx_trimmed = np.array(INidx_trimmed)
    OUTidx_trimmed = np.array(OUTidx_trimmed)

    return INidx_trimmed, OUTidx_trimmed


def get_VTC_epochs(LOGS_DIR, subj, bloc, stage='epo', lobound=None, hibound=None, save_epochs=False, filt_order=3, filt_cutoff=0.1):
    '''
    This functions allows to use the logfile to split the epochs obtained in the epo.fif file.
    It works by comparing the timestamps of IN and OUT events to the timestamps in the epo file events
    It returns IN and OUT indices that are to be used in the split_PSD_data function

    TODO : FIND A WAY TO EARN TIME BY NOT LOADING THE DATA BUT JUST THE EVENTS
    '''

    ### Get events after artifact rejection have been performed
    epo_path, epo_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage=stage, cond=None)
    events_artrej = mne.read_events(epo_filename, verbose=False) # get events from the epochs file (so no resp event)

    ### Find logfile to extract VTC
    log_file = LOGS_DIR + find_logfile(subj,bloc,os.listdir(LOGS_DIR))
    VTC, INbounds, OUTbounds, INidx, OUTidx, RT_array = get_VTC_from_file(log_file, lobound=lobound, hibound=hibound, filt=True, filt_order=filt_order, filt_cutoff=filt_cutoff)

    ### Get original events and split them using the VTC
    events_fname, events_fpath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='preproc_raw', cond=None)
    raw = read_raw_fif(events_fpath, preload=False, verbose=False)#, min_duration=2/epochs.info['sfreq'])
    try:
        events = mne.find_events(raw, min_duration=1/raw.info['sfreq'], verbose=False)
    except ValueError:
        events = mne.find_events(raw, min_duration=2/raw.info['sfreq'], verbose=False)

    events_noerr, events_comerr, events_omerr = remove_errors(log_file, events)
    events_trimmed, idx_trimmed = trim_events(events_noerr, events_artrej)
    #rewrite INidx and OUTidx
    INidx, OUTidx = trim_INOUT_idx(INidx, OUTidx, events_trimmed, events)


    VTC_epo = np.array([VTC[idx] for idx in idx_trimmed])

    return INidx, OUTidx, VTC_epo, idx_trimmed

def compute_PSD(epochs, sf, epochs_length, f=None):
    if f == None:
        f = [ [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
    # Choose MEG channels
    data = epochs.get_data() # On sort les data de l'objet MNE pour les avoir dans une matrice (un numpy array pour être précis)
    data = data.swapaxes(0,1).swapaxes(1,2) # On réarange l'ordre des dimensions pour que ça correspond à ce qui est requis par Brainpipe
    objet_PSD = feature.power(sf=int(sf), npts=int(sf*epochs_length), width=int((sf*epochs_length)/2), step=int((sf*epochs_length)/4), f=f, method='hilbert1') # La fonction Brainpipe pour créer un objet de calcul des PSD
    data = data[:,0:int(sf*epochs_length),:] # weird trick pour corriger un pb de segmentation jpense
    #print(data.shape)
    psds = objet_PSD.get(data)[0] # Ici on calcule la PSD !
    return psds

def compute_TFR(epochs, baseline=False):
    decim = 3
    fmin = 2
    fmax = 150
    n_bins=30
    freqs = np.logspace(*np.log10([fmin, fmax]), num=n_bins)
    #freqs = np.arange(2,15,1)  # define frequencies of interest
    n_cycles = freqs / freqs[0]
    zero_mean = False
    this_tfr = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=n_cycles,
                      decim=decim, average=False, zero_mean=zero_mean,
                      return_itc=False)
    if baseline:
        this_tfr.apply_baseline(mode='ratio', baseline=(None, -0.05))
    #this_power = this_tfr.data[:, :, :, :]  # we only have one channel.
    return this_tfr

def load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, time_avg=True, stage='PSD'):
    '''
    Returns a list containing n_subj lists of n_blocs matrices of shape n_freqs X n_channels X n_trials
    '''
    PSD_alldata = []
    for subj in SUBJ_LIST:
        all_subj = [] ## all the data of one subject
        for run in BLOCS_LIST:
            SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(FOLDERPATH, subj, run, stage=stage, cond=None)
            mat = loadmat(SAflow_bidspath)['PSD']
            if time_avg == True:
                mat = np.mean(mat, axis=2) # average PSDs in time across epochs
            all_subj.append(mat)
        PSD_alldata.append(all_subj)
    return PSD_alldata

def load_VTC_data(FOLDERPATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST):
    VTC_alldata = []
    for subj in SUBJ_LIST:
        all_subj = [] ## all the data of one subject
        for run in BLOCS_LIST:
            # get events from epochs file
            epo_path, epo_filename = get_SAflow_bids(FOLDERPATH, subj, run, 'epo', cond=None)
            events_epoched = mne.read_events(epo_filename, verbose=False) # get events from the epochs file (so no resp event)
            # get events from original file (only 599 events)
            events_fname, events_fpath = get_SAflow_bids(FOLDERPATH, subj, run, stage='preproc_raw', cond=None)
            raw = read_raw_fif(events_fpath, preload=False, verbose=False)#, min_duration=2/epochs.info['sfreq'])
            all_events = mne.find_events(raw, min_duration=2/raw.info['sfreq'], verbose=False)
            stim_idx = []
            for i in range(len(all_events)):
                 if all_events[i,2] in [21,31]:
                     stim_idx.append(i)
            all_events = all_events[stim_idx]
            # compute VTC for all trials
            log_file = find_logfile(subj,run,os.listdir(LOGS_DIR))
            VTC, INbounds, OUTbounds, INzone, OUTzone = get_VTC_from_file(LOGS_DIR + log_file, lobound=None, hibound=None)
            epochs_VTC = []
            for event_time in events_epoched[:,0]:
                idx = list(all_events[:,0]).index(event_time)
                epochs_VTC.append(VTC[idx])
            all_subj.append(np.array(epochs_VTC))
        VTC_alldata.append(all_subj)
    return VTC_alldata

def split_TFR(filepath, subj, bloc, by='VTC', lobound=None, hibound=None, stage='1600TFR', filt_order=3, filt_cutoff=0.05):
    if by == 'VTC':
        event_id = {'IN': 1, 'OUT': 0}
        INidx, OUTidx, VTC_epochs, idx_trimmed = get_VTC_epochs(LOGS_DIR, subj, bloc, lobound=lobound, hibound=hibound, stage=stage[:-3]+'epo', save_epochs=False, filt_order=filt_order, filt_cutoff=filt_cutoff)
        epo_path, epo_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage=stage[:-3]+'epo', cond=None)
        epo_events = mne.read_events(epo_filename, verbose=False) # get events from the epochs file (so no resp event)
        TFR_path, TFR_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage=stage, cond=None)
        TFR = mne.time_frequency.read_tfrs(TFR_filename)
        for i, event in enumerate(epo_events):
            if i in INidx:
                TFR[0].events[i,2] = 1
            if i in OUTidx:
                TFR[0].events[i,2] = 0
        TFR[0].event_id = event_id
    return TFR

def split_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, by='VTC', lobound=None, hibound=None, stage='PSD', filt_order=3, filt_cutoff=0.1):
    '''
    This func splits the PSD data into two conditions. It returns a list of 2 (cond1 and cond2), each containing a list of n_subject matrices of shape n_freqs X n_channels X n_trials
    '''
    PSD_alldata = load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, time_avg=True, stage=stage)
    PSD_cond1 = []
    PSD_cond2 = []
    for subj_idx, subj in enumerate(SUBJ_LIST):
        subj_cond1 = []
        subj_cond2 = []
        for bloc_idx, bloc in enumerate(BLOCS_LIST):
            print('Splitting sub-{}_run-{}'.format(subj, bloc))

            # Obtain indices of the two conditions
            if by == 'VTC':
                INidx, OUTidx, VTC_epochs, idx_trimmed = get_VTC_epochs(LOGS_DIR, subj, bloc, lobound=lobound, hibound=hibound, save_epochs=False, filt_order=filt_order, filt_cutoff=filt_cutoff)
                cond1_idx = INidx
                cond2_idx = OUTidx
            if by == 'odd':
                # Get indices of freq and rare events
                ev_fname, ev_fpath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='epo')
                events_artrej = mne.read_events(ev_fpath)
                log_file = LOGS_DIR + find_logfile(subj,bloc,os.listdir(LOGS_DIR))
                events_fname, events_fpath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='preproc_raw', cond=None)
                raw = read_raw_fif(events_fpath, preload=False, verbose=False)#, min_duration=2/epochs.info['sfreq'])
                try:
                    events = mne.find_events(raw, min_duration=1/raw.info['sfreq'], verbose=False)
                except ValueError:
                    events = mne.find_events(raw, min_duration=2/raw.info['sfreq'], verbose=False)

                events_noerr, events_comerr, events_omerr = remove_errors(log_file, events)
                events_trimmed, idx_trimmed = trim_events(events_noerr, events_artrej)
                cond1_idx = []
                cond2_idx = []
                for idx, ev in enumerate(events_trimmed):
                    if ev[2] == 21: # Frequent events
                        cond1_idx.append(idx)
                    if ev[2] == 31:
                        cond2_idx.append(idx)
                cond1_idx = np.array(cond1_idx)
                cond2_idx = np.array(cond2_idx)
                # Add this to keep the same number of trials in both conditions
                random.seed(0)
                cond1_idx = random.choices(cond1_idx, k=len(cond2_idx))
                print('N trials retained for each condition : {}'.format(len(cond2_idx)))
            # Pick the data of each condition
            if bloc_idx == 0: # if first bloc, init ndarray size using the first matrix
                subj_cond1 = PSD_alldata[subj_idx][bloc_idx][:,:,cond1_idx]
                subj_cond2 = PSD_alldata[subj_idx][bloc_idx][:,:,cond2_idx]
            else: # if not first bloc, just concatenate along the trials dimension
                subj_cond1 = np.concatenate((subj_cond1, PSD_alldata[subj_idx][bloc_idx][:,:,cond1_idx]), axis=2)
                subj_cond2 = np.concatenate((subj_cond2, PSD_alldata[subj_idx][bloc_idx][:,:,cond2_idx]), axis=2)
        PSD_cond1.append(subj_cond1)
        PSD_cond2.append(subj_cond2)
    splitted_PSD = [PSD_cond1, PSD_cond2]
    return splitted_PSD
