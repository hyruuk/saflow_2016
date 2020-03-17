import os
import numpy as np
import mne
from autoreject import AutoReject
from scipy.io import loadmat, savemat
from brainpipe import feature
from saflow_params import FOLDERPATH, LOGS_DIR
from mne.io import read_raw_fif, read_raw_ctf
from hytools.meg_utils import get_ch_pos
from utils import get_SAflow_bids
from behav import find_logfile, get_VTC_from_file

def find_rawfile(subj, bloc, BIDS_PATH):
    filepath = '/sub-{}/ses-recording/meg/'.format(subj)
    files = os.listdir(BIDS_PATH + filepath)
    for file in files:
        if file[-8] == bloc:
            filename = file
    return filepath, filename

def saflow_preproc(filepath, savepath, reportpath):
    report = mne.Report(verbose=True)
    raw_data = read_raw_ctf(filepath, preload=True)
    picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude='bads')
    fig = raw_data.plot(show=False);
    report.add_figs_to_section(fig, captions='Time series', section='Raw data')
    closefig(fig)
    fig = raw_data.plot_psd(average=False, picks=picks, show=False);
    report.add_figs_to_section(fig, captions='PSD', section='Raw data')
    closefig(fig)

    ## Filtering
    high_cutoff = 200
    low_cutoff = 0.5
    raw_data.filter(low_cutoff, high_cutoff, fir_design="firwin")
    raw_data.notch_filter(np.arange(60, high_cutoff+1, 60), picks=picks, filter_length='auto',phase='zero', fir_design="firwin")
    fig = raw_data.plot_psd(average=False, picks=picks, fmax=120, show=False);
    report.add_figs_to_section(fig, captions='PSD', section='Filtered data')
    closefig(fig)

    ## ICA
    ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
    fig = ica.plot_sources(raw_data, show=False);
    report.add_figs_to_section(fig, captions='Independent Components', section='ICA')
    closefig(fig)

    ## FIND ECG COMPONENTS
    ecg_threshold = 0.50
    ecg_epochs = create_ecg_epochs(raw_data, ch_name='EEG059')
    ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name='EEG059', method='ctps', threshold=ecg_threshold)
    fig = ica.plot_scores(ecg_scores, ecg_inds, show=False);
    report.add_figs_to_section(fig, captions='Correlation with ECG (EEG059)', section='ICA - ECG')
    closefig(fig)
    fig = list()
    try:
        fig = ica.plot_properties(ecg_epochs, picks=ecg_inds, image_args={'sigma': 1.}, show=False);
        for i, figure in enumerate(fig):
            report.add_figs_to_section(figure, captions='Detected component ' + str(i), section='ICA - ECG')
            closefig(figure)
    except:
        print('No component to remove')

    ## FIND EOG COMPONENTS
    eog_threshold = 4
    eog_epochs = create_eog_epochs(raw_data, ch_name='EEG057')
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name='EEG057', threshold=eog_threshold)
    fig = ica.plot_scores(eog_scores, eog_inds, show=False);
    report.add_figs_to_section(fig, captions='Correlation with EOG (EEG057)', section='ICA - EOG')
    closefig(fig)
    fig = list()
    try:
        fig = ica.plot_properties(eog_epochs, picks=eog_inds, image_args={'sigma': 1.}, show=False);
        for i, figure in enumerate(fig):
            report.add_figs_to_section(figure, captions='Detected component ' + str(i), section='ICA - EOG')
            closefig(figure)
    except:
        print('No component to remove')

    ## EXCLUDE COMPONENTS
    ica.exclude = ecg_inds
    ica.apply(raw_data)
    ica.exclude = eog_inds
    ica.apply(raw_data)
    fig = raw_data.plot(show=False); # Plot the clean signal.
    report.add_figs_to_section(fig, captions='After filtering + ICA', section='Raw data')
    closefig(fig)
    ## SAVE PREPROCESSED FILE
    report.save(reportpath, open_browser=False);
    try:
        raw_data.save(savepath, overwrite=False)
    except:
        print('File already exists')

    return raw_data

def segment_files(bids_filepath, tmin=0, tmax=0.8):
    raw = read_raw_fif(bids_filepath, preload=True)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False)
    ### Set some constants for epoching
    baseline = (None, -0.05)
    reject = {'mag': 4e-12}
    events = mne.find_events(raw, min_duration=2/raw.info['sfreq'])
    event_id = {'Freq': 21, 'Rare': 31, 'Resp': 99}
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, baseline=baseline, reject=None, picks=picks, preload=True)
    ar = AutoReject(n_jobs=-1)
    epochs_clean, autoreject_log = ar.fit_transform(epochs, return_log=True)
    return epochs_clean, autoreject_log


def split_events_by_VTC(INzone, OUTzone, events):
    '''
    This function uses the event IDs in INzone and OUTzone variables and splits
    MNE events (ndarray of shape 3 X n_events)
    '''
    INevents = []
    OUTevents = []
    counter = 0
    for i, event in enumerate(events):
        if event[2] == 21:
            try:
                if events[i+1][2] == 99:
                    if counter in INzone:
                        INevents.append(event)
                    if counter in OUTzone:
                        OUTevents.append(event)
            except:
                print('last event')
            counter += 1
        elif event[2] == 31:
            try:
                if events[i+1][2] != 99:
                    if counter in INzone:
                        INevents.append(event)
                    if counter in OUTzone:
                        OUTevents.append(event)
            except:
                print('last event')
            counter += 1
    INevents = np.asarray(INevents)
    OUTevents = np.asarray(OUTevents)
    return INevents, OUTevents

def split_events_by_VTC_alltrials(INzone, OUTzone, events):
    ### keeps all trials, miss included
    INevents = []
    OUTevents = []
    counter = 0
    for i, event in enumerate(events):
        if event[2] == 21:
            try:
                if counter in INzone:
                    INevents.append(event)
                if counter in OUTzone:
                    OUTevents.append(event)
            except:
                print('last event')
            counter += 1
        elif event[2] == 31:
            try:
                if counter in INzone:
                    INevents.append(event)
                if counter in OUTzone:
                    OUTevents.append(event)
            except:
                print('last event')
            counter += 1
    INevents = np.asarray(INevents)
    OUTevents = np.asarray(OUTevents)
    return INevents, OUTevents

def get_VTC_epochs(LOGS_DIR, subj, bloc, lobound=None, hibound=None, save_epochs=False, filt_order=3, filt_cutoff=0.05):
    '''
    This functions allows to use the logfile to split the epochs obtained in the epo.fif file.
    It works by comparing the timestamps of IN and OUT events to the timestamps in the epo file events
    It returns IN and OUT indices that are to be used in the split_PSD_data function
    HERE WE KEEP ONLY CORRECT TRIALS

    TODO : FIND A WAY TO EARN TIME BY NOT LOADING THE DATA BUT JUST THE EVENTS
    '''
    epo_path, epo_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond=None)
    epo_events = mne.read_events(epo_filename, verbose=False) # get events from the epochs file (so no resp event)
    ### Find logfile to extract VTC
    log_file = find_logfile(subj,bloc,os.listdir(LOGS_DIR))
    VTC, INbounds, OUTbounds, INzone, OUTzone = get_VTC_from_file(LOGS_DIR + log_file, lobound=lobound, hibound=hibound, filt=True, filt_order=filt_order, filt_cutoff=filt_cutoff)
    ### Find events, split them by IN/OUT and start epoching
    events_fname, events_fpath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='preproc_raw', cond=None)
    raw = read_raw_fif(events_fpath, preload=False, verbose=False)#, min_duration=2/epochs.info['sfreq'])
    events = mne.find_events(raw, min_duration=2/raw.info['sfreq'], verbose=False)
    INevents, OUTevents = split_events_by_VTC(INzone, OUTzone, events)
    INidx = []
    OUTidx = []
    # the droping of epochs is a bit confused because we have to get indices from the current cleaned epochs file
    for idx, ev in enumerate(epo_events):
        if ev[0] in INevents[:,0]: #compare timestamps
            INidx.append(idx)
        if ev[0] in OUTevents[:,0]:
            OUTidx.append(idx)
    INidx = np.array(INidx)
    OUTidx = np.array(OUTidx)

    if save_epochs == True:
        epo_idx = np.array(range(len(epo_events)))
        IN_todrop = np.delete(epo_idx, INidx) # drop all epochs EXCEPT INidx
        OUT_todrop = np.delete(epo_idx, OUTidx)
        INepochs = mne.read_epochs(epo_filename, preload=False, verbose=False)
        INepochs = INepochs.drop(indices=IN_todrop)
        OUTepochs = mne.read_epochs(epo_filename, preload=False, verbose=False)
        OUTepochs = OUTepochs.drop(indices=OUT_todrop)
        if lobound == None:
            INpath, INfilename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond='IN')
        else:
            INpath, INfilename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond='IN{}'.format(lobound*100))
        if hibound == None:
            OUTpath, OUTfilename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond='OUT')
        else:
            OUTpath, OUTfilename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond='OUT{}'.format(hibound*100))
        INepochs.save(INfilename)
        OUTepochs.save(OUTfilename)

    return INidx, OUTidx

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

def compute_TFR(epochs, baseline=True):
    decim = 2
    freqs = np.arange(2, 120, 1)  # define frequencies of interest
    n_cycles = freqs / freqs[0]
    zero_mean = False
    this_tfr = mne.time_frequency.tfr_morlet(condition, freqs, n_cycles=n_cycles,
                      decim=decim, average=False, zero_mean=zero_mean,
                      return_itc=False)
    if baseline:
        this_tfr.apply_baseline(mode='ratio', baseline=(None, 0))
    this_power = this_tfr.data[:, :, :, :]  # we only have one channel.
    return this_power

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
            print(len(VTC))
            print(log_file)
            print(len(all_events))
            print(events_fname)
            epochs_VTC = []
            for event_time in events_epoched[:,0]:
                idx = list(all_events[:,0]).index(event_time)
                epochs_VTC.append(VTC[idx])
            all_subj.append(np.array(epochs_VTC))
        VTC_alldata.append(all_subj)
    return VTC_alldata

def split_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, by='VTC', lobound=None, hibound=None, stage='PSD', filt_order=3, filt_cutoff=0.05):
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
            if by == 'VTC':
                INidx, OUTidx = get_VTC_epochs(LOGS_DIR, subj, bloc, lobound=lobound, hibound=hibound, save_epochs=False, filt_order=filt_order, filt_cutoff=filt_cutoff)
                cond1_idx = INidx
                cond2_idx = OUTidx
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
