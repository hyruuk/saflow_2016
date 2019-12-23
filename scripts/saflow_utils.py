import mne
import os
from mne.io import read_raw_fif
import numpy as np
import matplotlib.pyplot as plt
from hytools.vtc_utils import *
from hytools.meg_utils import get_ch_pos
from autoreject import AutoReject
from scipy.io import loadmat, savemat
from brainpipe import feature
from saflow_params import FOLDERPATH

def get_SAflow_bids(FOLDERPATH, subj, run, stage, cond=None):
    '''
    Constructs BIDS basename and filepath in the SAflow database format.
    '''
    if run == '1' or run == '8': # determine task based on run number
        task = 'RS'
    else:
        task = 'gradCPT'

    if stage == 'epo' or stage == 'raw': # determine extension based on stage
        extension = '.fif.gz'
    elif stage == 'PSD':
        extension = '.mat'
    elif stage == 'sources':
        extension = '.hd5'

    if cond == None: # build basename with or without cond
        SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_meg_{}{}'.format(subj, task, run, stage, extension)
    else:
        SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_meg_{}_{}{}'.format(subj, task, run, cond, stage, extension)

    SAflow_bidspath = os.path.join(FOLDERPATH, 'sub-{}'.format(subj), 'ses-recording', 'meg', SAflow_bidsname)
    return SAflow_bidsname, SAflow_bidspath

def find_logfile(subj, bloc, log_files):
    ### Find the right logfile for a specific subject and bloc in a list of log_files
    # (typically the list of files in the log folder, obtained by "os.listdir(LOGS_DIR)")
    for file in log_files:
        if file[7:9] ==  subj and file[10] == bloc:
            break
    return file

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

def split_events_by_VTC(INzone, OUTzone, events):
    '''
    This function uses the event IDs in INzone and OUTzone variables and splits
    events (ndarray of shape 3 X n_events)
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

def segment_files_INvsOUT(LOGS_DIR, subj, bloc, lobound=None, hibound=None):
    '''
    TODO : get lobound and hibound out of this function, in order to retain trials that are interpolated altogether
    '''
    ### Load pre-processed datafile
    bids_filename = 'sub-{}_ses-recording_task-gradCPT_run-0{}_meg_preproc_raw.fif.gz'.format(subj, bloc)
    bids_filepath = os.path.join(FOLDERPATH, 'sub-{}'.format(subj), 'ses-recording', 'meg', bids_filename)
    print(bids_filepath)
    raw = read_raw_fif(bids_filepath, preload=True)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False)

    ### Set some constants for epoching
    baseline = None #(None, 0.0)
    reject = {'mag': 4e-12}
    tmin, tmax = 0, 0.8

    ### Find logfile to extract VTC
    log_file = find_logfile(subj,bloc,os.listdir(LOGS_DIR))
    VTC, INbounds, OUTbounds, INzone, OUTzone = get_VTC_from_file(LOGS_DIR + log_file, lobound=None, hibound=None)
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
    return INepochs_clean, OUTepochs_clean

def compute_PSD(epochs, sf, epochs_length, f=None):
    if f == None:
        f = [ [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
    # Choose MEG channels
    data = epochs.get_data() # On sort les data de l'objet MNE pour les avoir dans une matrice (un numpy array pour être précis)
    data = data.swapaxes(0,1).swapaxes(1,2) # On réarange l'ordre des dimensions pour que ça correspond à ce qui est requis par Brainpipe
    objet_PSD = feature.power(sf=int(sf), npts=int(sf*epochs_length), width=int((sf*epochs_length)/2), step=int((sf*epochs_length)/4), f=f, method='hilbert1') # La fonction Brainpipe pour créer un objet de calcul des PSD
    data = data[:,0:960,:] # weird trick pour corriger un pb de segmentation jpense
    #print(data.shape)
    psds = objet_PSD.get(data)[0] # Ici on calcule la PSD !
    return psds

def array_topoplot(toplot, ch_xy, showtitle=False, titles=None, savefig=False, figpath=None, vmin=-1, vmax=1, cmap='magma', with_mask=False, masks=None, show=True):
    #create fig
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=5)
    fig, ax = plt.subplots(1,len(toplot), figsize=(20,10))
    for i, data in enumerate(toplot):
        if with_mask == False:
            image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], show=False, contours=None, extrapolate='box', outlines='head')
        elif with_mask == True:
            image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], show=False, contours=None, mask_params=mask_params, mask=masks[i], extrapolate='box', outlines='head')
        #option for title
        if showtitle == True:
            ax[i].set_title(titles[i], fontdict={'fontsize': 20, 'fontweight': 'heavy'})
    #add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[-1])
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=14)
    #save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)
    if show == True:
        plt.show()
        plt.close(fig=fig)
    else:
        plt.close(fig=fig)
    return fig

def create_pval_mask(pvals, alpha=0.05):
    mask = np.zeros((len(pvals),), dtype='bool')
    for i, pval in enumerate(pvals):
        if pval <= alpha:
            mask[i] = True
    return mask

def load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, COND_LIST):
    '''
    Returns a list containing 2 (n_conditions) lists, each with n_subj matrices of shape n_freqs X n_channels X n_trials
    '''
    PSD_alldata = []
    for cond in COND_LIST:
        all_cond = [] ## all the data of one condition
        for subj in SUBJ_LIST:
            all_subj = [] ## all the data of one subject
            for run in BLOCS_LIST:
                SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(FOLDERPATH, subj, run, stage='PSD', cond=cond)
                mat = loadmat(SAflow_bidspath)['PSD']
                if all_subj == []:
                    all_subj = mat
                else:
                    all_subj = np.concatenate((all_subj, mat), axis=2)
            all_cond.append(all_subj)
        PSD_alldata.append(all_cond)
    return PSD_alldata # List containing 2 (n_conditions) lists, each with n_subj matrices of shape n_freqs X n_channels X n_trials
