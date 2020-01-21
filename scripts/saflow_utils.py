import mne
import os
from mne.io import read_raw_fif
import numpy as np
import matplotlib.pyplot as plt
from hytools.meg_utils import get_ch_pos
from autoreject import AutoReject
from scipy.io import loadmat, savemat
from brainpipe import feature
from saflow_params import FOLDERPATH
from scipy.io import loadmat, savemat
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

### VTC computation

def interp_RT(RT):
    ### Interpolate missing reaction times using the average of proximal values.
    # Note that this technique behaves poorly when two 0 are following each other
    for i in range(len(RT)):
        if RT[i] == 0:
            try:
                RT[i] = np.mean((RT[i-1], RT[i+1]))
            except:
                RT[i] = RT[i-1]
    RT_interpolated = RT
    return RT_interpolated

def compute_VTC(RT_interp, filt=True, filt_order=3, filt_cutoff=0.05):
    ### Compute the variance time course (VTC) of the array RT_interp
    VTC = (RT_interp - np.mean(RT_interp))/np.std(RT_interp)
    if filt == True:
        b, a = signal.butter(filt_order,filt_cutoff)
        VTC_filtered = signal.filtfilt(b, a, abs(VTC))
    VTC = VTC_filtered
    return VTC

def in_out_zone(VTC, lobound = None, hibound = None):
    ### Collects the indices of IN/OUT zone trials
    # lobound and hibound are values between 0 and 1 representing quantiles
    INzone = []
    OUTzone = []
    if lobound == None and hibound == None:
        VTC_med = np.median(VTC)
        for i, val in enumerate(VTC):
            if val < VTC_med:
                INzone.append(i)
            if val >= VTC_med:
                OUTzone.append(i)
    else:
        low = np.quantile(VTC, lobound)
        high = np.quantile(VTC, hibound)
        for i, val in enumerate(VTC):
            if val < low:
                INzone.append(i)
            if val >= high:
                OUTzone.append(i)
    INzone = np.asarray(INzone)
    OUTzone = np.asarray(OUTzone)
    return INzone, OUTzone

def find_jumps(array):
    ### Finds the jumps in an array containing ordered sequences
    jumps = []
    for i,_ in enumerate(array):
        try:
            if array[i+1] != array[i]+1:
                jumps.append(i)
        except:
            break
    return jumps

def find_bounds(array):
    ### Create a list of tuples, each containing the first and last values of every ordered sequences
    # contained in a 1D array
    jumps = find_jumps(array)
    bounds = []
    for i, jump in enumerate(jumps):
        if jump == jumps[0]:
            bounds.append(tuple([array[0], array[jump]]))
        else:
            bounds.append(tuple([array[jumps[i-1]+1], array[jump]]))
        if i == len(jumps)-1:
            bounds.append(tuple([array[jump+1], array[-1]]))
    return bounds

def get_VTC_from_file(filepath, lobound = None, hibound = None):
    data = loadmat(filepath)
    df_response = pd.DataFrame(data['response'])
    RT_array= np.asarray(df_response.loc[:,4])
    RT_interp = interp_RT(RT_array)
    VTC = compute_VTC(RT_interp)
    INzone, OUTzone = in_out_zone(VTC, lobound=lobound, hibound=hibound)
    INbounds = find_bounds(INzone)
    OUTbounds = find_bounds(OUTzone)
    return VTC, INbounds, OUTbounds, INzone, OUTzone

def plot_VTC(VTC, figpath=None, save=False):
    x = np.arange(0, len(VTC))
    OUT_mask = np.ma.masked_where(VTC >= np.median(VTC), VTC)
    IN_mask = np.ma.masked_where(VTC < np.median(VTC), VTC)
    lines = plt.plot(x, OUT_mask, x, IN_mask)
    fig = plt.plot()
    plt.setp(lines[0], linewidth=2)
    plt.setp(lines[1], linewidth=2)
    plt.legend(('IN zone', 'OUT zone'), loc='upper right')
    plt.title('IN vs OUT zone')
    if save == True:
        plt.savefig(figpath)
    plt.show()



def get_SAflow_bids(FOLDERPATH, subj, run, stage, cond=None):
    '''
    Constructs BIDS basename and filepath in the SAflow database format.
    '''
    if run == '1' or run == '8': # determine task based on run number
        task = 'RS'
    else:
        task = 'gradCPT'

    if stage == 'epo' or stage == 'raw' or stage == 'preproc_raw': # determine extension based on stage
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

def split_epochs(LOGS_DIR, subj, bloc, lobound=None, hibound=None, save_epochs=False):
    '''
    This functions allows to use the logfile to split the epochs obtained in the epo.fif file.
    It works by comparing the timestamps of IN and OUT events to the timestamps in the epo file events
    Ultimately, we want to compute our features on all epochs then split them as needed. That's why we gonna use IN and OUTidx.
    HERE WE KEEP ONLY CORRECT TRIALS
    '''
    epo_path, epo_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, 'epo', cond=None)
    ### Find logfile to extract VTC
    log_file = find_logfile(subj,bloc,os.listdir(LOGS_DIR))
    VTC, INbounds, OUTbounds, INzone, OUTzone = get_VTC_from_file(LOGS_DIR + log_file, lobound=lobound, hibound=hibound)
    ### Find events, split them by IN/OUT and start epoching
    preproc_path, preproc_filename = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='preproc_raw', cond=None)
    raw = read_raw_fif(preproc_filename, preload=False)#, min_duration=2/epochs.info['sfreq'])
    events = mne.find_events(raw, min_duration=2/raw.info['sfreq'])
    INevents, OUTevents = split_events_by_VTC(INzone, OUTzone, events)
    print('event length : {}'.format(len(events)))
    INidx = []
    OUTidx = []
    epo_events = mne.read_events(epo_filename) # get events from the epochs file (so no resp event)
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
        INepochs = mne.read_epochs(epo_filename, preload=False)
        INepochs = INepochs.drop(indices=IN_todrop)
        OUTepochs = mne.read_epochs(epo_filename, preload=False)
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

def segment_files(bids_filepath):
    raw = read_raw_fif(bids_filepath, preload=True)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False)
    ### Set some constants for epoching
    baseline = None #(None, 0.0)
    reject = {'mag': 4e-12}
    tmin, tmax = 0, 0.8
    events = mne.find_events(raw, min_duration=2/raw.info['sfreq'])
    event_id = {'Freq': 21, 'Rare': 31, 'Resp': 99}
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, baseline=baseline, reject=None, picks=picks, preload=True)
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    return epochs_clean

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

def load_PSD_data(FOLDERPATH, SUBJ_LIST, BLOCS_LIST, COND_LIST, time_avg=True):
    '''
    Returns a list containing 2 (n_conditions) lists, each with n_subj matrices of shape n_freqs X n_channels X n_trials
    NEXT STEP : MODIFY THIS SO IT RETURNS A LIST OF N_SUBJ * N_BLOCS.
    AND MAKE A FUNCTION SO THIS LIST IS SPLITED BY IN/OUT IN THE STATS SCRIPT (AND AVERAGED ACROSS TIME)
    '''
    PSD_alldata = []
    for cond in COND_LIST:
        all_cond = [] ## all the data of one condition
        for subj in SUBJ_LIST:
            all_subj = [] ## all the data of one subject
            all_subj_OUT = []
            for run in BLOCS_LIST:
                SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(FOLDERPATH, subj, run, stage='PSD', cond=cond)
                mat = loadmat(SAflow_bidspath)['PSD']
                if time_avg == True:
                    mat = np.mean(mat, axis=2) # average PSDs in time across epochs
                if all_subj == []:
                    all_subj = mat
                else:
                    all_subj = np.concatenate((all_subj, mat), axis=2)
            all_cond.append(all_subj)
        PSD_alldata.append(all_cond)
    return PSD_alldata # List containing 2 (n_conditions) lists, each with n_subj matrices of shape n_freqs X n_channels X n_trials
