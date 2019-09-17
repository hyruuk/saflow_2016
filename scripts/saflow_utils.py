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

def find_logfile(subj, bloc, log_files):
    ### Find the right logfile for a specific subject and bloc in a list of log_files
    # (typically the list of files in the log folder, obtained by "os.listdir(LOGS_DIR)")
    for file in log_files:
        if file[7:9] ==  subj and file[10] == bloc:
            break
    return file

def find_preprocfile(subj, bloc, log_files):
    ### Find the right logfile for a specific subject and bloc in a list of log_files
    # (typically the list of files in the log folder, obtained by "os.listdir(LOGS_DIR)")
    for file in log_files:
        if file[2:4] ==  subj and file[5] == str(int(bloc)):
            break
    return file


def split_events_by_VTC(INzone, OUTzone, events):
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

def array_topoplot(toplot, ch_xy, showtitle=False, titles=None, savefig=False, figpath=None, vmin=-1, vmax=1):
    #create fig
    fig, ax = plt.subplots(1,len(toplot), figsize=(20,10))
    for i, data in enumerate(toplot):
        image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap='magma', vmin=vmin, vmax=vmax, axes=ax[i], show=False)
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
    plt.show()