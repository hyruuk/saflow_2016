import os.path as op
import numpy as np
import os
import mne
from joblib import Parallel, delayed
from mne import EpochsArray, create_info, pick_types, read_epochs, compute_source_morph
from mne.io import read_raw_ctf, read_raw_fif
from mne.io.pick import pick_info
from mne.viz import plot_alignment
from mne.minimum_norm import make_inverse_operator, compute_source_psd_epochs, apply_inverse,apply_inverse_epochs, source_band_induced_power, source_induced_power
from mne.coreg import coregister_fiducials
from mne.report import Report
from mne.parallel import parallel_func
from mne.time_frequency import csd_fourier
from mne.beamformer import make_dics, apply_dics_csd, tf_dics
import sys
from saflow_params import FOLDERPATH, FS_SUBJDIR
from utils import get_SAflow_bids

from itertools import product

subjects = ['06']
path = FOLDERPATH

path_meg = path + 'MEG_data/'
path_stc = path + 'source_rec/'

method = 'dSPM'
subjects_dir = FS_SUBJDIR

def compute_sources(subject, run, mri_available = True):
    epochs_fname = get_SAflow_bids(FOLDERPATH,subject, run, stage='epo')[1]
    epochs = read_epochs(epochs_fname)

    info = epochs.info
    noise_fname = '/storage/Yann/saflow_DATA/saflow_bids/sub-06/ses-recording/meg/sub-06_ses-recording_NOISE_meg.ds'
    noise_raw = read_raw_ctf(noise_fname,
                            preload=True)

    noise_raw.pick_channels(epochs.info['ch_names']) # Choose channels
    cov = mne.compute_raw_covariance(noise_raw, method='shrunk', cv=5, tmin = 0, tmax = 0.8) #change tmin and tmax ?
    src = mne.setup_source_space('sub-' + str(subject),subjects_dir=subjects_dir, add_dist=False)
    fname_src_fsaverage = subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif'


    surface = op.join(subjects_dir, 'sub-' + str(subject), 'bem', 'inner_skull.surf')
    vol_src = mne.setup_volume_source_space('sub-' + str(subject), subjects_dir=subjects_dir, mri='aseg.mgz',
                                            surface=surface)#,volume_label='Right-Pallidum')

    trans = get_SAflow_bids(FOLDERPATH, subject, run, stage='epotrans')[1]

    conductivity = (0.3,)  # for single layer
    fwd_filename = get_SAflow_bids(FOLDERPATH, subject, run, stage='epofwd')[1]
    #if not op.isfile(fwd_filename):
    model = mne.make_bem_model(subject='sub-' + str(subject), ico=4,
                   conductivity=conductivity,
                   subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(info,trans,vol_src,bem,eeg=False)
    mne.write_forward_solution(fwd_filename,fwd,overwrite=True)
    #         else:
    #             fwd = mne.read_forward_solution(fwd_filename)
    inverse_operator = make_inverse_operator(info, fwd, cov,loose=1)

    snr = 1.0
    lambda2 = 1.0 / snr ** 2


    for j, epoch in enumerate(epochs):
        print('Epoch {} of {}'.format(j, len(epochs)))
        epoch = epoch[np.newaxis,...]
        epoch = EpochsArray(epoch,info)
        epoch.pick_types(meg='mag')
        if method == 'dSPM':
            stc = apply_inverse_epochs(epoch,inverse_operator,lambda2,method='dSPM')
            src_fs = mne.read_source_spaces(fname_src_fsaverage)
            morph = mne.compute_source_morph(
                inverse_operator['src'], subject_from='sub-' + str(subject), subjects_dir=subjects_dir,
                src_to=src_fs, verbose=True)

            stc_fsaverage = morph.apply(stc[0])

            savepath = get_SAflow_bids(FOLDERPATH,subject, run, stage='eposources')[1]
            stc_fsaverage.save(savepath)

            del stc_fsaverage


if __name__ == "__main__":
    ARGS = sys.argv[1:]
    subject = ARGS[0]
    run = ARGS[1]
    compute_sources(subject, run)
