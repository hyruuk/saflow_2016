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

from itertools import product

subjects = ['H01','H02','H03','H04','H05','H08','H09','H11',
          'H12','H13','H14','H15','H16','H17','H18','H19','H21','H22','H23',
          'H24','H25','H26','H27','H28','H29','H30','H31','H32']

path = '/home/thomast/scratch/'
path_meg = path + 'MEG_data/'
path_stc = path + 'source_rec/'
method = 'dSPM'
subjects_dir = path + 'IRM/'
triggs = ['go', 'enter']

def compute_sources(s):
    filenames  =  [s + '_Fast1',s + '_Fast2',s + '_Fast3',s + '_Fast4',
          s + '_Slow1',s + '_Slow2',s + '_Slow3',s + '_Slow4']
    for f in filenames:
        for t in triggs:
            epochs_fname = path + 'MEG_data/' + s + '/MEG_data_epoched_' + t +  '/' + f + '.fif'
            stc_fname = path +'source_rec/stc_' + t +  '/'

            epochs = read_epochs(epochs_fname)
            epochs = epochs.copy().resample(600, npad='auto')

            info = epochs.info
            noise_raw = read_raw_fif(path + 'MEG_data/' + s + '/MEG_data_epoched/' + s + '_noise.fif',
                                    preload=True)

            noise_raw.pick_channels(epochs.info['ch_names']) # Choose channels
            cov = mne.compute_raw_covariance(noise_raw,method='shrunk', cv=5, tmin = 0., tmax = 10.) #change tmin and tmax ?
            src = mne.setup_source_space(s,subjects_dir=subjects_dir, add_dist=False)
            fname_src_fsaverage = subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif'


            surface = op.join(subjects_dir, s, 'bem', 'inner_skull.surf')
            vol_src = mne.setup_volume_source_space(s, subjects_dir=subjects_dir,mri='aseg.mgz',
                                                    surface=surface)#,volume_label='Right-Pallidum')

            trans = path + 'MEG_data/' + s + '/MEG_data_epoched/' + f + '-trans.fif'

            conductivity = (0.3,)  # for single layer
            fwd_filename = path_meg+ f + '-fwd.fif' 
            #if not op.isfile(fwd_filename):
            model = mne.make_bem_model(subject=s, ico=4,
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
            epoch = epoch[np.newaxis,...]
            epoch = EpochsArray(epoch,info)
            epoch.pick_types(meg='mag')
            if method == 'dSPM':
                stc = apply_inverse_epochs(epoch,inverse_operator,lambda2,method='dSPM')
                src_fs = mne.read_source_spaces(fname_src_fsaverage)
                morph = mne.compute_source_morph(
                    inverse_operator['src'], subject_from=s, subjects_dir=subjects_dir,
                    src_to=src_fs, verbose=True)

                stc_fsaverage = morph.apply(stc[0])

                savepath = path_stc + '/stc_block_' + t +'_volume/'
                directory = os.path.dirname(savepath + s + '/')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                else:
                    pass
                stc_fsaverage.save(savepath + s + '/' + f + '_epoch_' + str(j)) 

                del stc_fsaverage


if __name__ == "__main__":
    ARGS = sys.argv[1:]
    compute_sources(ARGS[0])