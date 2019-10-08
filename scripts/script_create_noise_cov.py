# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:19:48 2017

@author: pasca
"""

import os.path as op
import glob
#import locale

from mne.io import read_raw_ctf
from mne import compute_raw_covariance, pick_types, write_cov

from nipype.utils.filemanip import split_filename as split_f

from ephypype.preproc import create_reject_dict

from params import subject_ids, data_path, noise_path


print('*** COMPUTE RAW COV ***')

#locale.setlocale(locale.LC_ALL, "en_US.utf8")
noise_fname_template = 'lyon_Noise*.ds'
for sbj in subject_ids:
    print('\nsbj %s \n' %sbj)
    ds_path = op.join(noise_path, sbj, '*misc', noise_fname_template)
    
    for noise_ds_fname in glob.glob(ds_path):
        print('*** Empty room data file %s found!!!\n' % noise_ds_fname)

    if not op.isdir(noise_ds_fname):
        raise RuntimeError('*** Empty room data file %s NOT found!!!'
                           % noise_ds_fname)
                           
    raw = read_raw_ctf(noise_ds_fname)
    
    noise_fname = sbj + '_' + noise_fname_template.replace('*.ds', '-raw.fif')
    noise_fif_fname = op.join(data_path, sbj,  noise_fname)
    print('*** Raw fif file name   %s ***\n' % noise_fif_fname)
    raw.save(noise_fif_fname, overwrite=True)
        
    noise_cov_fname = noise_fif_fname.replace('-raw.fif', '-raw-cov.fif')
    print('*** Noise Cov file name   %s ***\n' % noise_cov_fname)
    
    reject = create_reject_dict(raw.info)
    picks = pick_types(raw.info, meg=True, ref_meg=False, exclude='bads')

    noise_cov = compute_raw_covariance(raw, picks=picks, reject=reject)

    write_cov(noise_cov_fname, noise_cov)

