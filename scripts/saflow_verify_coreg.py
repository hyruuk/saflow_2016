
import getpass
import mne
import os.path as op
import numpy as np
from mne.io import read_raw_fif
from mne.viz import plot_alignment
from nipype.utils.filemanip import split_filename as split_f
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab
from saflow_params import FOLDERPATH, FS_SUBJDIR
from utils import get_SAflow_bids

subj = '07'
run = '3'

raw_fname, raw_fpath = get_SAflow_bids(FOLDERPATH, subj=subj, run=run, stage='preproc_raw')
raw = read_raw_fif(raw_fpath, preload=True)


trans_fname, trans_fpath = get_SAflow_bids(FOLDERPATH, subj=subj, run=run, stage='epotrans')
trans_fpath = trans_fpath[:-3]

fig = plot_alignment(raw.info, trans=None, subject='SA' + str(subj),
                     dig=True, mri_fiducials=False, src=None,
                     coord_frame='head', subjects_dir=FS_SUBJDIR,
                     surfaces=['head', 'white'], meg='sensors')
print('*** Read trans file -> {}'.format(trans_fname))
trans = mne.read_trans(trans_fpath)
print(trans)
mri_head_t = mne.transforms.invert_transform(trans)
fig = plot_alignment(raw.info, trans=trans_fpath, subject='SA' + str(subj),
                     dig=True, mri_fiducials=True, src=None,
                     coord_frame='head', subjects_dir=FS_SUBJDIR,
                     surfaces=['head', 'white'], meg='sensors')
