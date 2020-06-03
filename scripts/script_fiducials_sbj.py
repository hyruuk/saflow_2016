import numpy as np
import os.path as op
from os import error
import nibabel as nib
import h5py

from mne.io import read_raw_ctf
from mne.io.constants import FIFF
from mne.io import write_fiducials
from smri_params import sbj_dir, subjects_list, MRI_path


subjects_list = subjects_list
sessions = ['Open', 'Closed']
data_path = '/media/karim/DATA2/Schizophrenia_Project/SZ_data_raw'
new_data_path = '/media/karim/DATA2/Schizophrenia_Project/SZ_data_ica_600Hz_MRI_individual'


for sbj_FS in subjects_list:
    for sess in sessions:

sbj_FS = '07'
run = 2
sess = run
        data_name = sbj_FS + '_' + sess + '.ds'

        if sbj_FS.startswith('Control'):
            ds_filename = data_name
            sbj_folder = sbj_FS
        elif sbj_FS.startswith('Patient'):
            ind = data_name.find('_')
            ds_filename = 'NEW' + data_name[:9].upper() + data_name[9:]
            sbj_folder = 'NEW' + sbj_FS.upper()
        # Convert ds to fif
        raw_ctf_path = op.join(data_path, sbj_folder, ds_filename)
        # print(raw_ctf_path)


        raw_fname, raw_fpath = get_SAflow_bids(FOLDERPATH, subj=sbj_FS, run=sess, stage='preproc_raw')
        raw_ctf_path = raw_fpath
        if op.exists(raw_ctf_path):
            print('*** Read raw filename -> {}'.format(ds_filename))
            raw = read_raw_ctf(raw_ctf_path)

            # Save fif
            raw_fif_path = op.join(new_data_path, sbj_FS,
                                   data_name.replace('.ds', '-raw.fif'))
            raw.save(raw_fif_path, overwrite=True)

            # Load sbj MRI volume and read the vox2ras matrix transformation
            mri_fname = op.join(FS_SUBJDIR, 'SA' + str(sbj_FS), 'mri/T1.mgz')
            if op.exists(mri_fname):
                print('*** Read MRI filename -> {}'.format(mri_fname))
                img = nib.load(mri_fname)
                vox2ras = img.header.get_vox2ras()
                print(vox2ras)

                # Read the fiducials points from the shape_info file of sbj_FS
                shape_dir = '/media/karim/Seagate Backup Plus Drive/Schizophrenia project/AnonymisedShapes/AnonymisedShapesFiles'  # TODO
                n_sbj = sbj_FS[-2:]
                if sbj_FS[-2] == '0':
                    new_sbj_FS = sbj_FS[:-2] + sbj_FS[-1]
                    n_sbj = sbj_FS[-1]
                else:
                    new_sbj_FS = sbj_FS

                if sbj_FS.startswith('Control'):
                    shape_info_file = op.join(
                            shape_dir, new_sbj_FS.upper().
                            replace(n_sbj, 'S_' + n_sbj) + '.shape_info')
                elif sbj_FS.startswith('Patient'):
                    shape_info_file = op.join(
                            shape_dir, new_sbj_FS.upper().
                            replace(n_sbj, '_' + n_sbj) + '.shape_info')

                print('*** Read shape_info_file: {} \n'.format(shape_info_file))

                # CTF MRI coo system
                # copy fiducials from shape_info in the vectors nasion,
                # left_ear, right_ear
                file = open(shape_info_file, 'r')
                print('fiducial points in CTF MRI coo system')
                for line in file:
                    # print line,
                    for part in line.split():
                        if "NASION" in part:
                            row = line.split()
                            naison = [float(row[1]), float(row[2]), float(row[3])]
                            print('NASION: {} '.format(naison))
                        if "LEFT_EAR" in part:
                            row = line.split()
                            left_ear = [float(row[1]), float(row[2]), float(row[3])]
                            print('LEFT_EAR: {} '.format(left_ear))
                        if "RIGHT_EAR" in part:
                            row = line.split()
                            right_ear = [float(row[1]), float(row[2]), float(row[3])]
                            print('RIGHT_EAR: {} '.format(right_ear))

                sbj_mat = sbj_FS.replace(sbj_FS[-2:], '00' + sbj_FS[-2:])
                MRI_mat_filename = op.join(MRI_path, sbj_mat + '.mat')
                print('*** Read matlab file: {} \n'.format(MRI_mat_filename))
                hf = h5py.File(MRI_mat_filename, 'r')
                nas = hf['hdr/fiducial/mri/nas'][()]
                lpa = hf['hdr/fiducial/mri/lpa'][()]
                rpa = hf['hdr/fiducial/mri/rpa'][()]
                print('nas {}'.format(nas.T))
                print('lpa {}'.format(lpa.T))
                print('rpa {}'.format(rpa.T))

                if not np.alltrue((nas, np.array(naison)[:, np.newaxis])) | \
                    np.alltrue((lpa, np.array(left_ear)[:, np.newaxis])) | \
                        np.alltrue((rpa, np.array(right_ear)[:, np.newaxis])):

                    error('something wrong!')

                # Convert from CTF MRI coo system to Freesurfer MRI coo system
                nas_fs = [255 - nas[0], nas[2], 255 - nas[1], 1]
                lpa_fs = [255 - lpa[0], lpa[2], 255 - lpa[1], 1]
                rpa_fs = [255 - rpa[0], rpa[2], 255 - rpa[1], 1]

                print('fiducial points in FS MRI coo system\n')
                RAS_nas = vox2ras.dot(nas_fs)
                print('NASION:  {}'.format(RAS_nas))
                RAS_lpa = vox2ras.dot(lpa_fs)
                print('LEFT_EAR: {} '.format(RAS_lpa))
                RAS_rpa = vox2ras.dot(rpa_fs)
                print('RIGHT_EAR: {}'.format(RAS_rpa))

                # FIFF.FIFFV_POINT_LPA = 1
                # FIFF.FIFFV_POINT_NASION = 2
                # FIFF.FIFFV_POINT_RPA = 3

                # Write new fiducial points to fsaverage-fiducials
                fiducials = [{'kind': FIFF.FIFFV_POINT_CARDINAL,
                              'ident': FIFF.FIFFV_POINT_LPA,
                              'r': np.float32(RAS_lpa)/1000},
                             {'kind': FIFF.FIFFV_POINT_CARDINAL,
                              'ident': FIFF.FIFFV_POINT_NASION,
                              'r': np.float32(RAS_nas)/1000},
                             {'kind': FIFF.FIFFV_POINT_CARDINAL,
                              'ident': FIFF.FIFFV_POINT_RPA,
                              'r': np.float32(RAS_rpa)/1000}]

                fiducials_file = op.join(sbj_dir, sbj_FS, 'bem',
                                         sbj_FS + '-fiducials.fif')
                write_fiducials(fiducials_file, fiducials, FIFF.FIFFV_COORD_MRI)

                print('fsaverage fiducial points \n {} saved in \n {}'.format(
                        fiducials, fiducials_file))
            else:
                print('!!! No Freesurfer folder for {}'.format(sbj_FS))
        else:
            print('!!! No DS folder for {}'.format(sbj_FS))
