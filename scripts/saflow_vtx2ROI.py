import mne
import numpy as np
import itertools
from saflow_params import FOLDERPATH, SUBJECTS_DIR, SUBJ_LIST, BLOCS_LIST


folderpath = FOLDERPATH
parc = 'aparc.a2009s'
sbj_dir = SUBJECTS_DIR

freq_bands = ['theta', 'alpha', 'beta', 'gamma']



for subj in SUBJ_LIST:
    if subj in ['06', '09']:
        sbj = 'sub-' + subj
    if subj == '07':
        sbj = 'SA07'
    else:
        sbj = 'fsaverage'

    for bloc in BLOCS_LIST:
        for cond in ['IN', 'OUT']:
            for freq in freq_bands:

                conmat_fname = folderpath + '/spectral_connectivity_wpli/ts_to_conmat/_cond_id_{}_freq_band_name_{}_run_id_run-0{}_session_id_ses-recording_subject_id_sub-{}/spectral/conmat_0_wpli.npy'.format(cond, freq, bloc, subj)
                conmat = np.load(conmat_fname)

                fwd = mne.read_forward_solution(folderpath + 'sub-{}/ses-recording/meg/sub-{}_ses-recording_task-gradCPT_run-0{}_meg_-epo-oct-6-fwd.fif'.format(subj, subj, bloc))

                labels_parc = mne.read_labels_from_annot(sbj, parc=parc,
                                                             subjects_dir=sbj_dir)

                src = fwd['src']
                vertno = [s['vertno'] for s in src]
                nvert = [len(vn) for vn in vertno]
                print('the src space contains {} spaces and {} points'.format(
                    len(src), sum(nvert)))
                print('the cortex contains {} spaces and {} points'.format(
                    len(src[:2]), sum(nvert[:2])))

                label_vertidx_cortex = list()
                label_name_cortex = list()

                for label in labels_parc:
                    if label.hemi == 'lh':
                        this_vertno = np.intersect1d(vertno[0], label.vertices)
                        vertidx = np.searchsorted(vertno[0], this_vertno)
                    elif label.hemi == 'rh':
                        this_vertno = np.intersect1d(vertno[1], label.vertices)
                        vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)

                    label_vertidx_cortex.append(vertidx)
                    label_name_cortex.append(label.name)

                nv_ROIs_cortex = [len(lab) for lab in label_vertidx_cortex]
                n_ROIs_cortex = len(label_vertidx_cortex)


                conmat_full = conmat + np.triu(conmat.T)
                conmat_ROIs = np.zeros((n_ROIs_cortex,n_ROIs_cortex))
                n_ROI = len(label_vertidx_cortex)
                for id_y, id_x in itertools.combinations(np.arange(n_ROI), 2):
                    print(id_y, id_x)
                    i_roi_x = label_vertidx_cortex[id_x]
                    i_roi_y = label_vertidx_cortex[id_y]

                    conmat_ROIs[id_x, id_y] = conmat_full[np.ix_(i_roi_x, i_roi_y)].mean()
                conmatROI_fname = folderpath + '/spectral_connectivity_wpli/ts_to_conmat/_cond_id_{}_freq_band_name_{}_run_id_run-0{}_session_id_ses-recording_subject_id_sub-{}/spectral/conmat_0_wpli_ROI.npy'.format(cond, freq, bloc, subj)
                np.save(conmatROI_fname, conmat_ROIs)
