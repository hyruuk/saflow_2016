import os.path as op
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio

import ephypype
from ephypype.nodes import create_iterator
from ephypype.datasets import fetch_omega_dataset

#base_path = op.join(op.dirname(ephypype.__file__), '..', 'examples')
#data_path = fetch_omega_dataset(base_path)
data_path = op.join('/storage/Yann/saflow_DATA/saflow_bids')

#### PARAMETERS
import json  # noqa
import pprint  # noqa
params = json.load(open("params.json"))

pprint.pprint({'experiment parameters': params["general"]})
subject_ids = params["general"]["subject_ids"]  # sub-003
session_ids = params["general"]["session_ids"]
run_ids = params["general"]["run_ids"]  # ses-0001
NJOBS = params["general"]["NJOBS"]

pprint.pprint({'inverse parameters': params["inverse"]})
spacing = params["inverse"]['spacing']  # ico-5 vs oct-6
snr = params["inverse"]['snr']  # use smaller SNR for raw data
inv_method = params["inverse"]['img_method']  # sLORETA, MNE, dSPM, LCMV
parc = params["inverse"]['parcellation']  # parcellation to use: 'aparc' vs 'aparc.a2009s'  # noqa
# noise covariance matrix filename template
noise_cov_fname = params["inverse"]['noise_cov_fname']

# set sbj dir path, i.e. where the FS folfers are
subjects_dir = op.join(data_path, params["general"]["subjects_dir"])

########

# workflow directory within the `base_dir`
src_reconstruction_pipeline_name = 'source_reconstruction_' + \
    inv_method + '_' + parc.replace('.', '')

main_workflow = pe.Workflow(name=src_reconstruction_pipeline_name)
main_workflow.base_dir = data_path

infosource = create_iterator(['subject_id', 'session_id', 'run_id'],
                             [subject_ids, session_ids, run_ids])
############

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=['raw_file', 'trans_file']),  # noqa
                     name='datasource')

datasource.inputs.base_directory = data_path
datasource.inputs.template = '*%s/%s/meg/%s_%s_task-gradCPT_%s_meg_%s.fif'

datasource.inputs.template_args = dict(
        raw_file=[['subject_id', 'session_id', 'subject_id', 'session_id', 'run_id', '-epo']],
        trans_file=[['subject_id', 'session_id', 'subject_id', 'session_id', 'run_id', 'epotrans']])

datasource.inputs.sort_filelist = True

###########
from ephypype.pipelines import create_pipeline_source_reconstruction  # noqa
event_id = {'Freq': 21, 'Rare': 31}
inv_sol_workflow = create_pipeline_source_reconstruction(
    data_path, subjects_dir, spacing=spacing, inv_method=inv_method, parc=parc,
    noise_cov_fname=noise_cov_fname, is_epoched=True, events_id={})

###########

main_workflow.connect(infosource, 'subject_id', datasource, 'subject_id')
main_workflow.connect(infosource, 'session_id', datasource, 'session_id')
main_workflow.connect(infosource, 'run_id', datasource, 'run_id')

##########

main_workflow.connect(infosource, 'subject_id',
                      inv_sol_workflow, 'inputnode.sbj_id')
main_workflow.connect(datasource, 'raw_file',
                      inv_sol_workflow, 'inputnode.raw')
main_workflow.connect(datasource, 'trans_file',
                      inv_sol_workflow, 'inputnode.trans_file')


##########
#main_workflow.write_graph(graph2use='colored')  # colored

#########
#import matplotlib.pyplot as plt  # noqa
#img = plt.imread(op.join(data_path, src_reconstruction_pipeline_name, 'graph.png'))  # noqa
#plt.figure(figsize=(8, 8))
#plt.imshow(img)
#plt.axis('off')

#########
main_workflow.config['execution'] = {'remove_unnecessary_outputs': 'false'}

# Run workflow locally on 1 CPU
main_workflow.run(plugin='MultiProc', plugin_args={'n_procs': NJOBS})

#########
import pickle  # noqa
from ephypype.gather import get_results  # noqa
from visbrain.objects import BrainObj, ColorbarObj, SceneObj  # noqa

time_series_files, label_files = get_results(main_workflow.base_dir,
                                             main_workflow.name,
                                             pipeline='inverse')

time_pts = 30

sc = SceneObj(size=(800, 500), bgcolor=(0, 0, 0))
lh_file = op.join(subjects_dir, 'fsaverage', 'label/lh.aparc.annot')
rh_file = op.join(subjects_dir, 'fsaverage', 'label/rh.aparc.annot')
cmap = 'bwr'
txtcolor = 'white'
for inverse_file, label_file in zip(time_series_files, label_files):
    # Load files :
    with open(label_file, 'rb') as f:
        ar = pickle.load(f)
        names, xyz, colors = ar['ROI_names'], ar['ROI_coords'], ar['ROI_colors']  # noqa
    ts = np.squeeze(np.load(inverse_file))
    cen = np.array([k.mean(0) for k in xyz])

    # Get the data of the left / right hemisphere :
    lh_data, rh_data = ts[::2, time_pts], ts[1::2, time_pts]
    clim = (ts[:, time_pts].min(), ts[:, time_pts].max())
    roi_names = [k[0:-3] for k in np.array(names)[::2]]

    # Left hemisphere outside :
    b_obj_li = BrainObj('white', translucent=False, hemisphere='left')
    b_obj_li.parcellize(lh_file, select=roi_names, data=lh_data, cmap=cmap)
    sc.add_to_subplot(b_obj_li, rotate='left')

    # Left hemisphere inside :
    b_obj_lo = BrainObj('white',  translucent=False, hemisphere='left')
    b_obj_lo.parcellize(lh_file, select=roi_names, data=lh_data, cmap=cmap)
    sc.add_to_subplot(b_obj_lo, col=1, rotate='right')

    # Right hemisphere outside :
    b_obj_ro = BrainObj('white',  translucent=False, hemisphere='right')
    b_obj_ro.parcellize(rh_file, select=roi_names, data=rh_data, cmap=cmap)
    sc.add_to_subplot(b_obj_ro, row=1, rotate='right')

    # Right hemisphere inside :
    b_obj_ri = BrainObj('white',  translucent=False, hemisphere='right')
    b_obj_ri.parcellize(rh_file, select=roi_names, data=rh_data, cmap=cmap)
    sc.add_to_subplot(b_obj_ri, row=1, col=1, rotate='left')

    # Add the colorbar :
    cbar = ColorbarObj(b_obj_li, txtsz=15, cbtxtsz=20, txtcolor=txtcolor,
                       cblabel='Intensity')
    sc.add_to_subplot(cbar, col=2, row_span=2)

sc.preview()
