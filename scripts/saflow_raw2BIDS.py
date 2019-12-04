import mne_bids
import mne
import os
import os.path as op
import numpy as np
from mne_bids import make_bids_folders, make_bids_basename, write_raw_bids


BIDS_PATH = '/storage/Yann/saflow_DATA/saflow_bids'
ACQ_PATH = '/storage/Yann/saflow_DATA/acquisition'
EVENT_ID = {'Freq': 21, 'Rare': 31, 'Response': 99}

# check if BIDS_PATH exists, if not, create it
if not os.path.isdir(BIDS_PATH):
    os.mkdir(BIDS_PATH)
    print('BIDS folder created at : {}'.format(BIDS_PATH))
else:
    print('{} already exists.'.format(BIDS_PATH))

# list folders in acquisition folder
recording_folders = os.listdir(ACQ_PATH)

# loop across recording folders (folder containing the recordings of the day)
for rec_date in recording_folders: # folders are named by date in format YYYYMMDD
    for file in os.listdir(op.join(ACQ_PATH, rec_date)):
        # Create emptyroom BIDS if doesn't exist already
        if 'NOISE_noise' in file:
            if not op.isdir(op.join(BIDS_PATH, 'sub-emptyroom', 'ses-{}'.format(rec_date))):
                er_bids_basename = make_bids_basename(subject='emptyroom', session=rec_date)
                er_raw_fname = op.join(ACQ_PATH, rec_date, file)
                er_raw = mne.io.read_raw_ctf(er_raw_fname)
                write_raw_bids(er_raw, er_bids_basename, BIDS_PATH)
        # Rewrite in BIDS format if doesn't exist yet
        if 'SA' in file and '.ds' in file and not 'procedure' in file:
            subject = file[2:4]
            run = file[-5:-3]
            session = 'recording'
            if run == '01' or run == '08':
                task = 'RS'
            else:
                task = 'gradCPT'
            bids_basename = make_bids_basename(subject=subject, session=session, task=task, run=run)
            if not op.isdir(op.join(BIDS_PATH, 'sub-{}'.format(subject), 'ses-{}'.format(session), 'meg', bids_basename + '_meg.ds')):
                raw_fname = op.join(ACQ_PATH, rec_date, file)
                raw = mne.io.read_raw_ctf(raw_fname, preload=False)
                try:
                    events = mne.find_events(raw)
                    write_raw_bids(raw, bids_basename, BIDS_PATH, events_data=events, event_id=EVENT_ID, overwrite=True)
                except:
                    write_raw_bids(raw, bids_basename, BIDS_PATH, overwrite=True)




'''
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

for subject_id in SUBJ_LIST:
    subject = 'sub%03d' % subject_id
    for run in runs:
        raw_fname = op.join(data_path, repo, subject, 'MEG',
                            'run_%02d_raw.fif' % run)

        raw = mne.io.read_raw_fif(raw_fname)
        bids_basename = make_bids_basename(subject=str(subject_id),
                                           session='01', task='VisualFaces',
                                           run=str(run))
        write_raw_bids(raw, bids_basename, output_path, event_id=event_id,
                       overwrite=True)
'''

'''
What I want to do :

HAVE A SCRIPT THAT CREATES A BIDS STRUCTURE FOR EVERY RECORDING FOLDERS IN THE SPECIFIED
'acquisition' folder
Steps :
- check/create the destination folder (the one in BIDS format)
    - create protocole-specific info files ?
- get a list of each folder in acquisition
- get a list of the subjects present in each folder
- for each folder, extract the emptyroom recording and convert to BIDS
- for each subject, load every file and convert to BIDS

'''
