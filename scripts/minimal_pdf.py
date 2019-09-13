import os
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs


filepath = '/home/karim/DATA/DATAmeg_gradCPT/20190820/SA11_SAflow-yharel_20190820_02.ds'
pattern = 'test01_raw.fif.gz'
path = os.getcwd()

raw = mne.io.read_raw_ctf(filepath)
raw.save(pattern)


report = mne.Report(verbose=True)
report.parse_folder(path, pattern='*raw.fif', render_bem=False)



ica = ICA(n_components=20, random_state=0).fit(raw, decim=3)

## FIND ECG COMPONENTS
fmax = 40. ## correlation threshold for ICA components (maybe increase to 40. ?)
ecg_epochs = create_ecg_epochs(raw_data, ch_name='EEG059')
ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name='EEG059')
fig = ica.plot_scores(ecg_scores, ecg_inds);

report.add_figs_to_section(fig, captions='Left Auditory', section='ICA')

report.save('report_basic.html')
