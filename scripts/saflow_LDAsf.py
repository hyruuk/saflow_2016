


SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
#SUBJ_LIST = ['14']
#BLOCS_LIST = ['5', '6']
BLOCS_LIST = ['1','2','3', '4', '5', '6']

### ML single subject classification of IN vs OUT epochs
# - single-features
# - CV k-fold (maybe 10 ?)
# - LDA, RF, kNN ?
for subj in SUBJ_LIST:
    all_freq_data = []
    for i, freq_name in enumerate(FREQS_NAMES):
        zonedata = []
        y = np.array([])
        for zone in ['IN', 'OUT']:
            blocdata = []
            for bloc in BLOCS_LIST:
                mat = loadmat(PSDS_DIR + 'SA' + subj + '_' + bloc + '_' + zone + '_' + freq_name + '.mat')
                data = mat['PSD']
                if blocdata == []:
                    blocdata = data
                else:
                    blocdata = np.hstack((blocdata, data))
            vstackif zone == 'IN':
                y = np.vstack((y, np.ones((blocdata.shape[1],1)))) ### 1 = IN
            else:
                y = np.vstack((y, np.zeros((blocdata.shape[1],1))))### 0 = OUT
            print(y.shape)
            
            if zonedata == []:
                zonedata = blocdata
            else:
                zonedata = np.hstack((zonedata, blocdata))
        if all_freq_data == []:
            all_freq_data = zonedata
        else:
            all_freq_data = np.vstack((all_freq_data, zonedata))
    print('SA' + subj)
    print(all_freq_data.shape)
                
                    
#### Résultat on veut : elec * freq X trials(IN+OUT) = 1890 X N_trials_tot

