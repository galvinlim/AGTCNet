class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

dataset_BCICIV2A = Config(subject_exemption=[], # [] for None # A04T (lacks 2 EOG)
                          fs=250, scale=1, 
                          # 2s (Fixation Cross) | 1.25s (Cue) | 2.75 (MI Task Continue) | 1~2s (Break)
                          # [0s] -2s (prior to cue) to [7s] 5s (incl. 1.25s+2.75s + 1s-break) |  (7.0s) duration
                          tmin=1.5, tmax=6.0, # sec => 1.5s (-0.5s) to 6.0s (4.0s) [exactly after MI Task, NO BREAK]
                          baseline_tmin=5.0, # sec in 30.0s duration
                          ch_list=['Fz', 
                                   'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 
                                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
                                   'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4',
                                   'P1', 'Pz', 'P2', 
                                   'Poz'],
                          events=['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
                          fc_min=0.5, fc_max=100, fc_notch=50
                         )

EEGMMIDB_ch_grp = [['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6'],
                   ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
                   ['Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6'],
                   ['Fp1', 'Fpz', 'Fp2'],
                   ['Af7', 'Af3', 'Afz', 'Af4', 'Af8'],
                   ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
                   ['Ft7', 'Ft8'],
                   ['T7', 'T8', 'T9', 'T10'],
                   ['Tp7', 'Tp8'],
                   ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
                   ['Po7', 'Po3', 'Poz', 'Po4', 'Po8'],
                   ['O1', 'Oz', 'O2'],
                   ['Iz'] ]
dataset_EEGMMIDB = Config(subject_exemption=[38, 88, 89, 92, 100, 104], # [] for None
                          fs=160, scale=1e6, 
                          # [0s] -1s (prior to cue) to [5.5s] 4.5s
                          tmin=1.0, tmax=5.0, # sec => 1.0s (0.0s) to 5.0s (4.0s) [exactly start of MI Task]
                          baseline_tmin=5.0, # sec in 30.0s duration
                          ch_list=[item for sublist in EEGMMIDB_ch_grp for item in sublist],
                          ch_grp=EEGMMIDB_ch_grp,
                          events=['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet'],
                          #fc_min=, fc_max=, fc_notch=
                         )

baseline_crop = Config(step=0.25 # sec
                      )

emd = Config(max_imfs=8
            )

emd_hht = Config(max_imfs=8,
                 f_min=0, # >=0 | same as filterband.fc_min
                 f_max=120, # <=120 | same as filterband.fc_max
                 f_bins=32
                )

stft = Config(window='hamming', # 'hamming' | ('kaiser', beta) | ('gaussian', std)
              window_size=64, # dictates the FFT length = window_size // 2 + 1
              window_step=1, 
              scaling='spectrum' # 'spectrum': Zxx | 'psd': abs(Zxx)**2
             )