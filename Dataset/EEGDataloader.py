import mne
import numpy as np
import scipy.io as sio

# load_{DATASET_NAME}
# INPUT: (subject, run, config, baseline=None)
# X: (S, N, T)
# y: (S, {0, 1, 2, 3})
# X_baseline: (Baselines, N, T)

def load_BCICIV2A(subject, run, events=['Left Hand', 'Right Hand', 'Feet', 'Tongue'], baseline=None, all_trials=True, dataset_path='RESOURCES'):
    run_type = 'E' if run else 'T'
    baseline_runs = [1, 2] # 1: 2-min open eyes (looking at fixation cross on screen) | 2: 1-min closed eyes | 3: 1-min eye movements
    
    # 2s (Fixation Cross) | 1.25s (Cue) | 2.75 (MI Task Continue) | 1~2s (Break)
    # [0s] -2s (prior to cue) to [7s] 5s (incl. 1.25s+2.75s + 1s-break)
    window_length = 7 # sec

    baseline_tmin = 15 # sec
    baseline_tmax = 45 # sec
    fs = 250
    event_list = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    X_baseline = []
    X = []
    y = []

    path = dataset_path + f'/BCICIV2A/A{subject:02d}{run_type}.mat'
    raw_data = sio.loadmat(path)['data']
    runs = 10
    
    if subject == 4 and run_type == 'T':
        runs -= 2
    
    for run in range(1, runs):
        data = raw_data[0, run - 1][0, 0]
        data_X = data[0]
        data_trial = data[1]
        data_y = data[2]
        data_artifacts = data[5]

        if baseline and run in baseline_runs:
            X_baseline.append(np.transpose(data_X[int(baseline_tmin * fs) : int(baseline_tmax * fs), :22]))

        for trial in range(data_trial.size):
            if(bool(data_artifacts[trial]) and not all_trials):
                continue
            X.append(np.transpose(data_X[int(data_trial[trial]) : int(data_trial[trial] + window_length * fs), :22]))
            y.append(data_y[trial])
        
    X_baseline = np.array(X_baseline) if baseline else None
    X = np.array(X)
    y = np.array(y) - 1

    # Events Selection
    X, y = event_selection(X, y, events, event_list)

    return X, y, X_baseline

def load_EEGMMIDB(subject, run, events=['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet'], baseline=None, motor_type='MI', Epochs_proj=False, dataset_path='RESOURCES'):
    baseline_runs = [1, 2] # 1: 1-min open eyes | 2: 1-min closed eyes
    task1_runs = [3, 4, 7, 8, 11, 12] # physical (odd run) & imagined (even run) left_fist right_fist
    task2_runs = [5, 6, 9, 10, 13, 14] # physical (odd run) & imagined (even run) both_fists both_feet
    path = dataset_path + '/eeg-motor-movementimagery-dataset-1.0.0/files/S{:03d}/S{:03d}R{:02d}.edf'
    event_id = {'T1': 1, 'T2': 2}
    # [0s] -1s (prior to cue) to [5.5s] 4.5s
    tmin = -1.0 # sec
    tmax = 4.5 # sec
    baseline_tmin = 15 # sec
    baseline_tmax = 45 # sec
    fs = 160
    event_list = ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet']

    # Filter Runs
    if motor_type == 'MI' and run%2 != 0:
        return None, None, None
    if motor_type == 'ME' and run%2 != 1:
        return None, None, None
    if run in task1_runs and not('Left Fist' in events or 'Right Fist' in events):
        return None, None, None
    if run in task2_runs and not('Both Fists' in events or 'Both Feet' in events):
            return None, None, None

    raw = mne.io.read_raw_edf(path.format(subject, subject, run), preload=True)
    event, _ = mne.events_from_annotations(raw, event_id=event_id)
    picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False, stim=False, exclude='bads')
    epoched = mne.Epochs(raw, event, event_id=event_id, tmin=tmin, tmax=tmax, proj=Epochs_proj, picks=picks, baseline=None, preload=True)
    X = np.concatenate([epoched['T1'].get_data(copy=False), epoched['T2'].get_data(copy=False)], axis=0)
    y = np.concatenate([[1 for _ in range(len(epoched['T1']))], [2 for _ in range(len(epoched['T2']))]], axis=0)
    y = y + 2 if run in task2_runs else y

    if baseline:
        X_baseline = []
        for run in baseline_runs:
            raw = mne.io.read_raw_edf(path.format(subject, subject, run), preload=True)
            raw.crop(tmin=baseline_tmin, tmax=baseline_tmax)
            X_baseline.append(raw.get_data())
        X_baseline = np.stack(X_baseline)
    else:
        X_baseline = None
    
    X = np.array(X)
    y = np.array(y) - 1

    # Events Selection
    X, y = event_selection(X, y, events, event_list)

    return X, y, X_baseline

def event_selection(X, y, events, event_list):
    # Events Selection
    selected_events = [event_list.index(event) for event in events]
    selected_events_mapping = {value: index for index, value in enumerate(selected_events)}

    selected_samples = np.where(np.isin(y, selected_events))[0]
    X = X[selected_samples]
    y = y[selected_samples]
    y = np.vectorize(selected_events_mapping.get)(y) # Map to 'events' sequence

    return X, y
