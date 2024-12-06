import numpy as np
from keras.utils import to_categorical
import io, contextlib

from . import config
from . import EEGDataloader as EEGDataloader

def EEGDataset(dataset, subject_run_map, classes,
               eeg_dataset_config,
               one_hot=True, dataset_path='RESOURCES',
               **kwargs):
    
    load_data = getattr(EEGDataloader, f'load_{dataset}')
    dataset_config = getattr(config, f'dataset_{dataset}') 
    subject_exemption = dataset_config.subject_exemption
    fs = dataset_config.fs
    scale = dataset_config.scale
    tmin = eeg_dataset_config['signal_duration'][0] if eeg_dataset_config['signal_duration'] else dataset_config.tmin
    tmax = eeg_dataset_config['signal_duration'][1] if eeg_dataset_config['signal_duration'] else dataset_config.tmax
    duration = tmax - tmin

    features = []
    labels = []
    for subject in subject_run_map:
        print(subject)
        if subject in subject_exemption:
            continue
        for run in subject_run_map[subject]:
            with contextlib.redirect_stdout(io.StringIO()):
                X, y, _ = load_data(subject, run, events=classes, baseline=None, dataset_path=dataset_path, **kwargs)
            # X: (S, N, T)
            # y: (S, {0, 1, 2, 3})
            
            if X is not None:
                X = X[:, :, int(tmin * fs) : int(tmax * fs)]

                features.append(X)
                labels.append(y)

    # (S, N, T)
    X = np.concatenate(features, axis=0).astype(np.float64)
    y = np.concatenate(labels, axis=0).astype(int)

    X = scale * X # mV
    X = np.expand_dims(X, axis=-1) # (S, N, T, F=1)

    num_classes = np.unique(y).size

    y = to_categorical(y, num_classes=num_classes) if one_hot else y

    return X, y