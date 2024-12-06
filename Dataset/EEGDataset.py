import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import spektral
from spektral.utils.convolution import add_self_loops
from spektral.utils.convolution import degree_power, normalized_adjacency, laplacian, normalized_laplacian, rescale_laplacian
import tensorflow as tf
from keras.utils import to_categorical

from . import config
from . import EEGDataloader as EEGDataloader
from .preprocessing import filter, notch_filter, resampling, normalizer, BatchNormScaler, dc_offset_remover
from .feature_extraction import feature_extraction

# DATASET = 'BCICIV2A' | 'EEGMMIDB'
# SUBJECT_RUN_MAP = {i: RUNS for i in SUBJECT}
# SUBJECT = range(1,2)
# RUNS = [False, True] | [4, 6, 8, 10, 12, 14]
# EVENTS = ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet', 'Tongue']
# ONE_HOT: Bool

# CH_SELECTION: None | [CH List]
# CH_ADJ = 'Defined' | 'Virtual' | '+Laplacian' | '+SelfLoop' | 'FullyConnected' , 'BCICIV2A' | 'EEGMMIDB'
# BASELINE: None | 'Fixed' | 'Varying'

# RESAMPLING: None | [fs_new, None] | w/ Anti-Aliasing LowPass = [fs_new, lowpass_filt_order = 8~16]
# DC_OFFSET_REMOVAL: None | 'mean' | 'mean-filter' | 'norm' | 'norm-filter'
# CH_REFERENCE: None | 'average' | 1CH

# FILTER: None | Lowpass: [None, fc_high, filt_order] | Highpass: [fc_low, None, filt_order] | Bandpass: [fc_low, fc_high, filt_order]
# NOTCH_FILTER: None | [fc_notch, Q_notch = fc_notch / BW_notch]

# FILTERBAND: None | [fc_min, fc_max, bw, fc_step, bp_order] 
# FEATURE_EXTRACTION: None | '[algorithm]': 'EMD' | 'EMD-HHT' | 'STFT'

# NORMALIZATION: None | 'batch' | 'channel' | 'time' | 'time-batch'
# NORM_SCALER: None | instance(BatchNormScaler) from TrainDataset

# SIGNAL_DURATION: None for Default Dataset tmin & tmax config | [tmin, tmax]
# SAMPLE_CROP: None | [duration, step]

# **kwargs = {'motor_type': 'MI' | 'ME, 'RQ': Bool, 'CQ': Bool, 'EQ': Bool} (for EmotivEpocFlex)

class EEGDataset(spektral.data.Dataset):
    def __init__(self, dataset, subject_run_map, events,
                 eeg_dataset_config, norm_scaler=None,
                 one_hot=True, dataset_path='RESOURCES',
                 optimization=True,
                 **kwargs):
        self.dataset = dataset
        self.subject_run_map = subject_run_map
        self.classes = events

        self.dataset_path = dataset_path

        self.ch_selection = eeg_dataset_config['ch_selection']
        self.ch_adj = eeg_dataset_config['ch_adj']
        self.baseline = eeg_dataset_config['baseline']
        self.resampling = eeg_dataset_config['resampling']
        self.dc_offset_removal = eeg_dataset_config['dc_offset_removal']
        self.ch_reference = eeg_dataset_config['ch_reference']
        self.filter = eeg_dataset_config['filter']
        self.notch_filter = eeg_dataset_config['notch_filter']
        self.filterband = eeg_dataset_config['filterband']
        self.feature_extraction = eeg_dataset_config['feature_extraction']
        self.normalization = eeg_dataset_config['normalization']
        self.sample_crop = eeg_dataset_config['sample_crop']
        
        self.norm_scaler = norm_scaler
        self.one_hot = one_hot

        self.optimization = optimization

        self.kwargs = kwargs

        self.dataset_config = getattr(config, f'dataset_{dataset}') 
        self.subject_exemption = self.dataset_config.subject_exemption
        self.fs = self.dataset_config.fs
        self.scale = self.dataset_config.scale
        self.tmin = eeg_dataset_config['signal_duration'][0] if eeg_dataset_config['signal_duration'] else self.dataset_config.tmin
        self.tmax = eeg_dataset_config['signal_duration'][1] if eeg_dataset_config['signal_duration'] else self.dataset_config.tmax
        self.baseline_tmin = self.dataset_config.baseline_tmin
        self.ch_list = self.dataset_config.ch_list.copy()
        self.duration = self.tmax - self.tmin
        
        super(EEGDataset, self).__init__()

    def read(self):
        load_data = getattr(EEGDataloader, f'load_{self.dataset}')
        
        def sliding_window_cropping(signals, labels, duration, step, precision=3):
            sig_duration = signals.shape[-1] / self.fs
            windows = np.array([[w_start, w_start + duration] for w_start in np.arange(0, sig_duration - duration, step)]).round(precision)

            cropped_signals = []
            for tmin, tmax in windows:
                cropped_signals.append(signals[..., int(tmin * self.fs) : int(tmax * self.fs)])
            cropped_signals = np.concatenate(cropped_signals, axis=0)

            if labels is not None:
                cropped_labels = np.concatenate([labels for _ in windows], axis=0)
                return cropped_signals, cropped_labels
            else:
                return cropped_signals

        def baseline_crop(baseline):
            return sliding_window_cropping(baseline, None, self.duration, config.baseline_crop.step)
        
        def baseline(signal, baseline):
            signal = np.expand_dims(signal, axis=1) # (S, 1, N, T)
            # (Baselines, N, T)
            if self.baseline == 'Fixed':
                baseline = baseline[:, :, int(self.baseline_tmin * self.fs) : int((self.baseline_tmin + self.duration) * self.fs)]
                baseline = np.stack([baseline for _ in range(signal.shape[0])], axis=0) # (S, Baselines, N, T)
                signal = np.concatenate([signal, baseline], axis=1)
            elif self.baseline == 'Varying':
                baseline = np.expand_dims(baseline, axis=0) # (1, Baselines, N, T)
                baseline = baseline_crop(baseline) # (Crops, Baselines, N, T)
                baseline = np.concatenate([baseline for _ in range(np.ceil(signal.shape[0] / baseline.shape[0]).astype(int))], axis=0) # (DupCrops, Baselines, N, T)
                signal = np.concatenate([signal, baseline[:signal.shape[0]]], axis=1)
            return signal # (S, L=3, N, T)
        
        def ch_select(signal):
            # (S, [L], N, T)
            ch_select_indices = [self.ch_list.index(ch) for ch in self.ch_selection]
            signal = np.stack([signal[..., i, :] for i in ch_select_indices], axis=-2)
            return signal
        
        def rereferencing(signal):
            # (S, [L], N, T)
            if self.ch_reference == 'average':
                if self.optimization:
                    signal = signal - np.mean(signal, axis=-2, keepdims=True)
                else: # rereferencing by Sample batch
                    for i in range(signal.shape[0]):
                        signal[i] -= np.mean(signal[i], axis=-2, keepdims=True)
            elif self.ch_reference in self.ch:
                ch_ref_idx = self.ch.index(self.ch_reference)
                signal = signal - np.expand_dims(signal[..., ch_ref_idx, :], axis=-2)
                signal = np.delete(signal, ch_ref_idx, axis=-2)
                self.ch.remove(self.ch_reference)
            else:
                raise ValueError('Invalid Reference Electrode')
            return signal
        
        def filter_band(X):
            # (S, [L], N, T)
            [fc_min, fc_max, bw, fc_step, bp_order] = self.filterband

            fc_bands = np.array([[fc, fc + bw] for fc in np.arange(fc_min, fc_max, fc_step)])

            X_bands = []
            for fc in fc_bands:
                X_filt = filter(X, fc, self.fs, bp_order)
                X_bands.append(X_filt)
            X_bands = np.array(X_bands).swapaxes(0, 1) # (B, S, [L], N, T) -> (S, B, [L], N, T)

            return X_bands
        
        def batch_norm(data, eps):
            if self.norm_scaler is None:
                # Training Dataset
                self.norm_scaler = BatchNormScaler(eps=eps)
                data = self.norm_scaler.fit_transform(data)
            else:
                # Validation Dataset
                data = self.norm_scaler.transform(data)
            return data
        
        def normalization(data, eps=1e-10):
            # (S, [B], [L], N, T, F)
            # Data normalization wrt time
            if self.normalization == 'batch':
                data = batch_norm(data, eps)
            elif self.normalization == 'channel':
                data = normalizer(data, axis=-3, eps=eps)
            elif self.normalization == 'time':
                data = normalizer(data, axis=-2, eps=eps)
            elif self.normalization == 'time-batch':
                data = normalizer(data, axis=-2, eps=eps)
                data = batch_norm(data, eps)
            return data

        def preprocessing(data):
            # (S, [L], N, T)
            data = self.scale * data # mV
            
            if self.ch_selection:
                data = ch_select(data) 
                self.ch = self.ch_selection
            else:
                self.ch = self.ch_list

            if self.resampling:
                [new_fs, lowpass_filt_order] = self.resampling
                if self.optimization:
                    data = resampling(data, new_fs, self.dataset_config.fs, lowpass_filt_order)
                else:
                    data_resampled = []
                    for i in range(data.shape[0]): # resampling by CH batch
                        data_resampled.append(resampling(data[i], new_fs, self.dataset_config.fs, lowpass_filt_order))
                    data = np.stack(data_resampled, axis=0)
                self.fs = new_fs
            data = dc_offset_remover(data, self.dc_offset_removal, self.fs) if self.dc_offset_removal else data
            data = rereferencing(data) if self.ch_reference else data

            data = filter(data, np.array(self.filter[0:2]), self.fs, self.filter[2]) if self.filter else data
            data = notch_filter(data, self.notch_filter[0], self.fs, self.notch_filter[1]) if self.notch_filter else data

            data = filter_band(data) if self.filterband else data # (S, [L], N, T) -> (S, B, [L], N, T)

            # (S, [B], [L], N, T) -> (S, [B], [L], N, T, F)
            data = feature_extraction(data, self.fs, algorithm=self.feature_extraction)

            data = normalization(data) if self.normalization else data
            
            return data

        def sample_crop(data, labels):
            # (S, [B], [L], N, T, F)
            self.duration = self.sample_crop[0]
            data = data.swapaxes(-2, -1) # (S, [B], [L], N, F, T)
            cropped_data, cropped_labels = sliding_window_cropping(data, labels, self.duration, self.sample_crop[1])
            cropped_data = cropped_data.swapaxes(-2, -1) # (S, [B], [L], N, T, F)
            return cropped_data, cropped_labels
        
        print('Data Loading ...')
        features = []
        labels = []
        for subject in self.subject_run_map:
            print(subject)
            if subject in self.subject_exemption:
                continue
            for run in self.subject_run_map[subject]:
                X, y, X_baseline = load_data(subject, run, events=self.classes, baseline=self.baseline, dataset_path=self.dataset_path, **self.kwargs)
                # X: (S, N, T)
                # y: (S, {0, 1, 2, 3})
                # X_baseline: (Baselines, N, T) | None for self.baseline=None
                
                if X is not None:
                    X = X[:, :, int(self.tmin * self.fs) : int(self.tmax * self.fs)]
                    
                    # Add Baseline Features
                    X = baseline(X, X_baseline) if self.baseline else X # (S, L=3, N, T) | (S, N, T)

                    features.append(X)
                    labels.append(y)

        features = np.concatenate(features, axis=0).astype(np.float64)
        labels = np.concatenate(labels, axis=0).astype(int)

        # (S, [L], N, T)
        print('Pre-Processing ...')
        features = preprocessing(features)

        # (S, [B], [L], N, T, F)
        if self.sample_crop:
            print('Sample Cropping ...')
            features, labels = sample_crop(features, labels)

        # (S, [B], [L], N, T, F)
        if self.baseline:
            features = np.moveaxis(features, -4, -2)
        # (S, [B], N, T, [L], F)

        # baseline = FALSE && filterband = FALSE --> (S,    N, T,      F)
        # baseline = TRUE  && filterband = FALSE --> (S,    N, T, L=3, F)
        # baseline = FALSE && filterband = TRUE  --> (S, B, N, T,      F)
        # baseline = TRUE  && filterband = TRUE  --> (S, B, N, T, L=3, F)
        
        if 'Virtual' in self.ch_adj[0]:
            axis = -3
            if self.baseline:
                axis = axis - 1
            features_mean = np.mean(features, axis=axis, keepdims=True)
            features = np.concatenate([features, features_mean], axis=axis)

        self.num_classes = np.unique(labels).size
        self.num_samples = features.shape[0]
        self.num_filterband = features.shape[1] if self.filterband else None
        self.num_channels = features.shape[-4] if self.baseline else features.shape[-3]
        self.num_sequence = features.shape[-3] if self.baseline else features.shape[-2]
        self.num_lines = features.shape[-2] if self.baseline else None
        self.num_feats = features.shape[-1]
        self.input_shape = features.shape[1:] # ([B], N, T, [L], F)

        # self.raw_features = features

        if self.filterband:
            features = features.swapaxes(1,2) # (S, [B], N, T, [L], F) -> (S, N, [B], T, [L], F)
        self.input_shape_ = features.shape[1:] # (N, [B], T, [L], F)

        features = features.reshape((*features.shape[:2], -1)) # (S, N, [B], T, [L], F) -> (S, N, [B]T[L]F)
        self.input_shape__ = features.shape[1:] # (N, [B]T[L]F)

        print('Data Completion ...')
        self.features = features
        self.labels = to_categorical(labels, num_classes=self.num_classes) if self.one_hot else labels

        self.adj = self.read_adj

        print('Dataset Compilation ...')
        return [spektral.data.Graph(x=feature, y=label, a=self.adj) 
                for feature, label in zip(self.features, self.labels)]

    @property
    def read_adj(self):
        if 'Defined' in self.ch_adj[0]:
            eeg_adj_path = self.dataset_path + '/eeg-adj-{:s}.csv'
            # eeg_adj = pd.read_csv(eeg_adj_path.format(64), index_col=0).fillna(0).loc[self.ch, self.ch]
            eeg_adj = pd.read_csv(eeg_adj_path.format(self.ch_adj[1]), index_col=0).fillna(0).loc[self.ch, self.ch].to_numpy()
        elif 'Virtual' in self.ch_adj[0]:
            eeg_adj_path = self.dataset_path + '/eeg-adj-{:s}.csv'
            # eeg_adj = pd.read_csv(eeg_adj_path.format(64), index_col=0).fillna(0).loc[self.ch, self.ch]
            eeg_adj = pd.read_csv(eeg_adj_path.format(self.ch_adj[1]), index_col=0).fillna(0).loc[self.ch, self.ch].to_numpy()
            node_adj = np.ones((self.num_channels, self.num_channels))
            node_adj[:self.num_channels - 1, :self.num_channels - 1] = eeg_adj
            node_adj[self.num_channels - 1, self.num_channels - 1] = 0 # Remove Self-Loop on Virtual
            eeg_adj = node_adj
        elif self.ch_adj[0] == 'FullyConnected':
            eeg_adj = np.ones((self.num_channels, self.num_channels)) - np.eye(self.num_channels)
        
        if 'SelfLoop' in self.ch_adj[0]:
            eeg_adj = add_self_loops(eeg_adj)
        
        if 'Laplacian' in self.ch_adj[0]:
            eeg_adj = laplacian(eeg_adj)
        
        return csr_matrix(eeg_adj)

    def load(self):
        return (self.features, np.array([self.adj.toarray() for _ in range (self.num_samples)])), self.labels
    
    # def load_raw(self):
    #     return (self.raw_features, np.array([self.adj.toarray() for _ in range (self.num_samples)])), self.labels
    
    def get(self, idx):
        return (self.features[idx], self.labels[idx]), self.adj

class EEGSubDataset(spektral.data.Dataset):
    def __init__(self, dataset, idx,
                 **kwargs):
        
        for attr_name in dir(dataset):
            if not attr_name.startswith('__') and not callable(getattr(dataset, attr_name)):
                if not hasattr(spektral.data.Dataset, attr_name):
                    attr_value = getattr(dataset, attr_name)
                    setattr(self, attr_name, attr_value)
        
        (self.features, self.labels), self.adj = dataset.get(idx)
        self.num_samples = self.features.shape[0]
        
        super(EEGSubDataset, self).__init__(**kwargs)

    def read(self):
        return [spektral.data.Graph(x=feature, y=label, a=self.adj) 
                for feature, label in zip(self.features, self.labels)]

    def load(self):
        return (self.features, np.array([self.adj.toarray() for _ in range (self.num_samples)])), self.labels