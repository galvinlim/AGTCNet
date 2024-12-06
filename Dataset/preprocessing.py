import mne
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, iirnotch
from mne.filter import filter_data, notch_filter, resample

# (S, [L], N, T)
def filter(signal, fc, fs, filt_order, filt_type='butter', filt_filt=True, axis=-1):
    nyq_freq = 0.5 * fs

    if fc[0] is None:
        # Lowpass: [None, fc_high]  
        btype = 'lowpass'
        wn = fc[1] / nyq_freq
    elif fc[1] is None: 
        # Highpass: [fc_low, None]
        btype = 'highpass'
        wn = fc[0] / nyq_freq
    else:
        # Bandpass: [fc_low, fc_high]
        btype = 'bandpass'
        wn = fc / nyq_freq

    if filt_type == 'butter':
        b, a = butter(filt_order, wn, btype=btype, analog=False, fs=None)
        # b, a = butter(filt_order, fc, btype=btype, analog=False, fs=fs)
    
    assert filter_is_stable(a), "Filter should be stable..."

    if filt_filt:
        signal_filt = filtfilt(b, a, signal, axis=axis)
    else:
        signal_filt = lfilter(b, a, signal, axis=axis)

    return signal_filt

def notch_filter(signal, fc, fs, Q, axis=-1):
    b, a = iirnotch(fc, Q, fs)
    signal_filt = filtfilt(b, a, signal, axis=axis)
    return signal_filt

def filter_is_stable(a):
    # a: filter denominator coefficients
    assert a[0] == 1.0, (
        "Invalid filter denominator coefficients\n"
        "a: {:s}".format(str(a))
    )

    return np.all(np.abs(np.roots(a)) < 1) # All poles are inside the unit circle -> filter is stable.

def resampling(signal, fs_resampled, fs, lowpass_filt_order):
    decimination = fs_resampled / fs
    
    if decimination < 1 and lowpass_filt_order is not None: # downsampling 
        # Lowpass Filter - for Anti-Aliasing
        fc = [None, fs_resampled / 2]
        signal = filter(signal, fc, fs, lowpass_filt_order, filt_type='butter', filt_filt=True, axis=-1)

    signal_resampled = mne.filter.resample(signal, up=decimination, down=1.0, npad='auto', axis=-1, window='boxcar')
    return signal_resampled

def normalizer(data, axis, mean=True, std=True, eps=1e-10):
    if mean:
        data_mean = np.mean(data, axis=axis, keepdims=True)
    if std:
        data_std = np.std(data, axis=axis, keepdims=True)
    
    # data = (data - mean) / (std + eps)
    if mean:
        data = data - data_mean
    if std:
        data = data / (data_std + eps)
        
    return data

class BatchNormScaler():
    def __init__(self, eps=1e-10):
        super().__init__()

        self.eps = eps
    
    def fit_transform(self, data):
        # (S, [B], [L], N, T, F)
        # Data normalization wrt batch
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        data = (data - self.mean) / (self.std + self.eps)

        return data
    
    def transform(self, data):
        data = (data - self.mean) / (self.std + self.eps)

        return data

def dc_offset_remover(data, process, fs, axis=-1):
    fc_low = 0.16
    filt_order = 1
    
    if process == 'mean':
        data = normalizer(data, axis, mean=True, std=False)
    elif process == 'mean-filter':
        data = normalizer(data, axis, mean=True, std=False)    
        data = filter(data, [fc_low, None], fs, filt_order, filt_filt=True, axis=axis)
    elif process == 'norm':
        data = normalizer(data, axis, mean=True, std=True)
    elif process == 'norm-filter':
        data = normalizer(data, axis, mean=True, std=True)
        data = filter(data, [fc_low, None], fs, filt_order, filt_filt=True, axis=axis)
    
    return data