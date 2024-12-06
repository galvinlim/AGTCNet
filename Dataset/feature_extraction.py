import numpy as np
import emd
from scipy import signal

from . import config

# (S, [B], [L], N, T) -> (S, [B], [L], N, T, F)

def feature_extraction(data, fs, algorithm=None):
    if algorithm == 'EMD': 
        max_imfs = config.emd.max_imfs

        data = emd_extraction(data, max_imfs) # (S, [B], [L], N, T, F=max_imfs)

    elif algorithm == 'EMD-HHT':
        max_imfs = config.emd_hht.max_imfs
        f_min = config.emd_hht.f_min
        f_max = config.emd_hht.f_max
        f_bins = config.emd_hht.f_bins

        data = emd_hht(data, fs, max_imfs, f_min, f_max, f_bins) # (S, [B], [L], N, T, F=f_bins)

    elif algorithm == 'STFT':
        window = config.stft.window
        window_size = config.stft.window_size
        window_step = config.stft.window_step
        scaling = config.stft.scaling

        data = stft(data, fs, window, window_size, window_step, scaling) # (S, [B], [L], N, T, F=f_bins)
        
    else:
        data = np.expand_dims(data, axis=-1) # (S, [B], [L], N, T, F=1)

    return data 

def emd_extraction(data, max_imfs):
    # (S, [B], [L], N, T) -> (S, [B], [L], N, T, F=max_imfs)
    data_shape = data.shape
    T = data.shape[-1]
    x = data.reshape((-1, T))

    data_imfs = []
    for i in range(x.shape[0]):
        # imf = emd.sift.sift(x[i], max_imfs=max_imfs, sift_thresh=1e-08)
        imf = emd.sift.mask_sift(x[i], max_imfs=max_imfs, sift_thresh=1e-08)
        # imf = emd.sift.sift(x[i], max_imfs=max_imfs, sift_thresh=1e-08, max_iter=15, iter_th=0.1)

        if imf.shape[-1] < max_imfs: # append trailing zeros
            imf = np.concatenate((imf, np.zeros((T, max_imfs - imf.shape[-1]))), axis=-1)
        data_imfs.append(imf)
    
    data_imfs = np.array(data_imfs)
    data_imfs = data_imfs.reshape((*data_shape, -1))

    return data_imfs

def emd_hht(data, fs, max_imfs, f_min, f_max, f_bins):
    # (S, [B], [L], N, T) -> (S, [B], [L], N, T, F=f_bins)
    data_shape = data.shape
    T = data.shape[-1]
    x = data.reshape((-1, T))

    freq_edges, _ = emd.spectra.define_hist_bins(f_min, f_max, f_bins, 'linear')

    data_hht = []
    for i in range(x.shape[0]):
        imf = emd.sift.mask_sift(x[i], max_imfs=max_imfs, sift_thresh=1e-08)
        IP, IF, IA = emd.spectra.frequency_transform(imf, fs, 'nht')
        f, hht = emd.spectra.hilberthuang(IF, IA, freq_edges, mode='amplitude', sum_time=False, sum_imfs=True) # (f_bins, T)
        data_hht.append(hht)

    data_hht = np.array(data_hht).swapaxes(-1,-2) # (S, T, f_bins)
    data_hht = data_hht.reshape((*data_shape, -1))

    return data_hht

def stft(data, fs, window, window_size, window_step, scaling):
    # (S, [B], [L], N, T) -> (S, [B], [L], N, T, F=f_bins)
    noverlap = window_size - window_step

    f, t, data_stft = signal.stft(data, fs=fs, window=window, nperseg=window_size, noverlap=noverlap, scaling=scaling, axis=-1)
    data_stft = data_stft.swapaxes(-1, -2)

    return data_stft