import tensorflow as tf
import numpy as np

class SimpMovAve(tf.keras.callbacks.Callback):
    def __init__(self, metric, mov_window_size, mov_ave=True, mov_std=False):
        super(SimpMovAve, self).__init__()
        self.metric = metric
        self.mov_window_size = mov_window_size
        self.mov_ave = mov_ave
        self.mov_std = mov_std
        self.train = []
        self.valid = []

    def on_epoch_end(self, epoch, logs=None):
        self.train.append(logs.get(self.metric))
        self.valid.append(logs.get('val_' + self.metric))

        if len(self.train) > self.mov_window_size:
            self.train.pop(0)
            self.valid.pop(0)

        if len(self.train) == self.mov_window_size:
            if self.mov_ave:
                logs['mov_ave_' + self.metric] = np.mean(self.train)
                logs['mov_ave_val_' + self.metric] = np.mean(self.valid)
            if self.mov_std:
                logs['mov_std_' + self.metric] = np.std(self.train)
                logs['mov_std_val_' + self.metric] = np.std(self.valid)

class MaxMetric(tf.keras.callbacks.Callback):
    def __init__(self, train_metric, valid_metric):
        super(MaxMetric, self).__init__()
        self.train_metric = train_metric
        self.valid_metric = valid_metric
        self.max_train_metric = 0.0
        self.max_valid_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get(self.train_metric) is not None:
            self.max_train_metric = max(self.max_train_metric, logs.get(self.train_metric))
            logs['max_' + self.train_metric] = self.max_train_metric
            self.max_valid_metric = max(self.max_valid_metric, logs.get(self.valid_metric))
            logs['max_' + self.valid_metric] = self.max_valid_metric

class MinMetric(tf.keras.callbacks.Callback):
    def __init__(self, train_metric, valid_metric):
        super(MinMetric, self).__init__()
        self.train_metric = train_metric
        self.valid_metric = valid_metric
        self.min_train_metric = float('inf')
        self.min_valid_metric = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs.get(self.train_metric) is not None:
            self.min_train_metric = min(self.min_train_metric, logs.get(self.train_metric))
            logs['min_' + self.train_metric] = self.min_train_metric
            self.min_valid_metric = min(self.min_valid_metric, logs.get(self.valid_metric))
            logs['min_' + self.valid_metric] = self.min_valid_metric