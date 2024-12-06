import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model, layers

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from Model.layers import Reshape
from .analyzer import model_performance_plot

class ModelWrapper(Model):
    def __init__(self, dataset):
        super(ModelWrapper, self).__init__()
        self.adj = dataset.adj
        self.classes = dataset.classes
        self.num_classes = dataset.num_classes
        self.B = dataset.num_filterband
        self.N = dataset.num_channels
        self.T = dataset.num_sequence
        self.L = dataset.num_lines
        self.F = dataset.num_feats
        self._input_shape = dataset.input_shape     # ([B], N, T, [L], F)
        self._input_shape_ = dataset.input_shape_   # (N, [B], T, [L], F)
        self._input_shape__ = dataset.input_shape__ # (N, [B]T[L]F)
        self.filterband = dataset.filterband

        self._T = self.T

        self.history_record = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [],
                               'mov_ave_accuracy': [], 'mov_ave_val_accuracy': [], 'mov_ave_loss': [], 'mov_ave_val_loss': [],
                               'mov_std_accuracy': [], 'mov_std_val_accuracy': [], 'mov_std_loss': [], 'mov_std_val_loss': []}

    # def call(self, batch):
    #     feats, graphs = batch 
    #     out = self.input_reshape(feats) # (S, N, [B]TF) -> (S, [B], N, T, F)
    #     return out
    
    def input_reshape(self, feats):
        feats = Reshape((-1, *self._input_shape_))(feats) # (S, N, [B], T, [L], F)

        if self.filterband: 
            feats = tf.einsum('SNB... -> SBN...', feats) # (S, N, [B], T, [L], F) -> (S, [B], N, T, [L], F)
            # feats = tf.experimental.numpy.swapaxes(feats, 1, 2)   
        
        return feats
    
    def T_size(self, kernel, strides):
        if strides is None:
            strides = kernel
        self._T = math.floor((self._T - kernel) / strides + 1)
        return
    
    def build_model(self, comprehensive_input=False):
        if comprehensive_input:
            feats = Input(shape=self._input_shape__) # (None, N, [B]T[L]F)
            graph = Input(shape=(1, self.N, self.N)) # (1, adj_row, adj_col)
            input = (feats, graph)
            output = self.call((feats, graph))
            return Model(inputs=input, outputs=output)
        else:
            input = Input(shape=self._input_shape__) # (None, N, [B]T[L]F)
            graph = self.adj.todense() # (adj_row, adj_col)
            graph = tf.expand_dims(graph, axis=0) # (1, adj_row, adj_col)
            output = self.call((input, graph))
            return Model(inputs=input, outputs=output)
        
    def plot(self, mov_ave=False, mov_std=False, grid_on=False, accuracy_metric='accuracy', loss_metric='loss'):
        self.history_record['accuracy']     = np.concatenate([self.history_record['accuracy'],      np.array(self.history.history[accuracy_metric])])
        self.history_record['val_accuracy'] = np.concatenate([self.history_record['val_accuracy'],  np.array(self.history.history['val_' + accuracy_metric])])
        self.history_record['loss']         = np.concatenate([self.history_record['loss'],          np.array(self.history.history[loss_metric])])
        self.history_record['val_loss']     = np.concatenate([self.history_record['val_loss'],      np.array(self.history.history['val_' + loss_metric])])

        if mov_ave:
            self.history_record['mov_ave_accuracy']     = np.concatenate([self.history_record['mov_ave_accuracy'],      np.array(self.history.history['mov_ave_' + accuracy_metric])])
            self.history_record['mov_ave_val_accuracy'] = np.concatenate([self.history_record['mov_ave_val_accuracy'],  np.array(self.history.history['mov_ave_val_' + accuracy_metric])])
            self.history_record['mov_ave_loss']         = np.concatenate([self.history_record['mov_ave_loss'],          np.array(self.history.history['mov_ave_' + loss_metric])])
            self.history_record['mov_ave_val_loss']     = np.concatenate([self.history_record['mov_ave_val_loss'],      np.array(self.history.history['mov_ave_val_' + loss_metric])])

        if mov_std:
            self.history_record['mov_std_accuracy']     = np.concatenate([self.history_record['mov_std_accuracy'],      np.array(self.history.history['mov_std_' + accuracy_metric])])
            self.history_record['mov_std_val_accuracy'] = np.concatenate([self.history_record['mov_std_val_accuracy'],  np.array(self.history.history['mov_std_val_' + accuracy_metric])])
            self.history_record['mov_std_loss']         = np.concatenate([self.history_record['mov_std_loss'],          np.array(self.history.history['mov_std_' + loss_metric])])
            self.history_record['mov_std_val_loss']     = np.concatenate([self.history_record['mov_std_val_loss'],      np.array(self.history.history['mov_std_val_' + loss_metric])])

        self.performance_plot = model_performance_plot(self.history_record, mov_ave=mov_ave, mov_std=mov_std, grid_on=grid_on)

        return self.performance_plot
    
    def test(self, *args, **kwargs):
        test_loss, test_accuracy = self.evaluate(*args, **kwargs)
        print('Test Loss: ', test_loss)
        print('Test Accuracy: ', test_accuracy)
        return test_loss, test_accuracy
    
    def confusion_matrix(self, dataset, AE=False, return_data=False):
        labels_round, test_predictions = self.pred(dataset, AE)
            
        cm = confusion_matrix(labels_round, test_predictions, normalize='all')
        
        if return_data:
            return cm
        
        else:
            cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)

            fig, ax = plt.subplots(figsize=(5,5))
            cmp.plot(cmap=plt.cm.Blues, ax=ax)

            return fig

    def classification_report(self, dataset, AE=False):
        labels_round, test_predictions = self.pred(dataset, AE)

        report = classification_report(labels_round, test_predictions, target_names=self.classes, output_dict=True)
        clsf_report = pd.DataFrame(report).T
        
        return clsf_report
    
    def pred(self, dataset, AE=False, pred_softmax=False):
        data, labels = dataset

        labels_round = np.argmax(labels, axis=1)
        
        if not AE:
            test_predictions = self.predict(data)
        else:
            test_predictions = self.predict(data)[0]
        
        test_predictions = test_predictions if pred_softmax else np.argmax(test_predictions, axis=1)

        return labels_round, test_predictions

class LayerWrapper(layers.Layer):
    def __init__(self, layer, *args, **kwargs):
        super().__init__(name=layer.name, *args, **kwargs)
        self.layer = layer

class GConvWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        feats, graph = inputs
        feats = tf.einsum('...NTF -> ...TNF', feats) # (S, [B], N, T, F) -> (S, [B], T, N, F)
        graph = tf.identity(graph)
        for _ in range(len(feats.shape) - 3):
            graph = tf.expand_dims(graph, axis=1) # (S, [B], T, N, N)
        out = self.layer([feats, graph], *args, **kwargs)
        out = tf.einsum('...TNF -> ...NTF', out) # (S, [B], T, N, F) -> (S, [B], N, T, F)
        return out

class ChPoolGPoolWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        inputs = tf.einsum('...NTF -> ...TNF', inputs) #(S, [B], N, T, F) -> (S, [B], T, N, F)
        out = self.layer(inputs, *args, **kwargs)
        return out

class ChPoolConvWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        inputs = tf.einsum('...NTF -> ...TNF', inputs) #(S, [B], N, T, F) -> (S, [B], T, N, F)
        in_shape = inputs.shape[1:-2] # ([B], T)
        inputs = Reshape((-1, *inputs.shape[-2:]))(inputs) # (S, [B], T, N, F) -> (S[B]T, N, F)
        out = self.layer(inputs, *args, **kwargs)
        out = Reshape((-1, *in_shape, out.shape[-1]))(out) # (S[B]T, F) -> (S, [B], T, F)
        return out
    
class ChPoolConvLSTMWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        inputs = tf.einsum('...NTF -> ...TNF', inputs) #(S, [B], N, T, F) -> (S, [B], T, N, F)
        in_shape = inputs.shape[1:-3] # ([B])
        inputs = Reshape((-1, *inputs.shape[-3:]))(inputs) # (S, [B], T, N, F) -> (S[B], T, N, F)
        out = self.layer(inputs, *args, **kwargs) # (S[B], T, 1, F)
        out = Reshape((-1, *in_shape, out.shape[-3], out.shape[-1]))(out) # (S[B], T, 1, F) -> (S, [B], T, F)
        return out

class ChAttnWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        inputs = tf.einsum('...NTF -> ...TNF', inputs) #(S, [B], N, T, F) -> (S, [B], T, N, F)
        in_shape = inputs.shape[1:-2] # ([B], T)
        inputs = Reshape((-1, *inputs.shape[-2:]))(inputs) # (S, [B], T, N, F) -> (S[B]T, N, F)
        out = self.layer(inputs, inputs, *args, **kwargs)
        out = Reshape((-1, *in_shape, *out.shape[-2:]))(out) # (S[B]T, N, F) -> (S, [B], T, N, F)
        out = tf.einsum('...TNF -> ...NTF', out) #(S, [B], T, N, F) -> (S, [B], N, T, F)
        return out
    
class TimeWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        in_shape = inputs.shape[1:-2] # ([B], [*N])
        inputs = Reshape((-1, *inputs.shape[-2:]))(inputs) # (S, [B], [*N], T, F) -> (S[B][*N], T, F)
        out = self.layer(inputs, *args, **kwargs)
        out = Reshape((-1, *in_shape, *out.shape[-2:]))(out) # (S[B][*N], T, F) -> (S, [B], [*N], T, F)
        return out

class ChWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        inputs = tf.einsum('...NTF -> ...TNF', inputs) #(S, [B], N, T, F) -> (S, [B], T, N, F)
        in_shape = inputs.shape[1:-2] # ([B], T)
        inputs = Reshape((-1, *inputs.shape[-2:]))(inputs) # (S, [B], T, N, F) -> (S[B]T, N, F)
        out = self.layer(inputs, *args, **kwargs)
        out = Reshape((-1, *in_shape, *out.shape[-2:]))(out) # (S[B]T, N, F) -> (S, [B], T, N, F)
        out = tf.einsum('...TNF -> ...NTF', out) #(S, [B], T, N, F) -> (S, [B], N, T, F)
        return out

class TimeCropsConvLSTMWrapper(LayerWrapper):
    def call(self, inputs, *args, **kwargs):
        in_shape = inputs.shape[1:-3] # ([B], [*N])
        inputs = Reshape((-1, *inputs.shape[-3:]))(inputs) # (S, [B], [*N], T, Tcrops, F) -> (S[B][*N], T, Tcrops, F)
        out = self.layer(inputs, *args, **kwargs) # (S[B][*N], T, Tcrops_new, F)
        out = Reshape((-1, *in_shape, *out.shape[-3:]))(out) # (S[B][*N], T, Tcrops_new, F) -> (S, [B], [*N], T, Tcrops, F)
        return out