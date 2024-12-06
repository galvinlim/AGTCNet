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

from .analyzer import model_performance_plot

class ModelWrapper(Model):
    def __init__(self, classes):
        super(ModelWrapper, self).__init__()

        self.classes = classes

        self.history_record = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [],
                               'mov_ave_accuracy': [], 'mov_ave_val_accuracy': [], 'mov_ave_loss': [], 'mov_ave_val_loss': [],
                               'mov_std_accuracy': [], 'mov_std_val_accuracy': [], 'mov_std_loss': [], 'mov_std_val_loss': []}
    
    def build_model(self, input_shape):
            input = Input(shape=input_shape) # (None, N, T, F)
            output = self.call(input)
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