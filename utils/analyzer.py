import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def graph_nodal_plot(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges)
    nx.draw(g, node_size=250, with_labels=True)
    plt.show()

def model_performance_plot(history, mov_ave=False, mov_std=False, grid_on=False, loss_ylim=None):
    epochs_range = np.arange(1, len(history['accuracy']) + 1)
    mov_ave_leading_None = [None] * (len(epochs_range) - len(history['mov_ave_accuracy']))

    alpha = 0.7
    
    fig = plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['accuracy'], 'c-', alpha=alpha, zorder=1, label='Training Accuracy')
    plt.plot(epochs_range, history['val_accuracy'],'y-', alpha=alpha, zorder=1, label='Validation Accuracy')
    if mov_ave:
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_ave_accuracy']]), 'b--', zorder=2, label='MovAve Training Accuracy')
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_ave_val_accuracy']]), 'r--', zorder=2, label='MovAve Validation Accuracy')
    if mov_std:
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_std_accuracy']]), 'g-.', zorder=3, label='Moving Std Training Accuracy')
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_std_val_accuracy']]), 'm-.', zorder=3, label='Moving Std Validation Accuracy')
    plt.legend(loc='best', fontsize=22)
    plt.xlabel('Epochs', fontsize=24)
    plt.title('Accuracy', fontsize=36)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.ylim(0, 1.05)
    plt.grid(grid_on, which='both', axis='both')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['loss'], 'c-', alpha=alpha, zorder=1, label='Training Loss')
    plt.plot(epochs_range, history['val_loss'],'y-', alpha=alpha, zorder=1, label='Validation Loss')
    if mov_ave:
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_ave_loss']]), 'b--', zorder=2, label='MovAve Training Loss')
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_ave_val_loss']]), 'r--', zorder=2, label='MovAve Validation Loss')
    if mov_std:
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_std_loss']]), 'g-.', zorder=3, label='Moving Std Training Loss')
        plt.plot(epochs_range, np.concatenate([mov_ave_leading_None, history['mov_std_val_loss']]), 'm-.', zorder=3, label='Moving Std Validation Loss')
    plt.legend(loc='best', fontsize=22)
    plt.xlabel('Epochs', fontsize=24)
    plt.title('Loss', fontsize=36)
    plt.tick_params(axis='both', which='major', labelsize=24)
    if loss_ylim is None:
        plt.ylim(bottom=0)
    else:
        plt.ylim(loss_ylim)
    plt.grid(grid_on, which='both', axis='both')

    # plt.show()
    return fig

def stats(data):
    mean = np.nanmean(data).round(4)
    std = np.nanstd(data).round(4)
    return '{:.4f} ± {:.4f}'.format(mean, std) if not np.isnan(mean) else 'SKIPPED'