""" Utility functions for working with model

"""
import os
import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import rawpy
from tensorflow.keras.preprocessing import image
from numpy.lib.stride_tricks import as_strided

def enable_cloud_log(level='INFO'):
    """ Enable logs using default StreamHandler """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
    }
    logging.basicConfig(level=levels[level],
                        format='%(asctime)s %(levelname)s %(message)s')

def create_patch(X, ps=32):

    m, H, W, C, = X.shape
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    X_patch = X[:,yy:yy + ps, xx:xx + ps, :]

    return X_patch

def plot_loss(fpath, history):
    """ Plot loss vs. epochs """
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig(fpath, format='png')

def plot_images(name, X_test, Y_pred, Y_true):
    tmp = np.concatenate((X_test, Y_pred, Y_true), axis=1)
    tmp = tmp.astype(np.uint8)
    plt.imsave(name, tmp)

def plot_imgpair(Y_pred, Y_true, name):
    """ Show the predicted and true images side by side. """
    tmp = np.concatenate((Y_pred, Y_true), axis=1)
    tmp = tmp.astype(np.uint8)
    plt.imsave(name, tmp)

