""" Utility functions for working with model

"""
import os
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy


def enable_cloud_log(level='INFO'):
    """ Enable logs using default StreamHandler """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
    }
    logging.basicConfig(level=levels[level],
                        format='%(asctime)s %(levelname)s %(message)s')

def create_patch(X, ps=16):

    m, H, W, C, = X.shape
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    X_patch = X[:,yy:yy + ps, xx:xx + ps, :]

    return X_patch

def plot_loss(fpath, history):
    """ Plot loss vs. epochs """
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
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

class ImageDataGenerator(object):

    def __init__(self,
                 preprocessing_function=None,
                 stride=1,
                 patch_size=None,
                 target_size=None):
        
        self.preprocessing_function = preprocessing_function
        self.stride = stride
        self.patch_size = patch_size
        self.target_size = target_size
        

    def dirflow_sony(self, X_dir, Y_dir, batch_size=32):

        X_fnames = os.listdir(X_dir)
        Y_fnames = os.listdir(Y_dir)

        # Validate x/y
        
        pass

    def dirflow_raise(self, dirpath):
        dirs = os.listdir(dirpath)

        X, Y = np.array([])
        for i in range(batch_size):

            fpath = dirs.pop()

            # TODO: read RGB image in as a np array

            # TODO: use np.copy so we don't shallow copy

            # TODO: apply preprocessing function

        yield (X, Y)

        
        
