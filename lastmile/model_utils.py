""" Utility functions for working with model

"""
import logging

import matplotlib.pyplot as plt
import numpy as np


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

def plot_imgpair(Y_pred, Y_true, name):
    """ Show the predicted and true images side by side. """
    
    fig, (ax1, ax2) = plt.subplots(1,2,sharey=True)
    ax1.matshow(Y_pred[0,...])
    ax1.set_title('Y_predict')
    ax2.matshow(Y_true[0,...])
    ax2.set_title('Y_true')

    plt.savefig(name)
    
