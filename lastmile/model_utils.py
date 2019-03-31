""" Utility functions for working with model

"""
import logging

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

