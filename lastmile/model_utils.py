""" Utility functions for working with model

"""
import os
import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import rawpy
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger(__name__)

def restore_model(mod: dict, model_name):
    save_dir = os.path.join(os.getcwd(), 'saved_models', model_name)
    if os.path.isdir(save_dir):
        latest = tf.train.latest_checkpoint(save_dir)

        model = mod.get('model', None)
        model.load_weights(latest)
        print(model.summary())
        return model

    else:
        logger.error("savedir not found: {}".format(save_dir))
        return None

def callbacks(model_type):
    """
    Callbacks handle model checkpointing. We can also use them
    to adjust learning rate.
    """
    # Prepare model, model saving directory
    
    save_dir = os.path.join(os.getcwd(), 'saved_models', model_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_name = '%s_model.{epoch:03d}-{val_loss:.2f}.ckpt' % model_type
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving (option to adjust lr)

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                 save_weights_only=True, period=1, verbose=1)
    callbacks = [checkpoint]

    return callbacks

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

