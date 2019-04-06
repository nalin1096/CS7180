""" Training and test for model v0.2.3

This model runs on multiple gpus. Checkpointing. RGB input images.
"""
import logging
import pickle
from urllib.parse import urljoin

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model02 import model02
from model_utils import enable_cloud_log, plot_images, plot_loss
from custom_loss import mean_absolute_error

logger = logging.getLogger(__name__)

# Helper functions

def read_pickle(fpath):
    with open(fpath, "rb") as infile:
        m = pickle.load(infile)
    return m

# Image augmentation functions

def bl(image, sample=False):
    """ Apply black level """
    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    BL = int(sample[0])
    image[image < BL] = BL
    image = image - BL
    return image

def bl_cd(image, sample=False):
    """ Apply black level with color distortion """

    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    image = bl(image, sample)

    WB = [ sample[1], sample[2], sample[3] ]
    
    image[... ,0] = WB[0] * image[... ,0]
    image[... ,1] = WB[1] * image[... ,1]
    image[... ,2] = WB[2] * image[... ,2]

    return image

def bl_cd_pn(image, sample=False):
    """ Apply black level with color distortion and poisson noise. """
    
    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    noise_param = 10

    image = bl_cd(image, sample)

    noise = lambda x : np.random.poisson(x / 255.0 * noise_param) / \
        noise_param * 255

    func = np.vectorize(noise)
    image = func(image)
    return image

def bl_cd_pn_ag(image, sample=False):
    """ 
    Apply black level, color distortion, poisson noise, adjust gamma. 
    """

    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    image = bl_cd_pn(image, sample)
    image = image**sample[4]
    return image




# Prepare callbacks for model saving and for learning rate adjustment.

filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath)
callbacks = [checkpoint]
