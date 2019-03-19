""" Train Imagenet

We want to see if we can generate useful parameters using data augmented
from ImageNet (http://www.image-net.org/) instead of from a professional
photographer with a fancy camera.  

Using CIFAR instead of imagenet for small example because it's
a default dataset in Keras.

Note that I've updated this model to use Keras instead of 
tensorflow.contrib.slim

We may move to kubeflow <https://www.kubeflow.org/>
"""
import glob
import logging
import os
import time

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from specify_imgnet_model import cifar_model, plot_costs
from augment_data import SimulateCondition
from logging_utils import enable_cloud_log


enable_cloud_log(level='DEBUG')
logger = logging.getLogger(__name__)


# Dataset of 50,000 32x32 color training images, 
# labeled over 10 categories, and 10,000 test images.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# validate basic model 

logger.info("Validating basic model")
parameters, costs, lr = cifar_model(X_train, X_train, X_test, X_test)


# now use a basic model with noise


# darkness or whatever


