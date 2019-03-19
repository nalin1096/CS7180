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

from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K

from specify_imgnet_model import debug_model
from augment_data import SimulateCondition
from logging_utils import enable_cloud_log


enable_cloud_log()
logger = logging.getLogger(__name__)


# Dataset of 50,000 32x32 color training images, 
# labeled over 10 categories, and 10,000 test images.

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# validate basic model 

logger.info("Validating basic model")
model = debug_model(x_train.shape)

logger.info("Compiling basic model without metrics")
model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['accuracy'])

logger.info("Fitting basic model without metrics")
#model.fit(x_train, x_train, epochs=1, batch_size=32)

logger.info("Evaluating against basic test set without metrics")
#loss_and_metrics = model.evaluate(x_test, x_test, batch_size=128)




# now use a basic model with noise


# darkness or whatever


