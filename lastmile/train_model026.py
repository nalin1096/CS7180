""" Training only. Separate file for evaluation, prediction, reporting.

"""
from datetime import datetime
import json
import os
import logging
from urllib.parse import urljoin

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from model05 import functional_sony
from model_utils import enable_cloud_log
from custom_loss import mean_absolute_error
from image_preprocessing import (ImageDataPipeline, RaiseDataGenerator)
                           

logger = logging.getLogger(__name__)


##################################################
# Model fitting, prediction, and review functions
##################################################

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

def fit_model(train_dataflow, val_dataflow, mod, imgproc, lr, epochs):
    """ Fits model, Returns model and history keras objects. """

    # Define model

    model = mod.get('model', None)
    model_id = mod.get('model_id', None)

    if model is None:
        raise TypeError("model must be defined: {}".format(model))

    # Compile model
    
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss=mean_absolute_error,
                  metrics=['accuracy'])

    # Fit model

    model_type = '{}_{}'.format(model_id, imgproc)
    calls = callbacks(model_type=model_type)
    history = model.fit_generator(
        generator=train_dataflow,
        epochs=epochs,
        callbacks=calls,
        validation_data=val_dataflow
    )

    return model, history

def run_simulation(mod: dict):
    """ Run simulation using N GPUs """
    logger.info("STARTED running simulations")

    #imgnames = ['bl', 'bl_cd', 'bl_cd_pn', 'bl_cd_pn_ag']
    imgnames = ['bl_cd_pn_ag']

    # Run model on each data augmentation scenario

    for imgproc in imgnames:
    
        # Specify Image Data Pipeline

        idp = ImageDataPipeline(preprocessing_function=imgproc,
                                stride=256,
                                batch_size=64,
                                patch_size=(512,512),
                                random_seed=42,
                                meanm_fpath='simulation_mean.pkl',
                                covm_fpath='simulation_cov.pkl',
                                num_images=10
        )

        # Specify train/val generators

        train_dir = 'raise/rgb/train/'
        y_train_set = [urljoin(train_dir, f) for f in os.listdir(train_dir)]
        train_dataflow = RaiseDataGenerator(y_train_set, idp)

        val_dir = 'raise/rgb/val/'
        y_val_set = [urljoin(val_dir, f) for f in os.listdir(val_dir)]
        val_dataflow = RaiseDataGenerator(y_val_set, idp)

        # Define model

        model = mod.get('model', None)
        model_id = mod.get('model_id', None)

        if model is None or model_id is None:
            raise TypeError("model or model_id not defined: {}".\
                            format(model_id))
        
        logger.info("Processing model: {}".format(imgproc))

        # Fit model

        model, history = fit_model(train_dataflow, val_dataflow, mod,
                                   imgproc, lr=1e-3, epochs=100)

        # Save history

        try:
            review_dir = os.path.join(os.getcwd(), 'review')
            if not os.path.isdir(review_dir):
                os.makedirs(review_dir)

            model_history_name = '{}_{}.json'.format(model_id, imgproc)
            datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
            mh_filepath = os.path.join(review_dir, model_history_name)

            with open(mh_filepath, "w") as outfile:
                json.dump(history.history, outfile)

            logger.info("Saved model history: {}".format(model_history_name))

        except Exception as exc:
            logger.exception(exc)

        
        # Reset model and history

        model = None
        history = None

    logger.info("FINISHED running simulations")

#################################################
# Main functions for each hardware configuration
#################################################

def main():
    """ Main function to run training and prediction. """

    mod = functional_sony()
    run_simulation(mod)

#######################################################
# Running train_model script, Jupyter Notebook config
#######################################################

enable_cloud_log('INFO')
main()

