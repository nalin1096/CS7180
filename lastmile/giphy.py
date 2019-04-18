""" Generate a Giph using different checkpoints. 

Predicting Raise 5881

"""
import logging
import os
import re

import cv2
import numpy as np

from model05 import functional_sony
from model_utils import enable_cloud_log, change_checkpoint, plot_images
from model_utils import restore_model
from custom_loss import mean_absolute_error
from image_preprocessing import (ImageDataPipeline, RaiseDataGenerator)


logger = logging.getLogger(__name__)
IMG_PATH = 'raise/rgb/test/5881.png'


def run_raise_giph(mod: dict, model_type):


    model_name = '{}_{}'.format(mod.get('model_id', ''), model_type)
    save_dir = os.path.join(os.getcwd(), 'saved_models', model_name)

    checkpoints = os.listdir(save_dir)

    # clean checkpoints

    cln = re.compile(".*.ckpt")
    cleaned = []
    for chkpt_name in checkpoints:

        if cln.match(chkpt_name):

            cln_name = cln.match(chkpt_name).group()
            cleaned.append(cln_name)

    checkpoints = list(set(cleaned))

    logger.info("Found '{}' checkpoints".format(len(checkpoints)))
    logger.info("Example checkpoint name: {}".format(checkpoints[0]))

    # Specify Image Data Pipeline

    idp = ImageDataPipeline(preprocessing_function='sony',
                            stride=32,
                            batch_size=32,
                            patch_size=(64,64),
                            random_seed=None,
                            meanm_fpath='simulation_mean.pkl',
                            covm_fpath='simulation_cov.pkl',
                            num_images=10
    )

    # Read file

    Y_test = cv2.imread(IMG_PATH)
    Y_patches = idp.extract_patches(Y_test, is_test=True)

    for checkpoint_name in checkpoints:

        change_checkpoint(checkpoint_name)
        model = restore_model(mod, model_name)


        y_pred_ij = []
        x_test_ij = []
        for Y_patch in Y_patches:

            # Apply relevant noise function

            X_patch = np.copy(Y_patch)
            X_patch = idp.prepfuncs[model_type](X_patch)
            x_test_ij.append(X_patch)

            # Predict against augmented X_patch

            y_pred = model.predict(np.expand_dims(X_patch, axis=0))
            y_pred_ij.append(y_pred)

        # Reconstruct X_test

        X_test_patches = np.array(x_test_ij)
        X_test = idp.reconstruct_patches(X_test_patches, Y_test.shape)

        # Reconstruct Y_pred

        Y_pred_patches = np.array(y_pred_ij)
        Y_pred = idp.reconstruct_patches(Y_pred_patches, Y_test.shape)

        # Write out image comparison

        review_dir = os.path.join(os.getcwd(), 'review')
        if not os.path.isdir(review_dir):
            os.makedirs(review_dir)

        model_chkpt_name = '{}_{}'.format(mod.get('model_id',''),
                                          checkpoint_name)
        mi_filepath = os.path.join(review_dir, model_chkpt_name)

        plot_images(mi_filepath, X_test, Y_pred, Y_test)

