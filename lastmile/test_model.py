""" Restore model from checkpoint. 

https://www.tensorflow.org/tutorials/keras/save_and_restore_models
"""
import logging
import os

import tensorflow as tf


logger = logging.getLogger(__name__)

def restore_model(model_func, model_type):
    save_dir = os.path.join(os.getcwd(), 'saved_models', model_type)
    if os.path.isdir(save_dir):
        latest = tf.train.latest_checkpoint(save_dir)

        model = model_func()
        model.load_weights(latest)
        print(model.summary())
        return model

    else:
        logger.error("savedir not found: {}".format(save_dir))
        return None

def review_model():
    """ Predict an image, then stitch it together. """
    pass
    



