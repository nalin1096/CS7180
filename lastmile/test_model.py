""" Restore model from checkpoint. 

https://www.tensorflow.org/tutorials/keras/save_and_restore_models
"""
import os

import tensorflow as tf

def restore_model(model_func):
    save_dir = os.path.join(os.getcwd(), 'saved_models', model_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    latest = tf.train.latest_checkpoint(save_dir)

    model = model_func()
    model.load_weights(latest)

    return model
    



