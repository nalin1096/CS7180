""" Restore model from checkpoint. 

https://www.tensorflow.org/tutorials/keras/save_and_restore_models
"""
from datetime import datetime
import logging
import os

import tensorflow as tf

from image_preprocessing import ImageDataPipeline


logger = logging.getLogger(__name__)

def restore_model(mod, model_name):
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

def evaluate_model(test_dataflow, model, model_name):
    """ Evalute trained model against test set. """
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    review_dir = os.path.join(os.getcwd(), 'review')
    if not os.path.isdir(review_dir):
        os.makedirs(review_dir)

    logger.info("STARTED evaluating model: {}".format(model_name))
    evaluate = model.evaluate_generator(test_dataflow)
    test_eval = str(model.metrics_names) + str(evaluate)
    test_eval_name = 'test_eval_{}_{}.txt'.format(model_name, datetime_now)
    te_filepath = os.path.join(review_dir, test_eval_name)
    with open(te_filepath, "w") as outfile:
        outfile.write(test_eval)

def review_model(model, image_path: str):
    """ Predict an image, then stitch it together. """
    idp = ImageDataPipeline(preprocessing_function='sony',
                            patch_size=(64,64),
                            random_seed=42
    )
    X_test = idp.image_to_arr(image_path)
    patches = idp.extract_patches(X_test)
    y_pred_patches = []
    for patch in patches:

        y_pred_ij = model.predict(patch)
        y_pred_patches.append(y_pred_ij)

    return y_pred_patches
    
    



