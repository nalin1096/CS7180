""" Restore model from checkpoint. 

https://www.tensorflow.org/tutorials/keras/save_and_restore_models
"""
from datetime import datetime
import logging
import os
import cv2
import numpy as np

from model05 import functional_sony

import tensorflow as tf

from image_preprocessing import ImageDataPipeline


logger = logging.getLogger(__name__)

def restore_model(model_func, model_type):
    save_dir = os.path.join(os.getcwd(), 'saved_models', model_type)
    if os.path.isdir(save_dir):
        latest = tf.train.latest_checkpoint(save_dir)

        model = model_func.get('model', None)
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
#     X_test = tf.keras.preprocessing.image.img_to_array(image_path)
    X_test = cv2.imread(image_path)
    idp = ImageDataPipeline(preprocessing_function='sony',
                            patch_size=(64,64),
                            random_seed=42, 
                            meanm_fpath='simulation_mean.pkl',
                            covm_fpath='simulation_cov.pkl',
                            stride=16
    )
    
    prepfuncs = {'bl': idp.bl,
                 'bl_cd': idp.bl_cd,
                 'bl_cd_pn': idp.bl_cd_pn,
                 'bl_cd_pn_ag': idp.bl_cd_pn_ag
                }
    
#     for func_name, func in prepfuncs.items():
    print(X_test.shape)
#     X_test = func(X_test)
    cropped_X_test = idp.crop(X_test)
    print(np.array(cropped_X_test).shape)
    X_test_shape = cropped_X_test.shape
    patches = idp.extract_patches(X_test, is_test=True)
    print(np.array(patches).shape)
    y_pred_patches = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)
        y_pred_ij = model.predict(patch)
        y_pred_patches.append(y_pred_ij[0])

    y_pred_patches = np.array(y_pred_patches)
    X_pred = idp.reconstruct_patches(y_pred_patches, X_test_shape) 
    cv2.imwrite('./model05_results/img4_' +'.png', X_pred)
    print('Done!')
   
    return X_pred

mod = functional_sony()
model = restore_model(mod, 'sony_bl_cd_pn_ag')
_ = review_model(model, './Sony_RGB/Sony/short/00015_07_0.1s.png')
    
    



