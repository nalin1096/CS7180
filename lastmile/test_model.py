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
from model_utils import plot_images


logger = logging.getLogger(__name__)
PIXEL_MAX = 255.0

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

def compute_mae(Y_pred_i, Y_test_i):
    return np.mean(np.abs(Y_pred_i - Y_test_i), axis=None)

def compute_psnr(Y_pred_i, Y_test_i):
    mse = np.mean((Y_pred_i - Y_test_i)**2)
    if mse == 0:
        return 100
    return 20 * np.log10(PIXEL_MAX/ np.sqrt(mse))

def custom_evaluate_sony(test_dataflow, sony_txt, model, model_name, idp):
    """ Custom evaluation function. 

    Args:
       test_dataflow: test generator
       sony_txt: file path to sony test file
       model: dict, {'model': keras.Model, 'model_id': <model_id_str>}
       model_name: full model name for reporting

    WARNING: all images will be brought into memory and copied.
             space is (k * 2 * m * n) where k is the number of
             images in the test set.

    Predict image then run evaluation metrics.
    Write the image, X_test, and Y_test, to a file.

    Evaluation metrics:
      MAE
      PSNR
    """
    
    # Store eval metrics

    store_mae = []
    store_psnr = []
    
    Y_pred = model.predict_generator(test_dataflow)
    Y_pred = [(y_pred * (255.0/y_pred.max())).astype(int)
              for y_pred in Y_pred]

    idx = 0
    pairs = idp.parse_sony_list(sony_txt)
    
    for x_filepath, y_filepath in pairs:

        # Load y_test

        y_test = cv2.imread(y_filepath)

        # Extract y_test patch

        y_test_i = idp.extract_patches(y_test)

        # Score
        
        score_mae = compute_mae(Y_pred_i=Y_pred[idx], Y_test_i=y_test_i)
        store_mae.append((y_filepath, score_mae))
                         
        score_psnr = compute_psnr(Y_pred_i=Y_pred[idx], Y_test_i=y_test_i)
        store_psnr.append((y_filepath, score_psnr))

        idx += 1

    return store_mae, store_psnr


def review_images(sony_txt, idp, model, model_type):
    """ Review X_test, Y_pred, Y_true """

    idx = 0
    pairs = idp.parse_sony_list(sony_txt)

    for x_filepath, y_filepath in pairs:

        # Load X_test, Y_test
        
        X_test = cv2.imread(x_filepath)
        Y_test = cv2.imread(y_filepath)

        # Generate all patches

        y_test_ij = idp.extract_patches(Y_test, is_test=True)
        x_test_ij = idp.extract_patches(X_test, is_test=True)

        # Run predictions against each patch

        y_pred_ij = []
        for i, x_test_i, in enumerate(x_test_ij):
            y_pred = model.predict(np.expand_dims(x_test_i, axis=0))
            y_pred_ij.append(y_pred[0])

        # Reconstruct each image

        Y_pred_patches = np.array(y_pred_ij)
        Y_pred = idp.reconstruct_patches(Y_pred_patches, X_test.shape)

        # Write out image comparison

        review_dir = os.path.join(os.getcwd(), 'review')
        if not os.path.isdir(review_dir):
            os.makedirs(review_dir)

        model_id = 'freeze_sony'
        model_name = '{}_{}'.format(model_id, model_type)

        datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        image_id = y_filepath.split('/')[-1]
        model_image_name = '{}_{}'.format(model_name, image_id)
        mi_filepath = os.path.join(review_dir, model_image_name)

        plot_images(mi_filepath, X_test, Y_pred, Y_test)


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

#mod = functional_sony()
#model = restore_model(mod, 'sony_bl_cd_pn_ag')
#_ = review_model(model, './Sony_RGB/Sony/short/00015_07_0.1s.png')
    
    



