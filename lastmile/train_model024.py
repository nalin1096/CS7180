""" Training and test for model v0.2.3

This model runs on multiple gpus. Checkpointing. RGB input images.

TODO:

* Set up sony with dataflow
* Include patches
* Update image augmentation functions from last night
* Run on GPUs
* Possibly include a GAN

"""
from datetime import datetime
import os
import logging
import pickle
from urllib.parse import urljoin

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from model03 import model03
from model_utils import (enable_cloud_log, plot_imgpair,
                         plot_loss, create_patch)
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

    model_name = '%s_model.{epoch:03d}-{val_loss:.2f}.hdf5' % model_type
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving (option to adjust lr)

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                 verbose=1)
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
    model.summary()

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

def fit_model_ngpus(train_dataflow, val_dataflow, model, model_id,
                    imgproc, lr, epochs):
    """ Fits model with N GPUs, Returns model and history keras objects. """

    # Replicate the model on 4 GPUs

    parallel_model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
    
    opt = Adam(lr=lr)
    parallel_model.compile(optimizer=opt,
                           loss=mean_absolute_error,
                           metrics=['accuracy'])
    parallel_model.summary()

    # Fit model

    model_type = '{}_{}'.format(model_id, imgproc)
    calls = callbacks(model_type=model_type)
    history = parallel_model.fit_generator(
        generator=train_dataflow,
        epochs=epochs,
        callbacks=calls,
        validation_data=val_dataflow
    )

    return parallel_model, history

def model_predict(test_imgreview_dataflow, model, idp, Y_true_fpath):
    """ Predict patches from a single image then reconstruct that image. """

    patches = np.array([pred for pred in
                        model.predict_generator(test_imgreview_dataflow)])

    Y_true = cv2.imread(Y_true_fpath)
    Y_pred = idp.reconstruct_patches(
        patches, image_size=Y_true.shape
    )

    return Y_pred, Y_true
 
def review_model(test_dataflow, test_imgreview, model, history,
                 model_id, imgproc, image_data_pipeline):
    """ Model diagnostics written to disk; performs prediction """

    model_name = '{}_{}'.format(model_id, imgproc)
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("STARTED model diagnostics for '{}'".format(model_name))

    # Managing review directory
    
    review_dir = os.path.join(os.getcwd(), 'review')
    if not os.path.isdir(review_dir):
        os.makedirs(review_dir)

    # Report train/val loss vs. epochs

    if history is not None:
        logger.info("Reporting train/val loss vs. epochs plot")
        model_history_name = 'history_{}_{}.png'.format(model_name,
                                                        datetime_now)
        mh_filepath = os.path.join(review_dir, model_history_name)
        plot_loss(mh_filepath, history)

    # Evaluate using test set

    logger.info("Evaluating test set and generating report")
    evaluate = model.evaluate_generator(test_dataflow)
    test_eval = str(model.metrics_names) + str(evaluate)
    test_eval_name = 'test_eval_{}_{}.txt'.format(model_name, datetime_now)
    te_filepath = os.path.join(review_dir, test_eval_name)
    with open(te_filepath, "w") as outfile:
        outfile.write(test_eval)

    # Prediction and reconstrution of N images

    logger.info("Predicting test image and generating review images")

    for file_path in test_imgreview:

        y_test_set = [file_path]
        test_imgreview_dataflow = RaiseDataGenerator(y_test_set,
                                                     image_data_pipeline)

        Y_pred, Y_true = model_predict(test_imgreview_dataflow, model,
                                       image_data_pipeline,
                                       Y_true_fpath=file_path)
                                       
        img_pred_name = 'img_pred_{}_{}.png'.format(model_name, datetime_now)
        img_filepath = os.path.join(review_dir, img_pred_name)
        
        plot_imgpair(Y_pred, Y_true, img_filepath)
        logger.info("Wrote out review image: {}".format(img_pred_name))

    logger.info("FINISHED model diagnostics for '{}'".format(model_name))

def run_simulation_ngpus(mod: dict):
    """ Run simulation using N GPUs """
    logger.info("STARTED running simulations")

    imgnames = ['bl', 'bl_cd', 'bl_cd_pn', 'bl_cd_pn_ag']
    
    # Run model on each data augmentation scenario

    for imgproc in imgnames:
    
        # Define model
        
        model = mod.get('model', None)
        model_id = mod.get('model_id', None)

        if model is None:
            raise TypeError("model must be defined: {}".format(model))

        logger.info("Processing model: {}".format(imgproc))

        # Specify Image Data Pipeline

        idp = ImageDataPipeline(preprocessing_function=imgproc,
                                stride=128,
                                batch_size=64,
                                patch_size=(256,256),
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

        # Fit Model

        model, history = fit_model_ngpus(train_dataflow, val_dataflow,
                                         model, model_id,
                                         imgproc, lr=1e-3, epochs=10)

        # Review model

        test_dir = 'raise/rgb/test/'
        y_test_set = [urljoin(test_dir, f) for f in os.listdir(test_dir)]
        test_dataflow = RaiseDataGenerator(y_test_set, idp)

        test_imgreview_dir = 'raise/rgb/test/'
        test_imgreview = [urljoin(test_imgreview_dir, f)
                          for f in os.listdir(test_imgreview_dir)]

        review_model(test_dataflow, test_imgreview, model, history,
                     model_id, imgproc, idp)

        # Reset model and history

        model = None
        history = None

    logger.info("FINISHED running simulations")


def run_simulation(mod: dict):
    """ Run models using data augmented with simulated images. 
    
    Arguments:
        mod: output from a model function

    Returns:
        Runs through training, validation, and testing for a given
        model using all data augmentation functions. Reports model 
        diagnostics for each data augmentation function. Checkpoints
        the given model using <model_id> and data augmentation function
        name <imgproc> using subfolders within a 'saved_models' directory.
        Output comparisons use the same subfolders and are located in 
        the 'reviews' directory.
    """
    logger.info("STARTED running simulations")

    imgnames = ['bl', 'bl_cd', 'bl_cd_pn', 'bl_cd_pn_ag']
    
    # Run model on each data augmentation scenario

    for imgproc in imgnames:

        # Define the data flow for training, validation, and test sets

        idp = ImageDataPipeline(preprocessing_function=imgproc,
                                    stride=128,
                                    batch_size=1,
                                    patch_size=(256,256),
                                    random_seed=42,
                                    meanm_fpath='simulation_mean.pkl',
                                    covm_fpath='simulation_cov.pkl',
                                    num_images=10
        )

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
                                   imgproc, lr=1e-3, epochs=3)

        # Review model

        test_dir = 'raise/rgb/test/'
        y_test_set = [urljoin(test_dir, f) for f in os.listdir(test_dir)]
        test_dataflow = RaiseDataGenerator(y_test_set, idp)

        test_imgreview_dir = 'raise/rgb/test/'
        test_imgreview = [urljoin(test_imgreview_dir, f)
                          for f in os.listdir(test_imgreview_dir)]

        review_model(test_dataflow, test_imgreview, model, history,
                     model_id, imgproc, idp)

        # Reset model and history

        model = None
        history = None

    logger.info("FINISHED running simulations")

def run_sony_images(mod, model_name):

    logger.info("STARTED running sony images")

    # Loading saved weights to model

    model = mod.get('model', None)

    if model is None:
        raise TypeError("model must be defined: {}".format(model))
    
    save_dir = os.path.join(os.getcwd(), 'saved_models', model_name)
    model.load_weights(save_dir)
    model.summary()

    # Training model
    # TODO: freeze layers and train model using sony images

    idp = ImageDataPipeline(preprocessing_function='sony',
                            stride=128,
                            batch_size=1,
                            patch_size=(256,256),
                            random_seed=42,
                            meanm_fpath='simulation_mean.pkl',
                            covm_fpath='simulation_cov.pkl',
                            num_images=10
    )

    # Review model

    
    #test_dataflow = datagen.dirflow_val_sony(
    #    sony_val_list='dataset/Sony_test_list.txt'
    #)

    #test_imgreview_dataflow = datagen.dirflow_test_sony(
    #    sony_test_list='dataset/Sony_test_list.txt'
    #)

    #review_model(test_dataflow, test_imgreview_dataflow, model,
    #             history=None, model_id=model_name, imgproc='bl_cd_pn_ag',
    #             image_data_pipeline=idp)

    logger.info("FINISHED running sony images")

#################################################
# Main functions for each hardware configuration
#################################################

def main():
    """ Main function to run training and prediction. """

    mod = model03()
    run_simulation(mod)

    mod = model03()
    model_id = mod.get('model_id', None)
    imgproc = 'bl_cd_pn_ag'
    model_name = '{}_{}'.format(model_id, imgproc)
    #run_sony_images(mod, model_name)

def main_ngpus():
    """ Main function to run training and predictions on N GPUs. """

    mod = model03()
    run_simulation_ngpus(mod)


#######################################################
# Running train_model script, Jupyter Notebook config
#######################################################

enable_cloud_log('INFO')
#main()
main_ngpus()

