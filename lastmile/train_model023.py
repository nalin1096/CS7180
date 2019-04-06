""" Training and test for model v0.2.3

This model runs on multiple gpus. Checkpointing. RGB input images.

TODO:

* Set up dataflow for each model run
* Set up sony with dataflow
* Verify checkpoint restoration
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

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model02 import model02
from model_utils import enable_cloud_log, plot_images, plot_loss
from custom_loss import mean_absolute_error

logger = logging.getLogger(__name__)

#####################
# Helper functions
#####################

def read_pickle(fpath):
    with open(fpath, "rb") as infile:
        m = pickle.load(infile)
    return m

###############################
# Image augmentation functions
###############################

def bl(image, sample=False):
    """ Apply black level """
    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    BL = int(sample[0])
    image[image < BL] = BL
    image = image - BL
    return image

def bl_cd(image, sample=False):
    """ Apply black level with color distortion """

    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    image = bl(image, sample)

    WB = [ sample[1], sample[2], sample[3] ]
    
    image[... ,0] = WB[0] * image[... ,0]
    image[... ,1] = WB[1] * image[... ,1]
    image[... ,2] = WB[2] * image[... ,2]

    return image

def bl_cd_pn(image, sample=False):
    """ Apply black level with color distortion and poisson noise. """
    
    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    noise_param = 10

    image = bl_cd(image, sample)

    noise = lambda x : np.random.poisson(x / 255.0 * noise_param) / \
        noise_param * 255

    func = np.vectorize(noise)
    image = func(image)
    return image

def bl_cd_pn_ag(image, sample=False):
    """ 
    Apply black level, color distortion, poisson noise, adjust gamma. 
    """

    if not np.all(sample):
        sample = np.random.multivariate_normal(MEANM, COVM)

    image = bl_cd_pn(image, sample)
    image = image**sample[4]
    return image

##################################################
# Model fitting, prediction, and review functions
##################################################

def callbacks(model_type):
    """
    Callbacks handle model checkpointing. We can also use them
    to adjust learning rate.
    """
    # Prepare model, model saving directory
    
    save_dir = os.path.join(os.getwcd(), 'saved_models')
    model_name = '%s_model.{epoch:03d}-{val_loss:.2f}.hdf5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving (option to adjust lr)

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                 verbose=1)
    callbacks = [checkpoint]

    return callbacks

def fit_model(dataflow, model, imgtup, lr=1e-3, epochs=100):
    """ Fits model, Returns model and history keras objects. """

    imgname, imgfun = imgtup

    # Compile model
    
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss=mean_absolute_error,
                  metrics=['accuracy'])
    model.summary()

    # Fit model

    calls = callbacks(model_type=imgname)
    history = model.fit_generator(dataflow, epochs=epochs, callbacks=calls)

    return model, history

def fit_model_ngpus(X_train, Y_train, model, imgtup, lr=1e-3, epochs=100):
    """ Fits model with N GPUs, Returns model and history keras objects. """
    pass

def model_predict(model, X_test, imgtup):
    """ Returns y_pred """

    imgname, imgfunc = imgtup

    X_noise_test = imgfunc(X_test)
    output = model.predict(X_noise_test)
    
    logger.debug("prediction output shape: {}".format(output.shape))

    return output

def review_model(X_test, Y_true, model, history, imgtup, num_images=10):
    """ Model diagnostics written to disk; performs prediction """

    logger.info("STARTED model diagnostics")

    imgname, imgfunc = imgtup

    # Managing review directory
    
    review_dir = os.path.join(os.getwcd(), 'review')
    if not os.path.isdir(review_dir):
        os.makedirs(review_dir)

    # Prediction

    Y_pred = model_predict(model, X_test, imgtup)

    # Output loss plot for model name

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    model_history_name = 'history_{}_{}.png'.format(imgname, datetime_now)
    mh_filepath = os.path.join(review_dir, model_history_name)
    plot_loss(mh_filepath, history)

    logger.info("Wrote out model history loss plt: {}".\
                format(model_history_name))

    # Output 'num_images' fmt: imgfunc(X_test), Y_pred, Y_true

    np.random.seed(0)

    y_pred_idx = np.array([i for i in range(Y_pred.shape[0])])
    np.random.shuffle(y_pred_idx)

    for i in range(num_images):

        img_idx = y_pred_idx[i]

        img_pred_name = 'img_pred_{}_idx_{}_{}.png'.\
            format(imgname, img_idx, datetime_now)
        img_filepath = os.path.join(review_dir, img_pred_name)
        
        plot_images(img_filepath, imgfunc(X_test[img_idx, ...]),
                    Y_pred[img_idx, ...], Y_true[img_idx, ...])

        logger.info("Wrote out review image: {}".format(img_pred_name))

    logger.info("FINISHED model diagnostics")

def run_simulation(fcov, fmean, dataflow):
    """ Run models using data augmented with simulated images. """

    logger.info("STARTED running simulations")

    imgman = [
        ('bl', bl),
        ('bl_cd', bl_cd),
        ('bl_cd_pn', bl_cd_pn),
        ('bl_cd_pn_ag', bl_cd_pn_ag),
    ]

    for imgtup in imgman:

        imgname, imgfunc = imgtup

        # Define the data flow

        dataflow = None # TODO

        # Define model

        model = model02()
        logger.info("Processing model: {}".format(imgname))

        # Fit model

        model, history = fit_model(dataflow, model, imgtup,
                                   lr=1e-3, epochs=100)

        # Review model

        review_model(X_test, Y_true, model, history, imgtup, num_images=10)

    logger.info("FINISHED running simulations")

def run_sony_images():
    pass

def main():
    """ Main function to run training and prediction. """

    datagen = ImageDataGenerator(preprocessing_function=imgfunc)

    # Transform all training images
    datagen.fit(X_train)

    dataflow = datagen.flow(X_train, Y_train, batch_size=32)


    
    pass

def main_ngpus():
    """ Main function to run training and predictions on N GPUs. """
    pass
        
    

                                
                                               


