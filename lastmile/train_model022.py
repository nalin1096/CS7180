""" Training and test for model v0.2.1

This is a version which includes the augmented data. We're
using a different approach than the Sony model so we don't
have to use patching. RGB input image.
"""
import os
import logging
import pickle
from urllib.parse import urljoin

import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model02 import model02
from model_utils import enable_cloud_log, plot_images, plot_loss
from custom_loss import mean_absolute_error

logger = logging.getLogger(__name__)

# Helper functions

def read_pickle(fpath):
    with open(fpath, "rb") as infile:
        m = pickle.load(infile)
    return m

def bl(image, sample=False):
    """ Apply black level """
    if not sample:
        sample = np.random.multivariate_normal(MEANM, COVM)

    BL = int(sample[0])
    image[image < BL] = BL
    image = image - BL
    return image

def bl_cd(image, sample=False):
    """ Apply black level with color distortion """

    if not sample:
        sample = np.random.multivariate_normal(MEANM, COVM)

    image = bl(image, sample)

    WB = [ sample[1], sample[2], sample[3] ]
    
    image[... ,0] = WB[0] * image[... ,0]
    image[... ,1] = WB[1] * image[... ,1]
    image[... ,2] = WB[2] * image[... ,2]

    return image

def bl_cd_pn(image, sample=False):
    """ Apply black level with color distortion and poisson noise. """
    
    if not sample:
        sample = np.random.multivariate_normal(MEANM, COVM)

    noise_param = 10

    image = bl_cd(image, sample)
    image = np.random.poisson(image / 255.0 * noise_param) / \
        noise_param * 255

    return image

def bl_cd_pn_ag(image, sample=False):
    """ 
    Apply black level, color distortion, poisson noise, adjust gamma. 
    """

    if not sample:
        sample = np.random.multivariate_normal(MEANM, COVM)

    image = bl_cd_pn(image, sample)
    image = image**sample[4]
    return image


# Create checkpoint callback

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True,
                              period=5, verbose=1)


# Retrieve latest checkpoint if it exists

def fit_model(X_train, Y_test, model, checkpoint_dir, imgtup):

    imgname, imgfunc = imgtup
    
    chk = os.listdir(checkpoint_dir)
    if len(chk) > 1:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)

    else:
        datagen = ImageDataGenerator(
            preprocessing_function=imgfunc)

        # Transform all training images
        datagen.fit(X_train)

        # Compile model

        learning_rate = 1e-3
        opt = AdamOptimizer(learning_rate=learning_rate)

        model.compile(optimizer=opt,
                      loss=mean_absolute_error,
                      metrics=['accuracy'])

        model.summary()

        # Fit model

        history = model.fit_generator(datagen.flow(X_train,Y_train,
                                                   batch_size=32),
                                      steps_per_epoch=X_train.shape[0] / 32,
                                      epochs=100)
        plot_loss('review/train_val_loss_021_{}.png'.format(imgname), history)

    return model

def model_predict(model, X_test, imgtup):

    imgname, imgfunc = imgtup

    X_noise_test = imgfunc(X_test)
    output = model.predict(X_noise_test)
    
    logger.debug("prediction output shape: {}".format(output.shape))

    return output

def review_image_output(X_test, Y_pred, Y_true, imgtup, every=10):
    # Review image output

    imgname, imgfunc = imgtup
    base = "review/"

    for i in range(Y_pred.shape[0]):

        if i % every == 0:

            name = urljoin(base, 'model_pred_{}_{}.png'.format(i, imgname))
            plot_images(name, imgfunc(X_test[i,...]), Y_pred[i, ...],
                        Y_test[i,...])


def run_simulation(fcov, fmean):

    logger.info("STARTED running simulations")

    # Dataset of 50,000 32x32 color training images, 
    # labeled over 10 categories, and 10,000 test images.

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    m = 64
    X_train = X_train[0:m,...]
    X_test = X_test[0:m,...]
    Y_train = X_train
    Y_test = X_test

    imgman = [
        ('bl', bl),
        ('bl_cd', bl_cd),
        ('bl_cd_pn', bl_cd_pn),
        ('bl_cd_pn_ag', bl_cd_pn_ag),
    ]

    for imgtup in imgman:

        # Define model    
        model = model02()

        imgname, imgfunc = imgtup
        logger.info("Processing: {}".format(imgtup[0]))
        model = fit_model(X_train, Y_test, model, checkpoint_dir, imgtup)
        Y_pred = model_predict(model, X_test, imgtup)
        review_image_output(X_test, Y_pred, Y_test, imgtup, every=10)

        model = None

    logger.info("FINISHED running simulations")

if __name__ == "__main__":

    COVM = read_pickle(fcov)
    MEANM = read_pickle(fmean)

    enable_cloud_log('DEBUG')
    
    fcov = "simulation_cov.pkl"
    fmean = "simulation_mean.pkl"

    run_simulation(fcov, fmean)

