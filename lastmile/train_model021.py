""" Training and test for model v0.2.1

This is a version which includes the augmented data. We're
using a different approach than the Sony model so we don't
have to use patching. RGB input image.
"""
import os
import logging
from urllib.parse import urljoin

import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model02 import model02
from model_utils import enable_cloud_log, plot_images, plot_loss
from custom_loss import mean_absolute_error
from augment_utils import adjust_gamma

logger = logging.getLogger(__name__)
enable_cloud_log('DEBUG')

# Create checkpoint callback

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True,
                              period=5, verbose=1)


# Dataset of 50,000 32x32 color training images, 
# labeled over 10 categories, and 10,000 test images.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

m = 64
X_train = X_train[0:m,...]
X_test = X_test[0:m,...]
Y_train = X_train
Y_test = X_test

# Define model

model = model02()

# Retrieve latest checkpoint if it exists

chk = os.listdir(checkpoint_dir)
if len(chk) > 0:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

else:

    datagen = ImageDataGenerator(
        preprocessing_function=adjust_gamma)

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

    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=32),
                                  steps_per_epoch=X_train.shape[0] / 32, epochs=100)
    plot_loss('review/train_val_loss_021.png', history)


# Predict with model

X_gamma_test = adjust_gamma(X_test)

output = model.predict(X_gamma_test)
logger.debug("prediction output shape: {}".format(output.shape))

# Review image output

every = 10
for i in range(output.shape[0]):

    base = "review/"
    if i % every == 0:

        name = urljoin(base, 'model_pred_{}.png'.format(i))
        plot_imgpair(name, X_gamma_test[i,...], output[i, ...], Y_test[i,...])






