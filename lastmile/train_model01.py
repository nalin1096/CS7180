""" Training and test for model01

This is a version of train_Sony from LSD that keeps 
their intuition but uses less layers and uses an RGB
instead of RAW input image.

It's helpful to keep in mind that Keras provides us with
a lot of debugging functionality. We can extract an image
at each convolutional layer, for example.
"""
import logging
from urllib.parse import urljoin

from tensorflow.train import AdamOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam

from model01 import simple_sony, full_sony
from model_utils import enable_cloud_log, plot_imgpair, plot_loss
from custom_loss import mean_absolute_error


logger = logging.getLogger(__name__)
enable_cloud_log(level='DEBUG')

# Create checkpoint callback
checkpoint_path = 'checkpoints/cp.ckpt'
cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True,
                              verbose=1)


# Dataset of 50,000 32x32 color training images, 
# labeled over 10 categories, and 10,000 test images.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

m = 64

#Y_train = X_train
#Y_test = X_test
X_train = X_train[0:m,...]
Y_train = X_train[0:m,...]
X_test = X_test[0:m,...]
Y_test = X_test[0:m,...]


logger.debug("X_train default shape: {}".format(X_train.shape))
logger.debug("Y_train default shape: {}".format(Y_train.shape))


# Compiling model using Keras

learning_rate = 1e-3
model = simple_sony()
#model = full_sony()
#opt = Adam(lr=1e-4)
opt = AdamOptimizer(learning_rate=learning_rate)

model.compile(optimizer=opt,
              loss=mean_absolute_error,
              metrics=['accuracy'])

# Fitting the model

history = model.fit(X_train, Y_train, validation_split=0.25,
                    epochs=100, batch_size=32,
                    callbacks=[cp_callback])
plot_loss('review/train_val_loss.png', history)

# Predicting with the model

model.summary()

output = model.predict(X_test)
logger.debug("prediction output shape: {}".format(output.shape))


# Review image output

every = 10000
for i in range(output.shape[0]):

    base = "review/"
    if i % every == 0:

        name = urljoin(base, 'model_pred_{}.png'.format(i))
        plot_imgpair(output[i, ...], Y_test[i,...], name)
        

