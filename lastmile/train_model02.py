""" Training and test for model02

This is a version which tries to remove the need for upsampling
that we saw in the Sony model. RGB input image.
"""
import logging
from urllib.parse import urljoin

from tensorflow.train import AdamOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10

from model02 import model02
from model_utils import enable_cloud_log, plot_imgpair, plot_loss
from custom_loss import mean_absolute_error


logger = logging.getLogger(__name__)
enable_cloud_log('DEBUG')

# Create checkpoint callback
checkpoint_path = 'checkpoints/cp.ckpt'
cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True,
                              verbose=1)


# Dataset of 50,000 32x32 color training images, 
# labeled over 10 categories, and 10,000 test images.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

m = 64
X_train = X_train[0:m,...]
X_test = X_test[0:m,...]
Y_train = X_train
Y_test = X_test

# Simulate noisy image

#X_train = 0.7 * X_train
#X_test = 0.7 * X_test

# Compiling model using Keras

learning_rate = 1e-3
model = model02()
opt = AdamOptimizer(learning_rate=learning_rate)


model.compile(optimizer=opt,
              loss=mean_absolute_error,
              metrics=['accuracy'])

model.summary()

# Fitting the model

history = model.fit(X_train, Y_train, validation_split=0.25,
                    epochs=100, batch_size=32,
                    callbacks=[cp_callback])
plot_loss('review/train_val_loss.png', history)

# Predicting with the model

output = model.predict(X_test)
logger.debug("prediction output shape: {}".format(output.shape))


# Review image output

every = 10
for i in range(output.shape[0]):

    base = "review/"
    if i % every == 0:

        name = urljoin(base, 'model_pred_{}.png'.format(i))
        plot_imgpair(output[i, ...], Y_test[i,...], name)



