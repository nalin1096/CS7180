""" Model 04 is our model for submission

We had to modify Chen et. al. (2018) Sony_train to
work with RGB instead of RAW images.
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential

def model04():
    """ Removed upsampling from model01. """
    model_id = 'model04'

    model = Sequential()

    # Block 1
    model.add(Conv2D(8, (3,3), activation=K.relu,
                        padding='same', input_shape=(256,256,3)))
    model.add(Conv2D(8, (3,3), padding='same', activation=K.relu))

    # Block 2
    model.add(Conv2D(16, (3,3), padding='same', activation=K.relu))
    model.add(Conv2D(16, (3,3), padding='same', activation=K.relu))

    # Block 3
    model.add(Conv2D(32, (3,3), padding='same', activation=K.relu))
    model.add(Conv2D(32, (3,3), padding='same', activation=K.relu))

    # Block 4
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(3))

    mod = {"model": model, "model_id": model_id}
    return mod
