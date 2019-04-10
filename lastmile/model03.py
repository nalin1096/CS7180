from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential


def my_relu(x):
    return K.relu(x, alpha=0.2)

def model03():
    """ Removed upsampling from model01. """
    model_id = 'model03'

    model = Sequential()

    # Block 1
    model.add(Conv2D(8, (3,3), activation=my_relu,
                        padding='same', input_shape=(256,256,3)))
    model.add(Conv2D(8, (3,3), padding='same', activation=my_relu))

    # Block 2
    model.add(Conv2D(16, (3,3), padding='same', activation=my_relu))
    model.add(Conv2D(16, (3,3), padding='same', activation=my_relu))

    # Block 3
    model.add(Conv2D(32, (3,3), padding='same', activation=my_relu))
    model.add(Conv2D(32, (3,3), padding='same', activation=my_relu))

    # Block 4
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(3))

    mod = {"model": model, "model_id": model_id}
    return mod
