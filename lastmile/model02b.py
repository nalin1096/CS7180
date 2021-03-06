""" Removing upsampling, just using convolutional layers.

"""
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential


class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)

def model02():
    """ Removed upsampling from model01. """
    model_id = 'model02'

    lrelu = LeakyReLU(alpha=0.2)
    model = Sequential()

    # Block 1
    model.add(Conv2D(8, (3,3), activation=lrelu,
                        padding='same', input_shape=(256,256,3)))
    model.add(Conv2D(8, (3,3), padding='same', activation=lrelu))

    # Block 2
    model.add(Conv2D(16, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(16, (3,3), padding='same', activation=lrelu))

    # Block 3
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))

    # Block 4
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(3))

    mod = {"model": model, "model_id": model_id}
    return mod

    

    


