""" Removing upsampling, just using convolutional layers.

"""
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.models import Sequential


class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)

def model02():
    """ Removed upsampling from model01. """

    lrelu = LeakyReLU(alpha=0.2)
    model = Sequential()

    # Block 1
    model.append(Conv2D(8, (3,3), activation=lrelu,
                        padding='same', input_shape=(32,32,3)))
    model.append(Conv2D(8, (3,3), padding='same', activation=lrelu))
    model.append(MaxPooling2D(pool_size=2))

    # Block 2
    model.append(Conv2D(16, (3,3), padding='same', activation=lrelu))
    model.append(Conv2D(16, (3,3), padding='same', activation=lrelu))
    model.append(MaxPooling2D(pool_size=2))

    # Block 3
    model.append(Conv2D(32, (3,3), padding='same', activation=lrelu))
    model.append(Conv2D(32, (3,3), padding='same', activation=lrelu))
    model.append(MaxPooling2D(pool_size=2))

    

    


