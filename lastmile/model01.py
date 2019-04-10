""" Trying a simpler approach using Keras. 


"""
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.layers import LeakyReLU, Lambda

from model_utils import create_patch

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)


def simple_sony():
    """ Simpified version of the sony LSD model. """

    lrelu = LeakyReLU(alpha=0.2)
    inputs = Input(shape=(32,32,3))

    # Convert X to patches

    x0 = Lambda(lambda x: create_patch(x))(inputs)

    # Block 1
    x1 = Conv2D(32, (3,3), padding='same', activation=lrelu)(x0)
    x1 = Conv2D(32, (3,3), padding='same', activation=lrelu)(x1)
    x1 = MaxPooling2D(pool_size=2)(x1)

    # Block 2, transition
    x2 = Conv2D(64, (3,3), padding='same', activation=lrelu)(x1)
    x2 = Conv2D(64, (3,3), padding='same', activation=lrelu)(x2)

    # Block 3, upsampling
    x3 = UpSampling2D(size=(2,2))(x2)
    x3 = Conv2D(32, (3,3), padding='same', activation=lrelu)(x3)
    x3 = Conv2D(32, (3,3), padding='same', activation=lrelu)(x3)

    # Block 4, output
    x4 = Conv2D(12, (3,3), padding='same', activation=lrelu)(x3)
    outputs = Lambda(lambda x : tf.depth_to_space(x, 2))(x4)

    # Define model
    model = Model(inputs=inputs, outputs=outputs)

    return model

def full_sony():
    """ Full version of the Sony LSD model. """

    model_id = 'sony'
    lrelu = LeakyReLU(alpha=0.2)
    model = Sequential()

    # Convert X to patches

    model.add(Lambda(lambda x: create_patch(x)))
    
    # Block 1
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=2))

    # Block 2
    model.add(Conv2D(64, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(64, (3,3), padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=2))

    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(128, (3,3), padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=2))

    # Block 4
    model.add(Conv2D(256, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(256, (3,3), padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=2))

    # Block 5
    model.add(Conv2D(512, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(512, (3,3), padding='same', activation=lrelu))

    # Block 6
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(256, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(256, (3,3), padding='same', activation=lrelu))

    # Block 7
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(128, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(128, (3,3), padding='same', activation=lrelu))

    # Block 8
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(64, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(64, (3,3), padding='same', activation=lrelu))

    # Block 9
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))

    # Block 10
    model.add(Conv2D(12, (3,3), padding='same', activation=lrelu))
    model.add(Lambda(lambda x : tf.depth_to_space(x, 2)))

    mod = {"model": model, "model_id": model_id}
    return mod






    

    
    

    

