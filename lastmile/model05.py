
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Lambda


from model_utils import create_patch

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)



def functional_sony():

    model_id = 'sony'

    lrelu = LeakyReLU(alpha=0.2)
    inputs = Input(shape=(256,256,3))

    x0 = Lambda(lambda x: create_patch(x))(inputs)

    # Block 1
    x1 = Conv2D(512, (3,3), padding='same', activation=lrelu)(x0)
    x1 = Conv2D(512, (3,3), padding='same', activation=lrelu)(x1)
    x1 = MaxPooling2D(pool_size=2)(x1)

    # Block 2
    x2 = Conv2D(64, (3,3), padding='same', activation=lrelu)(x1)
    x2 = Conv2D(64, (3,3), padding='same', activation=lrelu)(x2)
    x2 = MaxPooling2D(pool_size=2)(x2)

    # Block 3
    x3 = Conv2D(128, (3,3), padding='same', activation=lrelu)(x2)
    x3 = Conv2D(128, (3,3), padding='same', activation=lrelu)(x3)
    x3 = MaxPooling2D(pool_size=2)(x3)

    # Block 4
    x4 = Conv2D(256, (3,3), padding='same', activation=lrelu)(x3)
    x4 = Conv2D(256, (3,3), padding='same', activation=lrelu)(x4)
    x4 = MaxPooling2D(pool_size=2)(x4)

    # Block 5
    x5 = Conv2D(512, (3,3), padding='same', activation=lrelu)(x4)
    x5 = Conv2D(512, (3,3), padding='same', activation=lrelu)(x5)

    # Block 6
    x6 = UpSampling2D(size=(2,2))(x5)
    x6 = Conv2D(256, (3,3), padding='same', activation=lrelu)(x6)
    x6 = Conv2D(256, (3,3), padding='same', activation=lrelu)(x6)

    # Block 7
    x7 = UpSampling2D(size=(2,2))(x6)
    x7 = Conv2D(128, (3,3), padding='same', activation=lrelu)(x7)
    x7 = Conv2D(128, (3,3), padding='same', activation=lrelu)(x7)

    # Block 8
    x8 = UpSampling2D(size=(2,2))(x7)
    x8 = Conv2D(64, (3,3), padding='same', activation=lrelu)(x8)
    x8 = Conv2D(64, (3,3), padding='same', activation=lrelu)(x8)

    # Block 9
    x9 = UpSampling2D(size=(2,2))(x8)
    x9 = Conv2D(32, (3,3), padding='same', activation=lrelu)(x9)
    x9 = Conv2D(32, (3,3), padding='same', activation=lrelu)(x9)

    # Block 10
    x10 = Conv2D(12, (3,3), padding='same', activation=lrelu)(x9)
    outputs = Lambda(lambda x : tf.depth_to_space(x, 2))(x10)

    model = Model(inputs=inputs, outputs=outputs)

    mod = {"model": model, "model_id": model_id}

    return mod
