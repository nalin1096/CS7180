""" Trying a simpler approach using Keras. 


"""
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.kears.layers import LeakyReLU


def simple_sony():
    """ Simpified version of the sony LSD model. """

    lrelu = LeakyReLU(alpha=0.2)
    inputs = Input(shape(32,))

    # Block 1
    x1 = Conv2D(8, (3,3), padding='same', activation=lrelu)(inputs)
    x1 = Conv2D(8, (3,3), padding='same', activation=lrelu)(x1)
    x1 = MaxPooling2D(pool_size=2)(x1)

    # Block 2, transition
    x2 = Conv2D(16, (3,3), padding='same', activation=lrelu)(x1)
    x2 = Conv2D(16, (3,3), padding='same', activation=lrelu)(x2)

    # Block 3, upsampling
    x3 = UpSampling2D(size=(2,2))(x2)
    x3 = Conv2D(8, (3,3), padding='same', activation=lrelu)(x3)
    x3 = Conv2D(8, (3,3), padding='same', activation=lrelu)(x3)

    # Block 4, output
    x4 = Conv2D(8, (3,3), padding='same', activation=lrelu)(x3)
    outputs = K.depth_to_space(x4, 2)

    # Define model
    model = Model(inputs=inputs, outputs=outputs)

    return model
    

    
    

    

