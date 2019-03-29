""" Trying a simpler approach using Keras. 


"""
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input


def simple_sony():
    """ Simpified version of the sony LSD model. """

    inputs = Input(shape(None,))

    # Block 1
    x1 = Conv2D(32)(inputs)
    x1 = Conv2D(32)(x1)
    x1 = MaxPooling2D(x1)

    # Block 2, transition
    x2 = Conv2D(32)(x1)
    x2 = Conv2D(32)(x2)

    # Block 3, upsampling
    x3 = UpSampling2D(x2,x1)
    x3 = Conv2D(32)(x3)
    x3 = Conv2D(32)(x3)

    # Block 4, output
    x4 = Conv2D(32)(x3)
    outputs = K.depth_to_space(x4, 2)

    # Define model
    model = Model(inputs=inputs, outputs=outputs)

    return model
    

    
    

    

