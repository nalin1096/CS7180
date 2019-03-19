""" Specify imagenet model.

"""
import numpy as np
import rawpy
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, LeakyReLU, Dense,
                                     Reshape)
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal(
        [pool_size, pool_size, output_channels, in_channels], stddev=0.02
    ))
    
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2),
                                    strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def pool_block(model, output, activ):

    model.add(Conv2D(output, [3,3], dilation_rate=1, activation=activ))
    model.add(Conv2D(output, [3,3], dilation_rate=1, activation=activ))
    model.add(MaxPooling2D([2,2], padding='same'))

    return model

def upsample_block(model, innodes, outnodes, activ):

    model.add(UpsampleAndConcat())
    model.add(Conv2D(output, [3,3], dilation_rate=1, activation=lrelu))
    model.add(Conv2D(output, [3,3], dilation_rate=1, activation=lrelu))
    
    return model

def specify_model(input_shape):
    model = Sequential()
    lrelu = LeakyReLU(alpha=0.2)

    model = pool_block(model, 32, lrelu)         # Block 1
    model = pool_block(model, 64, lrelu)         # Block 2
    model = pool_block(model, 128, lrelu)        # Block 3
    model = pool_block(model, 256, lrelu)        # Block 4

    # Block 5
    
    model.add(Conv2D(512, [3,3], dilation_rate=1, activation=lrelu))
    model.add(Conv2D(512, [3,3], dilation_rate=1, activation=lrelu))

    # Upsample blocks

    model = upsample_block(model, 256, 512, lrelu) # Block 6
    model = upsample_block(model, 128, 256, lrelu) # Block 7
    model = upsample_block(model, 64, 128, lrelu)  # Block 8
    model = upsample_block(model, 32, 64, lrelu)   # Block 9

    return model
    
def debug_model(input_shape):
    model = Sequential()
    lrelu = LeakyReLU(alpha=0.2)

    # Input block

    model.add(Conv2D(32, (3,3), padding='same',
                     input_shape=input_shape[1:],
                     dilation_rate=1, activation=lrelu))
    model.add(Conv2D(32, (3,3), padding='same',
                     dilation_rate=1, activation=lrelu))
    model.add(MaxPooling2D((2,2), padding='same'))


    # Additional blocks

    

    # Output block

    # out = tf.depth_to_space(conv10, 2)

    #model = pool_block(model, 32, lrelu)
    #model.add(Dense(units=3, activation='softmax'))

    #model.add(Conv2D(input_shape, [3,3], dilation_rate=1, activation=lrelu))
    
    return model



    
