""" Train Imagenet

We want to see if we can generate useful parameters using data augmented
from ImageNet (http://www.image-net.org/) instead of from a professional
photographer with a fancy camera.  

Note that I've updated this model to use Keras instead of 
tensorflow.contrib.slim
"""
import glob
import logging
import os
import time

import numpy as np
import rawpy
import tensorflow as tf
from keras.layers import Conv2D

def lrelu(x):
    return tf.maximum(x * 0.2, x)

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

def model(input_shape):
    pass


