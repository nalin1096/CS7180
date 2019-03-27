""" Specify imagenet model.

"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import rawpy
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim

#from train_Sony import network


logger = logging.getLogger(__name__)

def create_placeholders(n_H, n_W, n_C):
    """ 
    Creates the placeholders for the tensorflow session.

    Note that the placeholders will most likely vary by input dataset.

    Arguments:
    n_H -- scalar, height of an input image
    n_W -- scalar, width of an input image
    n_C -- scalar, number of channels of the input

    Returns:
    X -- placeholder for the data input, of shape [None, n_H, n_W, n_C] and dtype "int"
    Y -- placeholder for the input labels, of shape and dtype equal to X
    """

    X = tf.placeholder(tf.float32, shape=(None, n_H, n_W, n_C))
    Y = tf.placeholder(tf.float32, shape=(None, n_H, n_W, n_C))

    return X,Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow.
    The shapes are:
        []

    Returns:
    parameters -- a dictionary of tensors containing W1,...,WN
    """
    tf.set_random_seed(1)

    # Block 1

    W1 = tf.get_variable("W1", [3,3,3,32],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [3,3,32,32],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 2

    W3 = tf.get_variable("W3", [3,3,32,64],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4", [3,3,64,64],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 3

    W5 = tf.get_variable("W5", [3,3,64,128],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W6 = tf.get_variable("W6", [3,3,128,128],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 4

    W7 = tf.get_variable("W7", [3,3,128,256],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W8 = tf.get_variable("W8", [3,3,256,256],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 5

    W9 = tf.get_variable("W9", [3,3,256, 512],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W10 = tf.get_variable("W10", [32,32,512,512],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 6

    W11 = tf.get_variable("W11", [3,3,512, 256],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W12 = tf.get_variable("W12", [3,3,256, 256],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 7

    W13 = tf.get_variable("W13", [3,3,256,128],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W14 = tf.get_variable("W14", [3,3,128,128],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 8

    W15 = tf.get_variable("W15", [3,3,128,64],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W16 = tf.get_variable("W16", [3,3,64,64],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 9

    W17 = tf.get_variable("W17", [3,3,64,32],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))
    W18 = tf.get_variable("W18", [3,3,32,32],
                         initializer=\
                         tf.contrib.layers.xavier_initializer(seed = 0))

    # Block 10

    W19 = tf.get_variable('W19', [1,1,32, 12],
                          initializer=\
                          tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {
        'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4, 'W5': W5, 'W6': W6,
        'W7': W7, 'W8': W8, 'W9': W9, 'W10':W10, 'W11': W11, 'W12': W12,
        'W13': W13, 'W14': W14, 'W15': W15, 'W16': W16, 'W17': W17,
        'W18': W18, 'W19': W19,
    }
    
    return parameters

def pool_block(X, Wa, Wb):
    """ This is like a mini forward propagation module. """
    Za = tf.nn.conv2d(X, Wa, strides=(1,1,1,1), padding='SAME')
    Aa = tf.nn.leaky_relu(Za, alpha=0.2)
    Zb = tf.nn.conv2d(Aa, Wb, strides=(1,1,1,1), padding='SAME')
    Ab = tf.nn.leaky_relu(Zb, alpha=0.2)
    P = tf.nn.max_pool(Ab, ksize=[1,2,2,1],
                        strides=[1,1,1,1], padding='SAME')
    return P

def conv_block(X, Wa, Wb):
    Za = tf.nn.conv2d(X, Wa, strides=(1,1,1,1), padding='SAME')
    Aa = tf.nn.leaky_relu(Za, alpha=0.2)
    Zb = tf.nn.conv2d(Aa, Wb, strides=(1,1,1,1), padding='SAME')
    Ab = tf.nn.leaky_relu(Zb, alpha=0.2)
    return Ab

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(
        tf.truncated_normal([pool_size, pool_size, output_channels,
                             in_channels], stddev=0.02)
    )
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2),
                                    strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def upsample_block(B, Bp, output_channels, in_channels, Wa, Wb):

    U = upsample_and_concat(B, Bp, output_channels, in_channels)
    Za = tf.nn.conv2d(U, Wa, strides=(1,1,1,1), padding='SAME')
    Aa = tf.nn.leaky_relu(Za, alpha=0.2)
    Zb = tf.nn.conv2d(Aa, Wb, strides=(1,1,1,1), padding='SAME')
    Ab = tf.nn.leaky_relu(Zb, alpha=0.2)
    return Ab

def output_block(X, W):

    Z = tf.nn.conv2d(X, W, strides=(1,1,1,1), padding='SAME')
    logger.debug("output block shape Z: {}".format(Z.get_shape()))
    Out = tf.depth_to_space(Z, 2)
    logger.debug("output block transform: {}".format(Out.get_shape()))
    return Out

def forward_propagation(X, parameters):

    # Block 1
    W1, W2 = parameters['W1'], parameters['W2']
    B1 = pool_block(X, W1, W2)
    logger.debug('B1 shape: {}'.format(B1.get_shape()))

    # Block 2
    W3, W4 = parameters['W3'], parameters['W4']
    B2 = pool_block(B1, W3, W4)
    logger.debug('B2 shape: {}'.format(B2.get_shape()))
    
    # Block 3
    W5, W6 = parameters['W5'], parameters['W6']
    B3 = pool_block(B2, W5, W6)
    logger.debug('B3 shape: {}'.format(B3.get_shape()))

    # Block 4
    W7, W8 = parameters['W7'], parameters['W8']
    B4 = pool_block(B3, W7, W8)
    logger.debug('B4 shape: {}'.format(B4.get_shape()))

    # Block 5
    W9, W10 = parameters['W9'], parameters['W10']
    B5 = conv_block(B4, W9, W10)
    logger.debug('B5 shape: {}'.format(B5.get_shape()))

    # Block 6
    W11, W12 = parameters['W11'], parameters['W12']
    B6 = upsample_block(B5, B4, 256, 512, W11, W12)
    logger.debug('B6 shape: {}'.format(B6.get_shape()))

    # Block 7
    W13, W14 = parameters['W13'], parameters['W14']
    B7 = upsample_block(B6, B3, 128, 256, W13, W14)
    logger.debug('B7 shape: {}'.format(B7.get_shape()))

    # Block 8
    W15, W16 = parameters['W15'], parameters['W16']
    B8 = upsample_block(B7, B2, 64, 128, W15, W16)
    logger.debug('B8 shape: {}'.format(B8.get_shape()))

    # Block 9
    W17, W18 = parameters['W17'], parameters['W18']
    B9 = upsample_block(B8, B1, 32, 64, W17, W18)
    logger.debug('B9 shape: {}'.format(B9.get_shape()))

    # Block 10
    W19 = parameters['W19']
    Bout = output_block(B9, W19)
    logger.debug('Bout shape: {}'.format(Bout.get_shape()))

    return Bout

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def sony_network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')
    logger.debug("pool1: {}".format(pool1.get_shape()))

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
    logger.debug("pool2: {}".format(pool2.get_shape()))

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')
    logger.debug("pool3: {}".format(pool3.get_shape()))

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')
    logger.debug("pool4: {}".format(pool4.get_shape()))

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    logger.debug('conv5: {}'.format(conv5.get_shape()))

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')
    logger.debug('conv6: {}'.format(conv6.get_shape()))

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')
    logger.debug('conv7: {}'.format(conv7.get_shape()))

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')
    logger.debug('conv8: {}'.format(conv8.get_shape()))

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')
    logger.debug('conv9: {}'.format(conv9.get_shape()))

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    logger.debug('out shape: {}'.format(out.get_shape()))
    return out

def compute_cost(Z, Y):
    """
    Computes the cost 

    Arguments:
    Z -- output of forward propagation
    Y -- "true" image as a label, same shape as Z

    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.abs(Z - Y))
    return cost

def random_mini_batches(m, minibatch_size, seed):
    """ Generator yielding minibatches """
    ids = [i for i in range(m)]
    np.random.seed(seed)
    np.random.shuffle(ids)

    for id_val in ids:
        yield id_val

def get_image_patch(ps=512):
    """ Each input image is processed using an image patch. """

    ratio = 300 # in_exposure/gt_exposure; ignored here

    # figure out which dimensions their using from their data.


def cifar_model(X_train, Y_train, X_test, Y_test, learning_rate=1e-4,
                num_epochs=1, minibatch_size=32, print_cost=True):
    """ 
    Implements Learn-to-See-in-the-Dark for CIFAR dataset

    Arguments:
    X_train -- training set, of shape (None, 32, 32, 3)
    Y_train -- test set, of shape (None, 32, 32, 3)
    X_test -- training set, of shape (None, 32, 32, 3)
    Y_test -- test set, of shape (None, 32, 32, 3)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can be used to predict.
    costs -- list of costs generated by the model
    learning_rate -- scalar real number, learning rate used by model
    """
    ops.reset_default_graph() # be able to rerun the model 
    tf.set_random_seed(42)
    seed = 42
    #(m, n_H, n_W, n_C) = X_train.shape
    assert X_train.shape == Y_train.shape
    costs = []

    #X,Y = create_placeholders(n_H, n_W, n_C)
    #parameters = initialize_parameters()
    #Z = forward_propagation(X, parameters)
    X, Y = X_train, Y_train
    
    Z = sony_network(X)

    cost = compute_cost(Z, Y)

    # Backpropagation

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop

        minibatches = random_mini_batches(m, minibatch_size, seed)
        
        for epoch in range(num_epochs):

            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            logger.debug('num minibatches: {}'.format(num_minibatches))
            seed = seed + 1
            minibatches = random_mini_batches(m, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                minibatch_X = X_train[minibatch,...]
                minibatch_Y = Y_train[minibatch,...]

                _, temp_cost = sess.run([optimizer, cost],
                                        feed_dict={X: minibatch_X,
                                                   Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

                logger.debug('temp_cost: {}, minibatch cost: {}'.\
                             format(temp_cost, num_minibatches))

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                logger.info("Cost after epoch %i: %f" % (epoch,
                                                         minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

    return (parameters, costs, learning_rate)


def plot_costs(costs, learning_rate, filepath):
    """ plot the cost from forward propagation """
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig(filepath)
