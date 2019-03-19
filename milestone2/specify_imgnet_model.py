""" Specify imagenet model.

"""
import logging

import numpy as np
import rawpy
import tensorflow as tf
from tensorflow.python.framework import ops


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

    X = tf.placeholder(tf.int32, shape=(None, n_H, n_W, n_C))
    Y = tf.placeholder(tf.int32, shape=(None, n_H, n_W, n_C))

    return X,Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
        []

    Returns:
    parameters -- a dictionary of tensors containing W1,...,WN
    """
    seed = tf.set_random_seed(42)

    W1 = tf.get_variable("W1", (32,32,3),tf.initializers.random_normal(seed=seed))

    parameters = {
        "W1": W1,
    }
    
    return parameters

def forward_propagation(X, parameters):

    # Weights for individual layers
    W1 = parameters["W1"]

    # Block 1
    Z1 = tf.nn.conv2d(X, W1, strides=(1,1,1,1), padding='SAME')
    A1 = tf.nn.leaky_relu(Z1, alpha=0.2)

    return A1

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

    begin = 0
    for end in range(0, m, minibatch_size):
        yield ids[begin:end]
        begin = end

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
    train_accuracy -- real number, accuracy o the train set (X_train)
    test_accuracy -- real number, accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can be used to predict.
    """
    ops.reset_default_graph() # be able to rerun the model without overwriting tf variables
    tf.set_random_seed(42)
    seed = 42
    (m, n_H, n_W, n_C) = X_train.shape
    assert X_train.shape == Y_train.shape
    costs = []

    X,Y = create_placeholders(n_H, n_W, n_C)
    parameters = initialize_parameters()
    Z = forward_propagation(X, parameters)

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
            seed = seed + 1
            minibatches = random_mini_batches(m, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run([optimizer, cost],
                                        feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                logger.info("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost

        # calculate the correct predictions

        # calculate accuracy on the test set
        train_accuracy = None
        test_accuracy = None

        return train_accuracy, test_accuracy, parameters
        
            
                
        
    
