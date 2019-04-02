""" We need to use a custom loss function for Keras. """
from tensorflow.keras import backend as K

def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=None)
