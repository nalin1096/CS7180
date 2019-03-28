""" Custom layers implemented in Keras

"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class UpsampleConcat(Layer):
    """ Perform upsample_and_concat function from LSD paper. """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(UpsampleConcat, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        # trainable weight variable for this layer

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1],
                                             self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(UpsampleConcat, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def comput_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]

    
        
