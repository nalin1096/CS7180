from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Sequential



def model03():
    """ Removed upsampling from model01. """
    model_id = 'model03'

    lrelu = ReLU()
    model = Sequential()

    # Block 1
    model.add(Conv2D(8, (3,3), activation=lrelu,
                        padding='same', input_shape=(256,256,3)))
    model.add(Conv2D(8, (3,3), padding='same', activation=lrelu))

    # Block 2
    model.add(Conv2D(16, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(16, (3,3), padding='same', activation=lrelu))

    # Block 3
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))
    model.add(Conv2D(32, (3,3), padding='same', activation=lrelu))

    # Block 4
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(3))

    mod = {"model": model, "model_id": model_id}
    return mod
