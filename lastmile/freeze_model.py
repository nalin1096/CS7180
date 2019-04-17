""" Transfer learning by freezing model layers

"""
from datetime import datetime
import json
import logging
import os
from urllib.parse import urljoin

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from model_utils import enable_cloud_log
from custom_loss import mean_absolute_error
from image_preprocessing import (ImageDataPipeline, SonyDataGenerator)
from test_model import restore_model
from model_utils import create_patch

from model05 import functional_sony
from model_utils import callbacks

logger = logging.getLogger(__name__)

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)

def freeze_sony_model(model):
    """ Must use functional sony model. """

    model_id = 'freeze_sony'

    lrelu = LeakyReLU(alpha=0.2)

    # Block 0

    inputs = Input(shape=(64,64,3))
    x0 = Lambda(lambda x: create_patch(x))(inputs)

    ## Block 1

    layer1a = model.get_layer(index=2)
    layer1a.trainable = False
    x1a = layer1a(x0)

    layer1b = model.get_layer(index=3)
    layer1b.trainable = False
    x1b = layer1b(x1a)

    layer1c = model.get_layer(index=4)
    layer1c.trainable = False
    x1c = layer1c(x1b)

    ## Block 2

    layer2a = model.get_layer(index=5)
    layer2a.trainable = False
    x2a = layer2a(x1c)

    layer2b = model.get_layer(index=6)
    layer2b.trainable = False
    x2b = layer2b(x2a)

    layer2c = model.get_layer(index=7)
    layer2c.trainable = False
    x2c = layer2c(x2b)

    ## Block 3

    layer3a = model.get_layer(index=8)
    layer3a.trainable = False
    x3a = layer3a(x2c)

    layer3b = model.get_layer(index=9)
    layer3b.trainable = False
    x3b = layer3b(x3a)

    layer3c = model.get_layer(index=10)
    layer3c.trainable = False
    x3c = layer3c(x3b)

    ## Block 4

    layer4a = model.get_layer(index=11)
    layer4a.trainable = False
    x4a = layer4a(x3c)

    layer4b = model.get_layer(index=12)
    layer4b.trainable = False
    x4b = layer4b(x4a)

    layer4c = model.get_layer(index=13)
    layer4c.trainable = False
    x4c = layer4c(x4b)

    ## Block 5

    layer5a = model.get_layer(index=14)
    layer5a.trainable = False
    x5a = layer5a(x4c)

    layer5b = model.get_layer(index=15)
    layer5b.trainable = False
    x5b = layer5b(x5a)

    ## Block 6

    layer6a = model.get_layer(index=16)
    layer6a.trainable = False
    x6a = layer6a(x5b)

    layer6b = model.get_layer(index=17)
    layer6b.trainable = False
    x6b = layer6b(x6a)

    layer6c = model.get_layer(index=18)
    layer6c.trainable = False
    x6c = layer6c(x6b)

    ## Block 7

    layer7a = model.get_layer(index=19)
    layer7a.trainable = False
    x7a = layer7a(x6c)

    layer7b = model.get_layer(index=20)
    layer7b.trainable = False
    x7b = layer7b(x7a)

    layer7c = model.get_layer(index=21)
    layer7c.trainable = False
    x7c = layer7c(x7b)

    ## Block 8

    layer8a = model.get_layer(index=22)
    layer8a.trainable = False
    x8a = layer8a(x7c)

    layer8b = model.get_layer(index=23)
    layer8b.trainable = False
    x8b = layer8b(x8a)

    layer8c = model.get_layer(index=24)
    layer8c.trainable = False
    x8c = layer8c(x8b)

    ## Block 9

    layer9a = model.get_layer(index=25)
    layer9a.trainable = True
    x9a = layer9a(x8c)

    layer9b = model.get_layer(index=26)
    layer9b.trainable = True
    x9b = layer9b(x9a)

    layer9c = model.get_layer(index=27)
    layer9c.trainable = True
    x9c = layer9c(x9b)

    ## Block 10

    layer10a = model.get_layer(index=28)
    layer10a.trainable = True
    x10a = layer10a(x9c)

    outputs = Lambda(lambda x : tf.depth_to_space(x, 2))(x10a)

    ## Define model

    model = Model(inputs=inputs, outputs=outputs)
    mod = {"model": model, "model_id": model_id}

    return mod

def train_frozen_model(train_dataflow, val_dataflow, epochs: int,
                       mod: dict, model_type: str, lr: float):
    """ Train frozen model using latest weights. """

    # Define model
    model_name = '{}_{}'.format(mod.get('model_id', ''),model_type)
    model = restore_model(mod, model_name)
    if model is None:
        raise TypeError("model must be defined: {}".format(model))

    # Update restored model to frozen layers

    frozen_mod = freeze_sony_model(model)
    frozen_model = frozen_mod.get('model', None)
    
    # Compile model

    opt = Adam(lr=lr)
    frozen_model.compile(optimizer=opt,
                         loss=mean_absolute_error,
                         metrics=['accuracy'])

    # Fit model

    frzn_name = model_name.replace(mod['model_id'], frozen_mod['model_id'])
    calls = callbacks(model_type=frzn_name)
    history = frozen_model.fit_generator(
        generator=train_dataflow,
        epochs=epochs,
        callbacks=calls,
        validation_data=val_dataflow
    )

    return frozen_model, history

def run_frozensony(mod: dict):
    logger.info("STARTED running frozen sony model updates.")

    # Specify Image Data Pipeline

    idp = ImageDataPipeline(preprocessing_function='sony',
                            stride=32,
                            batch_size=32,
                            patch_size=(64,64),
                            random_seed=42,
                            meanm_fpath='simulation_mean.pkl',
                            covm_fpath='simulation_cov.pkl',
                            num_images=10
    )
    
    # Specify train/val generators

    train_dir = 'raise/rgb/train/'
    y_train_set = [urljoin(train_dir, f) for f in os.listdir(train_dir)]
    train_dataflow = SonyDataGenerator(y_train_set, idp)

    val_dir = 'raise/rgb/val/'
    y_val_set = [urljoin(val_dir, f) for f in os.listdir(val_dir)]
    val_dataflow = SonyDataGenerator(y_val_set, idp)

    # Fit model

    mod = functional_sony()
    model_type = 'bl_cd_pn_ag'
    frozen_model, history = train_frozen_model(train_dataflow, val_dataflow,
                                               epochs=1, mod=mod,
                                               model_type=model_type,
                                               lr=1e-3)

    # Save history

    try:
        review_dir = os.path.join(os.getcwd(), 'review')
        if not os.path.isdir(review_dir):
            os.makedirs(review_dir)

        model_id = 'freeze_sony'
        model_name = '{}_{}'.format(model_id, model_type)

        datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_history_name = '{}_{}.json'.format(model_name, datetime_now)
        mh_filepath = os.path.join(review_dir, model_history_name)

        with open(mh_filepath, 'w') as outfile:
            json.dump(history.history, outfile)

        logger.info('Saved model history: {}'.format(mh_filepath))

    except Exception as exc:
        logger.exception(exc)



if __name__ == "__main__":

    mod = functional_sony()
    run_frozensony(mod)
