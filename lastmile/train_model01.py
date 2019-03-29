
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam

from model01 import simple_sony

# Dataset of 50,000 32x32 color training images, 
# labeled over 10 categories, and 10,000 test images.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train[0:25,...]
X_test = X_test[0:25,...]

# Compiling model using Keras

learning_rate = 1e-4
model = simple_sony()
opt = Adam(lr=1e-4)

model.compile(optimizer=opt,
              loss='mae',
              metrics=['accuracy'])

model.fit(X_train, X_train, epochs=10, batch_size=5)


