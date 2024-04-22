import numpy as np

import keras
from keras import datasets

def get_fmnist():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32) / 255
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32) / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return (x_train, x_test), (y_train, y_test)

def get_noise(shape, std:float):
    return np.random.randn(*shape) * std
