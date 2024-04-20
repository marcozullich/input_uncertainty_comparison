import numpy as np

import keras
from keras_uncertainty.models import StochasticClassifier

from tensorflow.keras.datasets import fashion_mnist

import utils.preact_resnet18_bayes as pa18

def get_fmnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32) / 255
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32) / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    return (x_train, x_test), (y_train, y_test)

def get_noise(shape, std:float):
    return np.random.randn(*shape) * std

def train_standard_model(x_train_with_noise, y_train):
    model = pa18.get_standard_preact_resnet18([28, 28, 1])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train_with_noise, y_train, verbose=1, epochs=15, batch_size=256)
    return model

def train_dropout_model(x_train_with_noise, y_train):
    model = pa18.get_dropout_preact_resnet18([28, 28, 1])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train_with_noise, y_train, verbose=1, epochs=15, batch_size=256)
    model_mc = StochasticClassifier(model, num_samples=25)
    return model_mc


if __name__ == "__main__":
    (x_train, x_test), (y_train, y_test) = get_fmnist()
    for noise_level in (0.0, 0.05, 0.5):
        print(f"### noise level {noise_level}###")
        noise_train = get_noise(x_train.shape, noise_level)
        noise_test = get_noise(x_test.shape, noise_level)
        model = train_dropout_model((x_train, noise_train), y_train)
        model.evaluate((x_test, noise_test), y_test)
    

