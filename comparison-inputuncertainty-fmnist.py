import numpy as np

import keras

from tensorflow.keras.datasets import fashion_mnist

import utils.preact_resnet18_bayes as pa18

def get_fmnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32) / 255
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32) / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    return (x_train, x_test), (y_train, y_test)

def train_standard_model(x_train, y_train):
    model = pa18.get_standard_preact_resnet18()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, verbose=1, epochs=15, batch_size=256)
    return model


if __name__ == "__main__":
    (x_train, x_test), (y_train, y_test) = get_fmnist()
    model = train_standard_model(x_train, y_train)
    model.evaluate(x_test, y_test)
