import numpy as np

import os

import keras
from keras_uncertainty.models import StochasticClassifier, DeepEnsembleClassifier

from tensorflow.keras.datasets import fashion_mnist

import utils.preact_resnet18_bayes as pa18

import argparse

import logging

from utils.data import get_fmnist, get_noise


# def get_fmnist():
#     (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#     x_train = np.expand_dims(x_train, axis=-1).astype(np.float32) / 255
#     x_test = np.expand_dims(x_test, axis=-1).astype(np.float32) / 255
#     y_train = keras.utils.to_categorical(y_train)
#     y_test = keras.utils.to_categorical(y_test)

#     return (x_train, x_test), (y_train, y_test)

# def get_noise(shape, std:float):
#     return np.random.randn(*shape) * std

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == y_true)

def train_standard_model(x_train_with_noise, y_train, x_test_with_noise, y_test, epochs=15, batch_size=256):
    model = pa18.get_standard_preact_resnet18(x_train[0].shape)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train_with_noise, y_train, verbose=1, epochs=epochs, batch_size=batch_size)
    train_accuracy = history.history["accuracy"][-1]

    history = model.evaluate(x_test_with_noise, y_test)
    test_accuracy = history[1]

    logging.info(f"Standard, Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")

    return model

def train_dropout_model(x_train_with_noise, y_train, x_test_with_noise, y_test, epochs=15, batch_size=256):
    model = pa18.get_dropout_preact_resnet18(x_train[0].shape)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train_with_noise, y_train, verbose=1, epochs=epochs, batch_size=batch_size)
    train_accuracy = history.history["accuracy"][-1]
    model_mc = StochasticClassifier(model, num_samples=25)

    history = model.evaluate(x_test_with_noise, y_test)
    test_accuracy = history[1]

    logging.info(f"Dropout, Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")

    return model_mc

def train_dropconnect_model(x_train_with_noise, y_train, x_test_with_noise, y_test, epochs=15, batch_size=256):
    model = pa18.get_dropconnect_preact_resnet18(x_train[0].shape)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train_with_noise, y_train, verbose=1, epochs=epochs, batch_size=batch_size)
    train_accuracy = history.history["accuracy"][-1]
    model_mc = StochasticClassifier(model, num_samples=25)

    history = model.evaluate(x_test_with_noise, y_test)
    test_accuracy = history[1]

    logging.info(f"DropConnect, Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")

    return model_mc

def train_flipout_model(x_train_with_noise, y_train, x_test_with_noise, y_test, epochs=15, batch_size=256):
    num_batches = len(x_train_with_noise) // batch_size + (0 if len(x_train_with_noise) % batch_size == 0 else 1)
    model = pa18.get_flipout_preact_resnet18(x_train[0].shape, num_batches)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train_with_noise, y_train, verbose=1, epochs=epochs, batch_size=batch_size)
    train_accuracy = history.history["accuracy"][-1]
    model_mc = StochasticClassifier(model, num_samples=25)

    history = model.evaluate(x_test_with_noise, y_test)
    test_accuracy = history[1]

    logging.info(f"Filpout, Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")

    return model_mc

def train_ensemble_model(x_train_with_noise, y_train, x_test_with_noise, y_test, epochs=15, batch_size=256):
    def model_fn():
        model = pa18.get_standard_preact_resnet18(x_train[0].shape)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    ensemble = DeepEnsembleClassifier(model_fn, num_estimators=5)
    history = ensemble.fit(x_train_with_noise, y_train, verbose=1, epochs=epochs, batch_size=batch_size)


    history = ensemble.evaluate(x_test_with_noise, y_test)
    test_accuracy = history[1]
    logging.info(f"Ensemble, Train accuracy: -- , Test accuracy: {test_accuracy:.4f}")
    



STD_TRAIN = 0.1
STD_TEST = [0.0, 0.1, 0.2, 0.3]
WEIGHTS_SAVE_FOLDER = "weights"



if __name__ == "__main__":
    # args = parse_args()
    # print(args)

    logging.basicConfig(
        filename="logging.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode="a"
    )

    os.makedirs(WEIGHTS_SAVE_FOLDER, exist_ok=True)

    (x_train, x_test), (y_train, y_test) = get_fmnist()
    noise_level = STD_TRAIN
    noise_train = get_noise(x_train.shape, noise_level)
    noise_test = get_noise(x_test.shape, noise_level)
    
    # standard = train_standard_model((x_train, noise_train), y_train, (x_test, noise_test), y_test)
    # standard.save_weights(os.path.join(WEIGHTS_SAVE_FOLDER, "standard.weights"))

    dropout = train_dropout_model((x_train, noise_train), y_train, (x_test, noise_test), y_test)
    dropout.model.save_weights(os.path.join(WEIGHTS_SAVE_FOLDER, "dropout.weights"))

    dropconn = train_dropconnect_model((x_train, noise_train), y_train, (x_test, noise_test), y_test)
    dropconn.model.save_weights(os.path.join(WEIGHTS_SAVE_FOLDER, "dropconn.weights"))

    # flip = train_flipout_model((x_train, noise_train), y_train, (x_test, noise_test), y_test)
    # flip.save_weights("filpout.weights")

    # ensemble = train_ensemble_model((x_train, noise_train), y_train, (x_test, noise_test), y_test)
    # ensemble.save_weights("ensemble.weights")


    
    

