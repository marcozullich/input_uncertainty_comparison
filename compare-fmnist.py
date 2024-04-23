import numpy as np
import keras
import logging
import os

import matplotlib.pyplot as plt

from typing import Tuple

import tensorflow

from keras.datasets import fashion_mnist

from keras_uncertainty.models import StochasticClassifier, DeepEnsembleClassifier
from keras_uncertainty.utils import classifier_calibration_error

import utils.preact_resnet18_bayes as pa18
from utils.data import get_fmnist, get_noise

def numpy_entropy(probs, axis=-1, eps=1e-6):
    return -np.sum(probs * np.log(probs + eps), axis=axis)

def uncertainty(predict_probs:np.ndarray):
    return numpy_entropy(predict_probs, axis=-1)

def predicted_class(predict_probs:np.ndarray):
    return np.argmax(predict_probs, axis=-1)

def confidence(predict_probs:np.ndarray):
    return np.max(predict_probs, axis=-1)

def get_predictions(model:keras.Model, x_test_with_noise:Tuple[np.ndarray, np.ndarray], method:str, noise_level:float):
    filename = PREDICTIONS_SAVEPATH.format(method, noise_level)
    run_predictions = True
    if PREDICTIONS_LOAD_IF_EXISTS:
        try:
            predict_probs = np.load(filename)
            run_predictions = False
        except FileNotFoundError:
            pass
    if run_predictions:
        predict_probs = model.predict(x_test_with_noise, batch_size=1024)
        if SAVE_PREDICTIONS:
            np.save(filename, predict_probs)
    return predict_probs

def get_uncertainty_base_model(predict_probs:np.ndarray):
    uncertainty_values = uncertainty(predict_probs)
    return uncertainty_values

# def get_uncertainty_ensemble(ensemble:keras.Model, x_test_with_noise:Tuple[np.ndarray, np.ndarray]):
#     raise NotImplementedError("Not implemented yet")


STD_EVAL = [0.0, 0.1, 0.2, 0.3]
WEIGHTS_FOLDER = "./weights"
WEIGHTS_FILENAMES = {
    "standard": [pa18.get_standard_preact_resnet18, "standard.weights"],
    "dropout": [pa18.get_dropout_preact_resnet18, "dropout.weights"],
    "dropconnect": [pa18.get_dropconnect_preact_resnet18, "dropconn.weights"],
    "flipout": [lambda size: pa18.get_flipout_preact_resnet18(size, batch_size=1, prior_params=None), "filpout.weights"],
    "ensemble": [pa18.get_standard_preact_resnet18, "ensemble.weights"]
}
SAVE_PREDICTIONS = True
PREDICTIONS_SAVEPATH = "./predictions/test_predictions_{}_{}.npy"
PREDICTIONS_LOAD_IF_EXISTS = True
FIG_SAVEPATH = "fmnist_uncertainty_{}.pdf"

if __name__ == "__main__":
    tensorflow.compat.v1.disable_eager_execution()

    if SAVE_PREDICTIONS:
        os.makedirs(os.path.dirname(PREDICTIONS_SAVEPATH), exist_ok=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').disabled = True

    (_, x_test), (_, y_test) = get_fmnist()
    # x_test = np.expand_dims(x_test, axis=-1).astype(np.float32) / 255
    # y_test = keras.utils.to_categorical(y_test)

    # mean_uncertainty = {}

    confidence_data = {}
    ece_data = {}

    for method, (model_constructor, filename) in WEIGHTS_FILENAMES.items():
        print(f"**** Method: {method} ****")


        if method == "ensemble":
            def model_fn():
                model = model_constructor(x_test[0].shape)
                model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
                return model

            model = DeepEnsembleClassifier(model_fn, num_estimators=5)
        else:
            model = model_constructor(x_test[0].shape)

        model.load_weights(os.path.join(WEIGHTS_FOLDER, filename))

        if method not in ("standard", "ensemble"):
            model = StochasticClassifier(model, num_samples=10)

        # mean_uncertainty[method] = []
        confidence_data[method] = {"noise": [], "mean": [], "std": []}
        ece_data[method] = {"noise": [], "ece": []}

        for std in STD_EVAL:
            print(f"   Standard deviation of noise: {std}")
            noise_test = get_noise(x_test.shape, std)
            x_test_with_noise = (x_test, noise_test)
            predicted_probs = get_predictions(model, x_test_with_noise, method, std)
            # uncertainty_values = get_uncertainty_base_model(model, x_test_with_noise, method, std)
            confidence_values = confidence(predicted_probs)
            
            confidence_data[method]["noise"].append(std)
            confidence_data[method]["mean"].append(confidence_values.mean())
            confidence_data[method]["std"].append(confidence_values.std())

            ece = classifier_calibration_error(predicted_class(predicted_probs), predicted_class(y_test), confidence_values, weighted=True)
            ece_data[method]["noise"].append(std)
            ece_data[method]["ece"].append(ece)
    
    plt.figure()
    for i, key in enumerate(WEIGHTS_FILENAMES.keys()):
        plt.plot(ece_data[key]["noise"], ece_data[key]["ece"], label=key)
    plt.title("Expected Calibration Error as Function of $\sigma$")
    plt.xlabel("$\sigma$")
    plt.ylabel("ECE")
    plt.legend()    
    plt.savefig(FIG_SAVEPATH.format("ece"), format="pdf", bbox_inches="tight")

    plt.figure()
    for i, key in enumerate(WEIGHTS_FILENAMES.keys()):
        yerr = np.array(confidence_data[key]["std"]) / 2.0
        plt.errorbar(confidence_data[key]["noise"], confidence_data[key]["mean"], yerr=yerr, label=key)
    
    plt.title("Output confidence as Function of $\sigma$")
    plt.xlabel("$\sigma$")
    plt.ylabel("Output confidence")
    plt.legend()
    plt.savefig(FIG_SAVEPATH.format("inout"), bbox_inches="tight")
            
