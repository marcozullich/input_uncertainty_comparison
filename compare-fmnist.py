import numpy as np
import keras
import logging
import os

import matplotlib.pyplot as plt

from typing import Tuple

from keras.datasets import fashion_mnist

from keras_uncertainty.models import StochasticClassifier

import utils.preact_resnet18_bayes as pa18
from utils.data import get_fmnist, get_noise

def numpy_entropy(probs, axis=-1, eps=1e-6):
    return -np.sum(probs * np.log(probs + eps), axis=axis)

def uncertainty(probs:np.ndarray):
    return numpy_entropy(probs, axis=-1)

def get_uncertainty_base_model(model:keras.Model, x_test_with_noise:Tuple[np.ndarray, np.ndarray]):
    predict_probs = model.predict(x_test_with_noise)
    uncertainty_values = uncertainty(predict_probs)
    return uncertainty_values

def get_uncertainty_ensemble(ensemble:keras.Model, x_test_with_noise:Tuple[np.ndarray, np.ndarray]):
    raise NotImplementedError("Not implemented yet")


STD_EVAL = [0.0, 0.1, 0.2, 0.3]
WEIGHTS_FOLDER = "./weights"
WEIGHTS_FILENAMES = {
    "standard": [pa18.get_standard_preact_resnet18, "standard.weights"],
    "dropout": [pa18.get_dropout_preact_resnet18, "dropout.weights"],
    "dropconnect": [pa18.get_dropconnect_preact_resnet18, "dropconn.weights"],
    # "flipout": [pa18.get_flipout_preact_resnet18, "flipout.weights"],
    # "ensemble": [pa18.get_standard_preact_resnet18, "standard.weights"]
}
FIG_SAVEPATH = "fmnist_uncertainty.pdf"

if __name__ == "__main__":
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # logging.getLogger('tensorflow').disabled = True

    (_, x_test), (_, y_test) = get_fmnist()
    # x_test = np.expand_dims(x_test, axis=-1).astype(np.float32) / 255
    # y_test = keras.utils.to_categorical(y_test)

    mean_uncertainty = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for method, (model_constructor, filename) in WEIGHTS_FILENAMES.items():
        print(f"**** Method: {method} ****")
        model = model_constructor(x_test[0].shape)
        model.load_weights(os.path.join(WEIGHTS_FOLDER, filename))

        if method != "standard":
            model = StochasticClassifier(model, num_samples=10)

        mean_uncertainty[method] = []

        for std in STD_EVAL:
            print(f"   Standard deviation of noise: {std}")
            noise_test = get_noise(x_test.shape, std)
            x_test_with_noise = (x_test, noise_test)
            uncertainty_values = get_uncertainty_base_model(model, x_test_with_noise)

            mean_uncertainty[method].append(np.mean(uncertainty_values))
        
        print(mean_uncertainty[method])
        ax.scatter(STD_EVAL, mean_uncertainty[method], label=method)
        ax.plot(STD_EVAL, mean_uncertainty[method]  )
    
    ax.set_xlabel("Standard deviation of noise")
    ax.set_ylabel("Mean entropy")
    ax.legend()
    plt.savefig(FIG_SAVEPATH, format="pdf", bbox_inches="tight")
            
