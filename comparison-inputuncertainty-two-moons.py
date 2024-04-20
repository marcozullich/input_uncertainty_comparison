import numpy as np

#import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Concatenate
from keras.utils import to_categorical

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleClassifier, StochasticClassifier, GradientClassificationConfidence
from keras_uncertainty.layers import StochasticDropout, duq_training_loop, add_gradient_penalty, add_l2_regularization
from keras_uncertainty.layers import DropConnectDense, VariationalDense, RBFClassifier, FlipoutDense
from keras_uncertainty.utils import numpy_entropy

import matplotlib.pyplot as plt

#keras.config.disable_traceback_filtering()

import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real

from sklearn.utils import check_random_state, shuffle
util_shuffle = shuffle

def make_moons(n_samples=100, *, shuffle=True, noise=None, random_state=None):
    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e

    generator = check_random_state(random_state)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X_noise = generator.normal(scale=noise, size=X.shape)

        return X, X_noise, y

    return X, y

def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)

def base_siamese_model(input_shape, bayesian_layer, num_classes=2, intermediate_layer=None):
    inp_mean = Input(shape=input_shape, name="mean")
    inp_std = Input(shape=input_shape, name="std")

    x_m = Dense(10, activation="relu")(inp_mean)
    x_m = Dense(10, activation="relu")(x_m)

    x_s = Dense(10, activation="relu")(inp_std)
    x_s = Dense(10, activation="relu")(x_s)

    x = Concatenate()([x_m, x_s])
    
    x = bayesian_layer(20, activation="relu")(x)
    if intermediate_layer is not None:
        x = intermediate_layer()(x)

    x = bayesian_layer(20, activation="relu")(x)
    if intermediate_layer is not None:
        x = intermediate_layer()(x)

    out = bayesian_layer(num_classes, activation="softmax")(x)

    model = Model([inp_mean, inp_std], [out])

    return model

def train_standard_model(x_train, y_train, domain):
    model = base_siamese_model((2,), Dense)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=100)

    return model

def train_dropout_model(x_train, y_train, domain, prob=0.2):
    model = base_siamese_model((2,), Dense, intermediate_layer=lambda: StochasticDropout(0.2))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=100)

    mc_model = StochasticClassifier(model, num_samples=100)

    return mc_model

def train_dropconnect_model(x_train, y_train, domain, prob=0.05):
    model = base_siamese_model((2,), DropConnectDense)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=100)

    mc_model = StochasticClassifier(model, num_samples=100)

    return mc_model

def train_ensemble_model(x_train, y_train, domain):
    def model_fn():
        model = base_siamese_model((2,), Dense)

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    model = DeepEnsembleClassifier(model_fn, num_estimators=5)
    model.fit(x_train, y_train, verbose=2, epochs=100)
    
    return model

def train_bayesbackprop_model(x_train, y_train, domain):
    num_batches = x_train[0].shape[0] / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0, 
        'prior_sigma_2': 2.0, 
        'prior_pi': 0.5
    }

    def bayes_layer(neurons, activation="relu"):
        return VariationalDense(neurons, kl_weight, **prior_params, activation=activation)

    model = base_siamese_model((2,), bayes_layer)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=500)

    mc_model = StochasticClassifier(model, num_samples=100)

    model.fit(x_train, y_train, verbose=2, epochs=1000)
    st_model = StochasticClassifier(model)

    return st_model

def train_flipout_model(x_train, y_train, domain):
    num_batches = x_train[0].shape[0] / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0, 
        'prior_sigma_2': 2.0, 
        'prior_pi': 0.5
    }

    def bayes_layer(neurons, activation="relu"):
        return FlipoutDense(neurons, kl_weight, **prior_params, activation=activation)

    model = base_siamese_model((2,), bayes_layer)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy", "sparse_categorical_crossentropy"])

    model.fit(x_train, y_train, verbose=2, epochs=300)
    st_model = StochasticClassifier(model, num_samples=100)

    return st_model

def train_duq_model(x_train, y_train, domain, num_classes=2):
    
    inp_mean = Input(shape=(2,), name="mean")
    inp_std = Input(shape=(2,), name="std")

    x_m = Dense(10, activation="relu")(inp_mean)
    x_m = Dense(10, activation="relu")(x_m)

    x_s = Dense(10, activation="relu")(inp_std)
    x_s = Dense(10, activation="relu")(x_s)

    x = Concatenate()([x_m, x_s])
    x = Dense(20, activation="relu")(x)
    
    out = RBFClassifier(num_classes, 0.1)(x)

    model = Model([inp_mean, inp_std], [out])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    add_gradient_penalty(model, lambda_coeff=0.5)
    add_l2_regularization(model)

    y_train = to_categorical(y_train, num_classes=2)

    model.fit(x_train, y_train, verbose=2, epochs=100)

    return model

def train_gradient_model(x_train, y_train, domain):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    y_train = to_categorical(y_train, num_classes=2)
    model.fit(x_train, y_train, verbose=2, epochs=50)

    grad_model = GradientClassificationConfidence(model, aggregation="l1_norm")
    conf = grad_model.predict(domain)

    return np.array(conf)

METHODS = {
    "Classical NN": train_standard_model,
    "Dropout": train_dropout_model,
    "DropConnect": train_dropconnect_model,
    "5 Ensembles": train_ensemble_model,
    "Flipout": train_flipout_model,
    "DUQ": train_duq_model,
    #"Gradient L1": train_gradient_model
}

NUM_SAMPLES = 30
NOISE_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]

if __name__ == "__main__":    
    x_mean, x_std, y = make_moons(n_samples=2000, noise=0.2)

    x_std_noisy = {}

    for noise_val in NOISE_VALUES:
        _, std, __ = make_moons(n_samples=1000, noise=noise_val)
        x_std_noisy[noise_val] = std

    min_x, max_x = [-2, -2] , [3, 2]
    res = 0.08

    xx, yy = np.meshgrid(np.arange(min_x[0], max_x[0], res), np.arange(min_x[1], max_x[1], res))
    domain = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(ncols=len(NOISE_VALUES), nrows=len(METHODS.keys()), figsize=(3 * len(NOISE_VALUES), 2 * len(METHODS.keys())))
    methods = list(METHODS.keys())

    # Want to add more plots of input uncertainty vs output uncertainty

    for i, key in enumerate(methods):
        model = METHODS[key]([x_mean, x_std], y, domain)

        for j, noise_val in enumerate(NOISE_VALUES):
            ax = axes[i, j]

            #x_noise = x_std_noisy[noise_val]
            x_noise = np.random.normal(scale=noise_val, size=domain.shape)

            domain_conf = model.predict([domain, x_noise])

            if key is "DUQ":
                domain_conf = np.max(domain_conf, axis=1)
            else:
                domain_conf = uncertainty(domain_conf)

            domain_conf = domain_conf.reshape(xx.shape)

            x = x_mean + x_std

            cont = ax.contourf(xx, yy, domain_conf, vmin=0.0, vmax=np.log(2.0))

            if j == 0:
                scat = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="binary")

            ax.set_title("$\sigma = {:.1f}$".format(noise_val))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if j == 0:
                ax.set_ylabel(key)

    #plt.tight_layout()
    plt.savefig("inp-uncertainty-two-moons.pdf", bbox_inches="tight")
    plt.show()
