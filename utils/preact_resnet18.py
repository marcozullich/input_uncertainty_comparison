import keras
import keras.backend
from keras.layers import Conv2D, BatchNormalization, Input, Add, Concatenate, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense
import tensorflow as tf

from keras_uncertainty.layers import RBFClassifier

'''
Implementation inspired by https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras
'''

class PreActResidualBlock(keras.Model):
    def __init__(self, bayesian_layer, channels:int, kernel_size:int=3, downsample:bool=False):
        super().__init__()
        self._channels = channels
        self._kernel_size = kernel_size
        self._downsample = downsample

        self._stride = [2, 1] if downsample else [1, 1]

        self.bn1 = BatchNormalization()
        self.conv1 = bayesian_layer(channels, strides=self._stride[0], kernel_size=kernel_size, padding="same")

        self.bn2 = BatchNormalization()
        self.conv2 = bayesian_layer(channels, strides=self._stride[1], kernel_size=kernel_size, padding="same")

        if downsample:
            self.bn_downsample = BatchNormalization()
            self.conv_downsample = bayesian_layer(channels, strides=2, kernel_size=1, padding="same")
        
        self.merge = Add()
    
    def call(self, x):
        residual = x
        
        x = self.bn1(x)
        x = tf.keras.backend.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = tf.keras.backend.relu(x)
        x = self.conv2(x)

        if self._downsample:
            residual = self.bn_downsample(residual)
            residual = tf.keras.backend.relu(residual)
            residual = self.conv_downsample(residual)
        
        out = self.merge([x, residual])
        out = tf.keras.backend.relu(out)
        return out

    def compute_output_shape(self, input_shape):
        if self._downsample:
            return (input_shape[0], input_shape[1] // 2 + 1, input_shape[2] // 2 + 1, self._channels)
        return input_shape

# def two_inputs_base_model(input_mean, input_std):
#     x_mean = Conv2D(32, kernel_size=7, strides=2, padding="same")(input_mean)
#     x_std = Conv2D(32, kernel_size=7, strides=2, padding="same")(input_std)

#     x = Concatenate([x_mean, x_std])
#     model = keras.Model([])
#     return model

def two_inputs_preact_resnet18(input_shape, bayesian_layer, num_classes=10, lightweight_version:bool=False, output_head="dense"):
    if output_head == "dense":
        output_layer = Dense(num_classes, activation="softmax")
    elif output_head == "duq":
        output_layer = RBFClassifier(num_classes, length_scale=.1)
    else:
        raise AttributeError(f"Invalid output head: {output_head}. Must be either 'dense' or 'duq'.")
    standard_layer = Conv2D if lightweight_version else bayesian_layer

    input_mean = Input(shape=input_shape, name="mean")
    input_std = Input(shape=input_shape, name="std")
 
    # base layers
    x_mean = Conv2D(32, kernel_size=7, strides=2, padding="same")(input_mean)
    x_std = Conv2D(32, kernel_size=7, strides=2, padding="same")(input_std)

    x = Concatenate()([x_mean, x_std])

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2), strides=2, padding="same")(x)

    # residual blocks
    x = PreActResidualBlock(standard_layer, 64)(x)
    x = PreActResidualBlock(standard_layer, 64)(x)
    x = PreActResidualBlock(standard_layer, 128, downsample=True)(x)
    x = PreActResidualBlock(standard_layer, 128)(x)
    x = PreActResidualBlock(standard_layer, 256, downsample=True)(x)
    x = PreActResidualBlock(standard_layer, 256)(x)
    x = PreActResidualBlock(standard_layer, 512, downsample=True)(x)
    x = PreActResidualBlock(bayesian_layer, 512)(x)

    # classification head
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    out = output_layer(x)

    model = keras.Model([input_mean, input_std], [out])

    return model


