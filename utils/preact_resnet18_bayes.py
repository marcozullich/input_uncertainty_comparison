from utils.preact_resnet18 import two_inputs_preact_resnet18

from keras.layers import Conv2D
from keras_uncertainty.layers import DropConnectConv2D, VariationalConv2D, FlipoutConv2D, StochasticDropout, DropConnectDense, FlipoutDense

from typing import List

class DropoutConv2D(Conv2D):
    def __init__(self, filters, kernel_size, p_drop:float, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, **kwargs)
        self._p_drop = p_drop
        self.dropout = StochasticDropout(rate=p_drop)
    
    def call(self, x):
        x = super().call(x)
        x = self.dropout(x)
        return x

def get_standard_preact_resnet18(input_shape, num_classes:int=10):
    return two_inputs_preact_resnet18(input_shape, Conv2D, num_classes)

def get_dropout_preact_resnet18(input_shape, num_classes:int=10, p_drop=0.1):
    return two_inputs_preact_resnet18(input_shape, lambda filters, kernel_size, strides, padding : DropoutConv2D(filters, strides=strides, kernel_size=kernel_size, padding=padding, p_drop=p_drop), num_classes, lightweight_version=True)

def get_dropconnect_preact_resnet18(input_shape, num_classes:int=10, p_drop:float=0.05):
    return two_inputs_preact_resnet18(input_shape, lambda filters, kernel_size, strides, padding : DropConnectConv2D(filters, strides=strides, kernel_size=kernel_size, padding=padding, prob=p_drop), num_classes, lightweight_version=True)

def get_flipout_preact_resnet18(input_shape, batch_size:int, prior_params:dict, num_classes:int=10):
    kl_weight = 1.0 / batch_size
    return two_inputs_preact_resnet18(input_shape, lambda filters, kernel_size, strides, padding : FlipoutConv2D(filters, strides=strides, kernel_size=kernel_size, padding=padding, kl_weight=kl_weight), num_classes, lightweight_version=True)

def get_duq_preact_resnet18(input_shape, num_classes:int=10):
    return two_inputs_preact_resnet18(input_shape, Conv2D, num_classes, lightweight_version=True, output_head="duq")

# def get_ensemble_preact_resnet18(input_shape, n_components:int, loss:str="categorical_crossentropy", optimizer:str="adam", metrics:List[str]=["accuracy"], num_classes:int=10):
#     def model_fn():
#         model = two_inputs_preact_resnet18(input_shape, Conv2D, num_classes)
#         model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#         return model
    
#     ensemble = DeepEnsembleClassifier(model_fn, num_estimators=n_components)
#     return ensemble

