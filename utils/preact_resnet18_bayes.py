from utils.preact_resnet18 import two_inputs_preact_resnet18

from keras.layers import Conv2D
from keras_uncertainty.layers import DropConnectConv2D, VariationalConv2D, FlipoutConv2D, StochasticDropout

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
    return two_inputs_preact_resnet18(input_shape, lambda filters, kernel_size, strides, padding : DropoutConv2D(filters, strides=strides, kernel_size=kernel_size, padding=padding, p_drop=p_drop), num_classes)

def get_dropconnect_preact_resnet18(num_classes:int=10):
    pass