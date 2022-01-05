import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
import numpy as np


class InputDependentWeightLayer(Layer):
    """
    Custom layer where the weights are functions of the input.

    Given a cell represented by the function O = I * W + B,

    W is a function of I: W = I * w + b

    """
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.input_length = input_shape[-1]

        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[-1] * self.units),
            initializer='GlorotUniform',
            trainable=True
        )
        self.b_inner = self.add_weight(
            shape=(1, input_shape[-1] * self.units),
            initializer='GlorotUniform',
            trainable=True
        )
        self.b_outer = self.add_weight(
            shape=(1, self.units),
            initializer='GlorotUniform',
            trainable=True
        )

    def call(self, inputs, **kwargs):
        inner_weights = tf.matmul(inputs, self.w) + self.b_inner

        inner_weights = tf.transpose(tf.reshape(inner_weights, (self.units, self.input_length)))

        return tf.matmul(inputs, inner_weights) + self.b_outer
