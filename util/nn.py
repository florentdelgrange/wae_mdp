from typing import NamedTuple, Tuple, Callable, Optional

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python import bijectors as tfb


class ModelArchitecture(NamedTuple):
    hidden_units: Tuple[int, ...]
    activation: str
    name: Optional[str] = None


def generate_sequential_model(architecture: ModelArchitecture):
    return tfk.Sequential([
        tfkl.Dense(
            units,
            activation=get_activation_fn(architecture.activation),
            name="{}_layer{:d}".format(architecture.name, i) if architecture.name is not None else None,
        ) for i, units in enumerate(architecture.hidden_units)],
        name=architecture.name)


def get_activation_fn(activation: str):
    if hasattr(tf.nn, activation):
        return getattr(tf.nn, activation)
    elif hasattr(tfb, activation):
        return getattr(tfb, activation)()
    else:
        # custom activations
        other_activations = {
            'smooth_elu': lambda x: tf.nn.softplus(2. * x + 2.) / 2. - 1.,
            'SmoothELU': tfb.Chain([tfb.Shift(-1.), tfb.Scale(.5), tfb.Softplus(), tfb.Shift(2.), tfb.Scale(2.)])
        }
        return other_activations.get(
            activation,
            ValueError("activation {} unknown".format(activation)))


def scan_model(model: tfk.Model):
    hidden_units = []
    activation = None
    if model is None:
        return [128, 128], tf.nn.relu
    for layer in model.layers:
        if hasattr(layer, 'units'):
            hidden_units.append(layer.units)
        if hasattr(layer, 'activation') and activation != layer.activation:
            activation = layer.activation
    return hidden_units, activation
