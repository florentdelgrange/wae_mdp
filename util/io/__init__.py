import tensorflow as tf
import tensorflow.keras as tfk


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
