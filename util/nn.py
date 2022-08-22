from typing import NamedTuple, Tuple, Callable, Optional, Union, List

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python import bijectors as tfb


class ModelArchitecture(NamedTuple):
    hidden_units: Optional[Tuple[int, ...]] = None
    activation: Optional[str] = None
    output_dim: Optional[Tuple[int, ...]] = None
    input_dim: Optional[Tuple[int, ...]] = None
    name: Optional[str] = None
    batch_norm: bool = False
    filters: Optional[Tuple[int, ...]] = None
    kernel_size: Optional[Union[Tuple[int, ...], int]] = None
    strides: Optional[Union[Tuple[int, ...], int]] = None
    padding: Union[Tuple[str, ...], str] = None
    raw_last: bool = True
    transpose: bool = False

    @property
    def is_cnn(self):
        return self.filters is not None

    def invert(self, input_dim: Optional[Tuple[int, ...]]):
        assert not self.transpose
        model_arch = self._asdict()
        if self.name is not None:
            model_arch['name'] = 'inv_' + self.name
        if input_dim is None:
            input_dim = self.input_dim
        if self.is_cnn:
            model_arch['filters'] = tuple(reversed(self.filters[:-1])) + (input_dim[-1],)
            model_arch['kernel_size'] = tuple(reversed(self.kernel_size))
            model_arch['strides'] = tuple(reversed(self.strides))
            model_arch['padding'] = tuple(reversed(self.padding))
        else:
            model_arch["hidden_units"] = tuple(reversed(self.hidden_units))
        model_arch['output_dim'] = self.input_dim
        model_arch['input_dim'] = self.output_dim
        model_arch['transpose'] = True
        model_arch = ModelArchitecture(**model_arch)
        return model_arch

    def short_dict(self):
        return {k: v for k, v in self._asdict().items() if v is not None}


def generate_sequential_model(architecture: ModelArchitecture):
    return tfk.Sequential([
        tfkl.Dense(
            units,
            activation=get_activation_fn(architecture.activation),
            name="{}_layer{:d}".format(architecture.name, i) if architecture.name is not None else None,
        ) for i, units in enumerate(architecture.hidden_units)],
        name=architecture.name)


def get_model(
        model_arch: ModelArchitecture,
        invert: bool = False,
        output_dim: Optional[Tuple[int, ...]] = None,
        input_dim: Optional[Tuple[int, ...]] = None,
        as_model: bool = False,
):
    if model_arch.is_cnn:
        model_arch = model_arch._replace(hidden_units=None)
    if as_model:
        if invert:
            if model_arch.output_dim is not None:
                assert output_dim is None
                output_dim = model_arch.output_dim
            if output_dim is None:
                # dirty output dim inference
                _net = get_model(model_arch, as_model=True)
                output_dim = _net.outputs[0].shape[1:]
                del _net
            input_ = tfk.Input(output_dim)
            model_arch = model_arch.invert(output_dim)
        else:
            if model_arch.input_dim is not None:
                assert input_dim is None
                input_dim = model_arch.input_dim
            assert input_dim is not None, "input_dim should be provided"
            input_ = tfk.Input(input_dim)
        if model_arch.is_cnn:
            output = _conv_network(input_=input_, **model_arch.short_dict())
        else:
            output = _fc_network(input_=input_, **model_arch.short_dict())
        model = tfk.Model(inputs=input_, outputs=output)
        return model
    if model_arch.is_cnn:
        if invert:
            layer = Deconvolutional(model_arch=model_arch, output_shape=output_dim)
        else:
            layer = Convolutional(model_arch)
    else:
        if invert:
            layer = TransposeFullyConnected(model_arch=model_arch, output_shape=output_dim)
        else:
            layer = FullyConnected(model_arch=model_arch)
    return layer


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


def _pass_in_layers(layers: List[tfkl.Layer], input_):
    output = input_
    for layer in layers:
        output = layer(output)
    return output


def _fc_network_layers(hidden_units: Tuple[int, ...],
                       # F: general activation functions can now be provided by hand
                       # the set of available activation function is now larger (see get_activation_fn)
                       activation: Union[Callable, str],
                       output_dim: Optional[Tuple[int, ...]],
                       batch_norm: bool,
                       raw_last: bool,
                       **kwargs,
                       ):
    layers = [tfkl.Flatten()]
    assert output_dim is not None, "output_dim should be provided"
    units = tuple(list(hidden_units) + [np.prod(output_dim)])
    for i, unit in enumerate(units):
        apply_ = not raw_last and (i + 1 == len(units))
        layers.append(tfkl.Dense(unit))
        if apply_ and batch_norm:
            layers.append(tfkl.BatchNormalization())
        if apply_ and activation:
            if callable(activation):
                layers.append(tfkl.Activation(activation))
            else:
                layers.append(tfkl.Activation(get_activation_fn(activation)))
    if output_dim is not None:
        layers.append(tfkl.Reshape(output_dim))
    return layers


def _fc_network(input_: tfkl.Input,
                hidden_units: Tuple[int, ...],
                activation: str,
                output_dim: Optional[Tuple[int, ...]],
                batch_norm: bool,
                raw_last: bool,
                **kwargs,
                ):
    layers = _fc_network_layers(hidden_units, activation, output_dim, batch_norm, raw_last)
    return _pass_in_layers(layers, input_)


class FullyConnected(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            **kwargs
    ):
        self.model_arch = model_arch
        d = model_arch.short_dict()
        name = d.pop('name', None)
        super().__init__(name=name, **kwargs)
        self._layers = _fc_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


class TransposeFullyConnected(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            output_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ):
        self.model_arch = model_arch
        self.invert_model_arch = model_arch.invert(output_shape)
        d = self.invert_model_arch.short_dict()
        name = d.pop('name')
        super().__init__(name=name, **kwargs)
        self._layers: List[tfkl.Layer] = _fc_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


def _cnn_network_layers(
        filters: Tuple[int, ...],
        kernel_size: Tuple[Union[Tuple[int, ...], int], ...],
        strides: Tuple[Union[Tuple[int, ...], int], ...],
        padding: Tuple[str, ...],
        activation: Union[Callable, str],
        batch_norm: bool,
        raw_last: bool,
        transpose: bool = False,
        **kwargs,
):
    layers = []
    tf_layer = tfkl.Conv2D if not transpose else tfkl.Conv2DTranspose
    elements = [filters, kernel_size, strides, padding]
    # check that the number of elements is the same for all components
    n = len(elements[0])
    for element in elements[1:]:
        assert len(element) == n, "the number of filters, kernel_size, strides, padding should be the same"
    for i, (filters_, kernel_size_, stride_, padding_) in enumerate(zip(*elements)):
        apply_ = (i + 1 != len(filters)) or not raw_last
        layers.append(tf_layer(
            filters=filters_,
            kernel_size=kernel_size_,
            strides=stride_,
            padding=padding_,
        ))
        if apply_ and batch_norm:
            layers.append(tfkl.BatchNormalization())
        if apply_ and activation:
            if callable(activation):
                layers.append(tfkl.Activation(activation))
            else:
                layers.append(tfkl.Activation(get_activation_fn(activation)))
    return layers


def _conv_network(input_: tfkl.Input,
                  filters: Tuple[int, ...],
                  kernel_size: Tuple[Union[Tuple[int, ...], int], ...],
                  strides: Tuple[Union[Tuple[int, ...], int], ...],
                  padding: Tuple[str, ...],
                  activation: str,
                  batch_norm: bool,
                  raw_last: bool,
                  transpose: bool = False,
                  **kwargs,
                  ):
    layers = _cnn_network_layers(filters, kernel_size, strides, padding, activation, batch_norm, raw_last, transpose)
    return _pass_in_layers(layers, input_)


class Convolutional(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            **kwargs,
    ):
        self.model_arch = model_arch
        d = model_arch.short_dict()
        name = d.pop('name', None)
        super().__init__(name=name, **kwargs)
        # kernel_size, strides, padding = [
        #     tf.nest.flatten((param if param is not None else default))
        #     for param, default in zip((kernel_size, strides, padding), (3, 1, 'valid'))
        # ]
        self._layers: List[tfkl.Layer] = _cnn_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


class Deconvolutional(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            output_shape: Optional[Tuple[int, ...]] = None,
            **kwargs,
    ):
        self.model_arch = model_arch
        self.invert_model_arch = model_arch.invert(output_shape)
        d = self.invert_model_arch.short_dict()
        name = d.pop('name')
        super().__init__(name=name, **kwargs)
        self._layers: List[tfkl.Layer] = _cnn_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)
