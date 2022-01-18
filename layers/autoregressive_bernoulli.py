from typing import Union, Tuple, Callable, Optional

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from layers.base_models import DiscreteDistributionModel


class AutoRegressiveBernoulliNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            event_shape: Union[tf.TensorShape, Tuple[int, ...]] = None,
            activation: Callable[[Float], Float] = None,
            hidden_units: Tuple[int, ...] = None,
            conditional_input: Optional[tfkl.Input] = None,
            output_softclip: Optional[Callable[[Float], Float]] = tfb.Identity(),
            network_name: Optional[str] = None,
    ):

        conditional_event_shape = conditional_input.shape[1:] if conditional_input is not None else None
        input_event = tfk.Input(shape=event_shape, dtype=tf.float32)

        _made = tfb.AutoregressiveNetwork(
            params=1,
            hidden_units=hidden_units,
            event_shape=event_shape,
            conditional=conditional_input is not None,
            conditional_event_shape=conditional_event_shape,
            activation=activation,
            name=network_name)
        made = self._made(input_event, conditional_input=conditional_input)
        made = tfkl.Lambda(
            lambda x: output_softclip(x[..., 0])
        )(made)

        super(AutoRegressiveBernoulliNetwork, self).__init__(
            inputs=input_event if conditional_input is None else [input_event, conditional_input],
            outputs=made)
        self.event_shape = event_shape

    @staticmethod
    def _process_made_inputs(x: Float, conditional: Optional[Float] = None, *args, **kwargs):
        return [x, conditional] if conditional is not None else x

    def relaxed_distribution(
            self,
            temperature: Float,
            conditional_input: Optional[Float] = None
    ) -> tfd.Distribution:

        def distribution_fn(x: Float = None):
            if x is None:
                loc = tf.zeros(shape=self.event_shape, dtype=tf.float32)
            else:
                inputs = self._process_made_inputs(x, conditional_input)
                loc = self(inputs) / temperature
            return tfd.Independent(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(
                        loc=loc,
                        scale=1. / temperature),
                    bijector=tfb.Sigmoid()),
                reinterpreted_batch_ndims=1)

        return tfd.Autoregressive(distribution_fn)

    def discrete_distribution(
            self,
            conditional_input: Optional[Float] = None
    ) -> tfd.Distribution:

        def distribution_fn(x: Optional[Float] = None):
            if x is None:
                logits = tf.zeros(self.event_shape)
            else:
                inputs = self._process_made_inputs(x, conditional_input)
                logits = self(inputs)
            return tfd.Independent(
                distribution=tfd.Bernoulli(logits=logits),
                reinterpreted_batch_ndims=1)

        return tfd.Autoregressive(distribution_fn)

    def relaxed_invert_maf_distribution(
            self,
            temperature: Float,
            conditional_input: Optional[Float] = None,
            batch_size: Optional[int] = None,
    ) -> tfd.Distribution:
        if batch_size is None and conditional_input is not None:
            batch_size = tf.shape(conditional_input)[0]

        shape = (batch_size,) + self.event_shape if batch_size is not None else self.event_shape
        distribution = tfd.TransformedDistribution(
            distribution=tfd.Sample(
                tfd.Independent(
                    distribution=tfd.Logistic(
                        loc=tf.zeros(shape=shape, dtype=tf.float32),
                        scale=1. / temperature),
                    reinterpreted_batch_ndims=1)),
            bijector=tfb.Invert(
                tfb.MaskedAutoregressiveFlow(
                    lambda x: (self(self._process_made_inputs(x, conditional_input)) / temperature, None))))

        return tfd.TransformedDistribution(
            distribution=distribution,
            bijector=tfb.Sigmoid())

    def discrete_invert_maf_distribution(
            self,
            temperature: Float,
            conditional_input: Optional[Float] = None,
            batch_size: Optional[int] = None
    ) -> tfd.Distribution:
        return tfd.TransformedDistribution(
            distribution=self.relaxed_invert_maf_distribution(
                temperature, conditional_input, batch_size),
            bijector=tfb.Inline(
                forward_fn=tf.round,
                inverse_fn=lambda x: tf.clip_by_value(
                    x, clip_value_min=1e-7, clip_value_max=1. - 1e-7),
                forward_min_event_ndims=0))

    def get_config(self):
        config = super(AutoRegressiveBernoulliNetwork, self).get_config()
        config.update({
            "event_shape": self.event_shape,
            "_process_made_inputs": self._process_made_inputs,
            "relaxed_invert_maf_distribution": self.relaxed_invert_maf_distribution,
            "discrete_invert_maf_distribution": self.discrete_invert_maf_distribution})
        return config
