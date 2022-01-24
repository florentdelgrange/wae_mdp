from typing import Union, Tuple, Callable, Optional

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.layers as tfpl
from tf_agents.typing.types import Float, Int

from layers.base_models import DiscreteDistributionModel


def relaxed_distribution(
        made: tfb.AutoregressiveNetwork,
        output_softclip: Callable[[Float], Float],
        temperature: Float,
) -> tfd.TransformedDistribution:
    event_shape = made._event_shape

    def bijector_fn(x, conditional_input: Optional[Float] = None) -> tfb.Bijector:
        shift = output_softclip(
            made(x, conditional_input=conditional_input)[..., 0]
        ) / temperature
        return tfb.Chain([tfb.Sigmoid(), tfb.Shift(shift)])

    maf = tfd.TransformedDistribution(
        distribution=tfd.Independent(
            tfd.Logistic(
                loc=tf.zeros(event_shape),
                scale=tf.pow(temperature, -1)),
            reinterpreted_batch_ndims=1),
        bijector=tfb.MaskedAutoregressiveFlow(
            bijector_fn=bijector_fn,
            is_constant_jacobian=True))
    maf._made_variables = made.variables
    return maf


def discrete_distribution(
        made: tfb.AutoregressiveNetwork,
        output_softclip: Callable[[Float], Float],
        conditional_input: Optional[Float] = None
) -> tfd.Autoregressive:
    event_shape = made._event_shape

    def distribution_fn(x: Optional[Float] = None):
        if x is None:
            # enforce distribution0 to forward samples with zero values as input of MADE
            logits = -10. * tf.ones(event_shape)
        else:
            logits = output_softclip(
                made(x, conditional_input=conditional_input))

        return tfd.Independent(
            distribution=tfd.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=1)

    return tfd.Autoregressive(distribution_fn)

class AutoregressiveTransform(tfpl.DistributionLambda):
    def __init__(
            self,
            made: tfb.AutoregressiveNetwork,
            temperature: Optional[Float] = None,
            output_softclip: Optional[Callable[[Float], Float]] = None,
            **kwargs
    ):
        if temperature is None:
            temperature = 1e-5
        if output_softclip is None:
            output_softclip = tfb.Identity()

        super(AutoregressiveTransform, self).__init__(self._transform, **kwargs)
        self._made = made
        self._temperature = temperature
        self._output_softclip = output_softclip

    def build(self, input_shape):
        if self._made._conditional:
            inputs = tfk.Input(input_shape[0][1:], dtype=self.dtype)
            conditional_input = tfk.Input(input_shape[1][1:], dtype=self.dtype)
            outputs = self._made(inputs, conditional_input=conditional_input)
            outputs = tfkl.Lambda(lambda x: self._output_softclip(x) / self._temperature)(outputs)
            tfk.Model(inputs=[inputs, conditional_input], outputs=outputs)
        else:
            tfk.Sequential([
                tfkl.InputLayer(
                    input_shape=input_shape[1:], dtype=self.dtype),
                self._made,
                tfkl.Lambda(lambda x: self._output_softclip(x) / self._temperature)
            ])
        super(AutoregressiveTransform, self).build(input_shape)

    def _transform(self, previous_outputs):
        if self._made._conditional:
            distribution, conditional_input = previous_outputs
        else:
            distribution, conditional_input = previous_outputs, None

        # print("distribution", distribution)
        # print("reparameterizable?", distribution.reparameterization_type)

        def bijector_fn(x) -> tfb.Bijector:
            shift = self._output_softclip(
                self._made(x, conditional_input=conditional_input)[..., 0]
            ) / self._temperature
            return tfb.Chain([tfb.Sigmoid(), tfb.Shift(shift)])

        return tfd.TransformedDistribution(
            bijector=tfb.MaskedAutoregressiveFlow(
                bijector_fn=bijector_fn,
                is_constant_jacobian=True, ),
            distribution=distribution, )


class AutoRegressiveBernoulliNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            event_shape: Union[tf.TensorShape, Tuple[int, ...]],
            activation: Callable[[Float], Float],
            hidden_units: Tuple[int, ...],
            conditional_event_shape: Optional[Union[tf.TensorShape, Tuple[int, ...]]] = None,
            temperature: Optional[Float] = 1e-5,
            output_softclip: Optional[Callable[[Float], Float]] = tfb.Identity(),
            dtype: tf.dtypes = tf.float32,
            name: Optional[str] = None,
            made_name: Optional[str] = None,
            input_event_name: str = 'input_event',
            conditional_input_name: str = 'conditional_input',
    ):
        if conditional_event_shape is None:
            inputs = tfk.Input(shape=(0,), dtype=dtype, name=input_event_name)
            conditional_input = None
        else:
            inputs = tfk.Input(shape=conditional_event_shape, dtype=dtype, name=conditional_input_name)
            conditional_input = inputs

        logistic_distribution_layer = tfk.Sequential([
            tfkl.InputLayer(input_shape=conditional_event_shape, dtype=dtype, name="logistic_layer_input"),
            tfpl.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Logistic(
                        loc=tf.zeros(tf.concat([tf.shape(t)[:-1], event_shape], axis=0)),
                        scale=tf.pow(temperature, -1), ),
                    reinterpreted_batch_ndims=1, )),
        ], name="sequential_logistic_distribution_layer")

        made = tfb.AutoregressiveNetwork(
            params=1,
            hidden_units=hidden_units,
            event_shape=None if conditional_input is None else event_shape,
            conditional=conditional_input is not None,
            conditional_event_shape=conditional_event_shape,
            activation=activation,
            name=made_name)

        x = logistic_distribution_layer(inputs)
        x = AutoregressiveTransform(
            made=made,
            temperature=temperature,
            output_softclip=output_softclip,
            name="autoregressive_transform",
        )(x if conditional_input is None else [x, conditional_input])

        super(AutoRegressiveBernoulliNetwork, self).__init__(
            inputs=inputs,
            outputs=x,
            name=name)

        self.event_shape = event_shape
        self._temperature = temperature
        self._output_softclip = output_softclip
        self._logistic_distribution_network = logistic_distribution_layer
        self._made = made

    def relaxed_distribution(
            self,
            temperature: Float,
            conditional_input: Optional[Float] = None,
    ) -> tfd.Distribution:
        """
        Construct a distribution whose parameters are produced by a Masked Autoregressive Flow.
        More specifically, the Flow uses the internal masked autoregressive network to infer a location of a logistic
        distribution at each event step. This allows (via a chain of reparameterization) to generate logistic samples
        followed by a sigmoid at each time step, in order to generate (dependent) samples of relaxed Bernoulli.
        """
        return MaskedAutoregressiveFlowDistributionWrapper(
            relaxed_distribution(self._made, self._output_softclip, self._temperature),
            conditional=conditional_input)

    def discrete_distribution(
            self,
            conditional_input: Optional[Float] = None
    ) -> tfd.Distribution:
        """
        Important: to sample from this distribution when a conditional input is provided, the batch size of the
        conditional need to be provided in parameter of the tfd.Distribution.sample() function:
        ```python
        event_shape = (3, )
        cond_shape = (5, )
        batch_size = 4

        autoregressive_model = AutoRegressiveBernoulliNetwork(
            ..., event_shape=event_shape, conditional_event_shape=cond_shape)
        conditional_samples = tf.random.uniform((batch_size, ) + cond_shape)
        autoregressive_model.relaxed_distribution(
            temperature=.5,
            conditional_input=conditional_samples
        ).sample()  # no need to provide batch_size here

        autoregressive_model.discrete_distribution(
            conditional_input=conditional_samples
        ).sample(batch_size)  # here, batch_size need to be provided;
        # note that batch_size must always match tf.shape(conditional_samples)[0]
        ```
        """
        return discrete_distribution(self._made, self._output_softclip, conditional_input)

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


class MaskedAutoregressiveFlowDistributionWrapper(tfd.Distribution):

    def __init__(
            self,
            masked_autoregressive_flow_transformed_distribution: tfd.TransformedDistribution,
            conditional: Float
    ):
        super().__init__(
            masked_autoregressive_flow_transformed_distribution.dtype,
            masked_autoregressive_flow_transformed_distribution.reparameterization_type,
            masked_autoregressive_flow_transformed_distribution.validate_args,
            masked_autoregressive_flow_transformed_distribution.allow_nan_stats)
        self._wrapped_distribution: tfd.TransformedDistribution = masked_autoregressive_flow_transformed_distribution
        self._conditional = conditional

    def _batch_shape_tensor(self):
        return self._wrapped_distribution._batch_shape_tensor()

    def _event_shape_tensor(self):
        return self._wrapped_distribution._event_shape_tensor()

    def _sample_n(self, n, seed=None, **kwargs):
        return self._wrapped_distribution.sample(
            n, seed=seed, bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _log_survival_function(self, value, **kwargs):
        return self._wrapped_distribution._log_survival_function(
            value, bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _survival_function(self, value, **kwargs):
        return self._survival_function(
            value, bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _entropy(self, **kwargs):
        return self._wrapped_distribution._entropy(bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _mean(self, **kwargs):
        self._wrapped_distribution._mean(bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _quantile(self, value, **kwargs):
        return self._wrapped_distribution._quantile(
            value, bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _variance(self, **kwargs):
        return self._wrapped_distribution._variance(bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _stddev(self, **kwargs):
        return self._wrapped_distribution._stddev(bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _covariance(self, **kwargs):
        return self._wrapped_distribution._covariance(
            bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _mode(self, **kwargs):
        return self._wrapped_distribution._mode(bijector_kwargs={'conditional_input': self._conditional}, **kwargs)

    def _default_event_space_bijector(self, *args, **kwargs):
        return self._wrapped_distribution._default_event_space_bijector(
            bijector_kwargs={'conditional_input': self._conditional}, *args, **kwargs)
