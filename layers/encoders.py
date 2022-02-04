from typing import Optional, Callable, Union, Tuple

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from layers.autoregressive_bernoulli import AutoRegressiveBernoulliNetwork
from layers.base_models import DiscreteDistributionModel
from util.io import scan_model


class StateEncoderNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            state: tfkl.Input,
            state_encoder_network: tfk.Model,
            latent_state_size: int,
            atomic_props_dims: int,
            time_stacked_states: bool = False,
            time_stacked_lstm_units: int = 128,
            output_softclip: Callable[[Float], Float] = tfb.Identity(),
            state_encoder_pre_processing_network: Optional[tfk.Model] = None,
            lstm_output: bool = False,
    ):
        hidden_units, activation = scan_model(state_encoder_network)
        if time_stacked_states:
            if state_encoder_pre_processing_network is not None:
                encoder = tfkl.TimeDistributed(state_encoder_pre_processing_network)(state)
            else:
                encoder = state
            encoder = tfkl.LSTM(units=time_stacked_lstm_units)(encoder)
            encoder = state_encoder_network(encoder)
        else:
            if state_encoder_pre_processing_network is not None:
                _state = state_encoder_pre_processing_network(state)
            else:
                _state = state
            encoder = state_encoder_network(_state)

        if lstm_output:
            units = (latent_state_size - atomic_props_dims)
            encoder = tfkl.Dense(
                units=hidden_units[-1] // units * units,
                activation=activation
            )(encoder)
            encoder = tfkl.Reshape(target_shape=(units, hidden_units[-1] // units))(encoder)
            encoder = tfkl.LSTM(units=1, activation=output_softclip, return_sequences=True)(encoder)
            encoder = tfkl.Reshape(target_shape=(latent_state_size - atomic_props_dims,))(encoder)
        else:
            encoder = tfkl.Dense(
                units=latent_state_size - atomic_props_dims,
                activation=output_softclip
                # allows avoiding exploding logits values and probability errors after applying a sigmoid
            )(encoder)

        super(StateEncoderNetwork, self).__init__(
            inputs=state,
            outputs=encoder,
            name='state_encoder')

    def relaxed_distribution(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            logistic: bool = True,
    ) -> tfd.Distribution:

        logits = self(state)
        if logistic:
            distribution = tfd.TransformedDistribution(
                distribution=tfd.Independent(
                    tfd.Logistic(
                        loc=logits / temperature,
                        scale=tf.pow(temperature, -1.)),
                    reinterpreted_batch_ndims=1,),
                bijector=tfb.Sigmoid())
        else:
            distribution = tfd.Independent(
                tfd.RelaxedBernoulli(
                    logits=logits,
                    temperature=temperature,
                    allow_nan_stats=False),
                reinterpreted_batch_ndims=1)
        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=label),
                reinterpreted_batch_ndims=1)
            return tfd.Blockwise([d1, distribution])
        else:
            return distribution

    def discrete_distribution(
            self,
            state: Float,
            label: Optional[Float] = None,
    ) -> tfd.Distribution:
        logits = self(state)
        d2 = tfd.Independent(
            tfd.Bernoulli(
                logits=logits,
                dtype=self.dtype),
            reinterpreted_batch_ndims=1)

        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=tf.cast(label, dtype=self.dtype)),
                reinterpreted_batch_ndims=1)

            def mode(name='mode', **kwargs):
                return tf.concat([
                    d1.mode(name='label_' + name, **kwargs),
                    d2.mode(name='latent_state_' + name, **kwargs)],
                    axis=-1)

            distribution = tfd.Blockwise([d1, d2])
            distribution.mode = mode
            return distribution
        else:
            return d2

    def get_logits(self, state: Float, *args, **kwargs):
        return self(state)

    def get_config(self):
        config = super(AutoRegressiveStateEncoderNetwork, self).get_config()
        config.update({
            "get_logits": self.get_logits,
        })
        return config


class AutoRegressiveStateEncoderNetwork(AutoRegressiveBernoulliNetwork):
    def __init__(
            self,
            state_shape: Union[tf.TensorShape, Tuple[int, ...]],
            activation: Union[str, Callable[[Float], Float]],
            hidden_units: Tuple[int, ...],
            latent_state_size: int,
            atomic_props_dims: int,
            temperature: Float,
            time_stacked_states: bool = False,
            time_stacked_lstm_units: int = 128,
            output_softclip: Callable[[Float], Float] = tfb.Identity(),
            state_encoder_pre_processing_network: Optional[tfk.Model] = None,
    ):
        super(AutoRegressiveStateEncoderNetwork, self).__init__(
            event_shape=(latent_state_size - atomic_props_dims,),
            activation=activation,
            hidden_units=hidden_units,
            conditional_event_shape=state_shape,
            temperature=temperature,
            output_softclip=output_softclip,
            time_stacked_input=time_stacked_states,
            time_stacked_lstm_units=time_stacked_lstm_units,
            pre_processing_network=state_encoder_pre_processing_network,
            name='autoregressive_state_encoder')
        self._atomic_props_dims = atomic_props_dims

    def relaxed_distribution(
            self,
            state: Optional[Float] = None,
            label: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        if state is None:
            raise ValueError("a state to encode should be provided.")

        distribution = super(
            AutoRegressiveStateEncoderNetwork, self
        ).relaxed_distribution(conditional_input=state)

        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=label),
                reinterpreted_batch_ndims=1)
            return tfd.Blockwise([d1, distribution])
        else:
            return distribution

    def discrete_distribution(
            self,
            state: Optional[Float] = None,
            label: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        if state is None:
            raise ValueError("a state to encode should be provided.")

        d2 = super(
            AutoRegressiveStateEncoderNetwork, self
        ).discrete_distribution(conditional_input=state)

        def mode(name='mode', **kwargs):
            def d2_distribution_fn_mode(x: Optional[Float] = None):
                d = d2.distribution_fn(x)

                def call_mode_n(*args, **kwargs):
                    mode = d.mode(**kwargs)
                    return mode

                d._call_sample_n = call_mode_n
                return d

            return tfd.Autoregressive(
                distribution_fn=d2_distribution_fn_mode,
            ).sample(sample_shape=tf.shape(state)[:-1], name=name, **kwargs)

        d2.mode = mode

        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=tf.cast(label, dtype=self.dtype)),
                reinterpreted_batch_ndims=1)

            def mode(name='mode', **kwargs):
                return tf.concat([
                    d1.mode(name='label_' + name, **kwargs),
                    d2.mode(name='latent_state_' + name, **kwargs)],
                    axis=-1)

            def sample(sample_shape=(), seed=None, name='sample', **kwargs):
                return tf.concat([
                    d1.sample(sample_shape, seed=seed, name='label_' + name, **kwargs),
                    d2.sample(sample_shape, seed=seed, name='latent_state_' + name, **kwargs)],
                    axis=-1)

            # dirty Blockwise; do not trigger any warning
            distribution = tfd.TransformedDistribution(d1, bijector=tfb.Identity())
            distribution.mode = mode
            distribution.sample = sample
            return distribution
        else:
            return d2

    def get_logits(
            self,
            state: Float,
            latent_state: Float,
            include_label: bool = True,
            *args, **kwargs
    ) -> Float:
        if include_label:
            latent_state = latent_state[..., self._atomic_props_dims:]
        if self.pre_process_input:
            state = self._preprocess_fn(state)
        return self._output_softclip(self._made(latent_state, conditional_input=state)[..., 0])

    def get_config(self):
        config = super(AutoRegressiveStateEncoderNetwork, self).get_config()
        config.update({
            "_atomic_props_dims": self._atomic_props_dims,
            "get_logits": self.get_logits,
        })
        return config


class ActionEncoderNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            latent_state: tfk.Input,
            action: tfk.Input,
            number_of_discrete_actions: int,
            action_encoder_network: tfk.Model,
            relaxed_exp_one_hot_action_encoding: bool = True,
            epsilon: Float = 1e-12,
    ):

        action_encoder = tfkl.Concatenate(name='action_encoder_input')(
            [latent_state, action])
        action_encoder = action_encoder_network(action_encoder)
        action_encoder = tfkl.Dense(
            units=number_of_discrete_actions,
            activation=None,
            name='action_encoder_exp_one_hot_logits'
        )(action_encoder)

        super(ActionEncoderNetwork, self).__init__(
            inputs=[latent_state, action],
            outputs=action_encoder,
            name="action_encoder")
        self.relaxed_exp_one_hot_action_encoding = relaxed_exp_one_hot_action_encoding
        self.epsilon = tf.Variable(epsilon, trainable=False)

    def relaxed_distribution(
            self,
            latent_state: Float,
            action: Float,
            temperature: Float,
    ) -> tfd.Distribution:
        logits = self([latent_state, action])
        if self.relaxed_exp_one_hot_action_encoding:
            return tfd.TransformedDistribution(
                distribution=tfd.ExpRelaxedOneHotCategorical(
                    temperature=temperature,
                    logits=logits,
                    allow_nan_stats=False),
                bijector=tfb.Exp())
        else:
            return tfd.RelaxedOneHotCategorical(
                logits=logits,
                temperature=temperature,
                allow_nan_stats=False)

    def discrete_distribution(
            self,
            latent_state: Float,
            action: Float,
    ) -> tfd.Distribution:
        logits = self([latent_state, action])
        if self.relaxed_exp_one_hot_action_encoding:
            relaxed_distribution = tfd.ExpRelaxedOneHotCategorical(
                temperature=1e-5,
                logits=logits,
                allow_nan_stats=False)
            log_probs = tf.math.log(relaxed_distribution.probs_parameter() + self.epsilon)
            return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False)
        else:
            return tfd.OneHotCategorical(logits=logits, allow_nan_stats=False)

    def get_config(self):
        config = super(ActionEncoderNetwork, self).get_config()
        config.update({"relaxed_exp_one_hot_action_encoding": self.relaxed_exp_one_hot_action_encoding})
        return config
