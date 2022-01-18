from typing import Optional, Callable

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from layers.base_models import DiscreteDistributionModel


class StateEncoderNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            state: tfkl.Input,
            state_encoder_network: tfk.Model,
            latent_state_size: int,
            atomic_props_dims: int,
            time_stacked_states: bool,
            output_softclip: Callable[[Float], Float] = tfb.Identity(),
            state_encoder_pre_processing_network: Optional[tfk.Model] = None,
    ):
        if time_stacked_states:
            if state_encoder_pre_processing_network is not None:
                encoder = tfkl.TimeDistributed(state_encoder_pre_processing_network)(state)
            else:
                encoder = state
            encoder = tfkl.LSTM(units=self.time_stacked_lstm_units)(encoder)
            encoder = state_encoder_network(encoder)
        else:
            if state_encoder_pre_processing_network is not None:
                _state = state_encoder_pre_processing_network(state)
            else:
                _state = state
            encoder = state_encoder_network(_state)
        logits_layer = tfkl.Dense(
            units=latent_state_size - atomic_props_dims,
            # allows avoiding exploding logits values and probability errors after applying a sigmoid
            activation=output_softclip,
            name='encoder_latent_distribution_logits'
        )(encoder)

        super(StateEncoderNetwork, self).__init__(
            inputs=state,
            outputs=logits_layer,
            name='state_encoder')

    def relaxed_distribution(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            logistic: bool = True,
    ) -> tfd.Distribution:

        logits = self(state)
        if label is not None:
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        if logistic:
            return tfd.TransformedDistribution(
                distribution=tfd.Logistic(
                    loc=logits / temperature,
                    scale=1. / temperature,
                    allow_nan_stats=False),
                bijector=tfb.Sigmoid())
        else:
            return tfd.Independent(
                tfd.RelaxedBernoulli(
                    logits=logits,
                    temperature=temperature,
                    allow_nan_stats=False))

    def discrete_distribution(
            self,
            state: Float,
            label: Optional[Float] = None,
    ) -> tfd.Distribution:
        logits = self(state)
        if label is not None:
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        return tfd.Independent(
            tfd.Bernoulli(
                logits=logits,
                allow_nan_stats=False))


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

        self.epsilon = epsilon
        self.relaxed_exp_one_hot_action_encoding = relaxed_exp_one_hot_action_encoding
        self.epsilon = tf.Variable(epsilon, trainable=False)
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
        logits = self.action_encoder_network([latent_state, action])
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
