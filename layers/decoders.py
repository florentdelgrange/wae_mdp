from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from layers.base_models import DistributionModel


class StateReconstructionNetwork(DistributionModel):

    def __init__(
            self,
            next_latent_state: tfkl.Input,
            decoder_network: tfk.Model,
            state_shape: Union[Tuple[int, ...], tf.TensorShape],
            time_stacked_states: bool = False,
            state_decoder_pre_processing_network: Optional[tfk.Model] = None,
            time_stacked_lstm_units: Optional[int] = None,
    ):
        decoder = decoder_network(next_latent_state)
        if time_stacked_states:
            time_dimension = state_shape[0]
            _state_shape = state_shape[1:]

            if decoder.shape[-1] % time_dimension != 0:
                decoder = tfkl.Dense(
                    units=decoder.shape[-1] + time_dimension - decoder.shape[-1] % time_dimension
                )(decoder)

            decoder = tfkl.Reshape(
                target_shape=(time_dimension, decoder.shape[-1] // time_dimension)
            )(decoder)
            decoder = tfkl.LSTM(
                units=time_stacked_lstm_units, return_sequences=True
            )(decoder)

            if state_decoder_pre_processing_network is not None:
                decoder = tfkl.TimeDistributed(state_decoder_pre_processing_network)(decoder)

        else:
            if state_decoder_pre_processing_network is not None:
                decoder = state_decoder_pre_processing_network(decoder)
            _state_shape = state_shape

        decoder_output = tfk.models.Sequential([
            tfkl.Dense(
                units=np.prod(_state_shape),
                activation=None,
                name='state_decoder_raw_output'),
            tfkl.Reshape(
                target_shape=_state_shape,
                name='state_decoder_raw_output_reshape')],
            name="state_decoder")

        if time_stacked_states:
            decoder_output = tfkl.TimeDistributed(decoder_output)(decoder)
        else:
            decoder_output = decoder_output(decoder)

        super(StateReconstructionNetwork, self).__init__(
            inputs=next_latent_state,
            outputs=decoder_output,
            name='state_reconstruction_network')
        self.time_stacked_states = time_stacked_states

    def distribution(self, latent_state: Float) -> tfd.Distribution:
        if self.time_stacked_states:
            return tfd.Independent(tfd.Deterministic(loc=self(latent_state)))
        else:
            return tfd.Deterministic(loc=self(latent_state))

    def get_config(self):
        config = super(StateReconstructionNetwork, self).get_config()
        config.update({"time_stacked_states": self.time_stacked_states})
        return config


class ActionReconstructionNetwork(DistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input,
            latent_action: tfkl.Input,
            action_decoder_network: tfk.Model,
            action_shape: Union[Tuple[int, ...], tf.TensorShape],
    ):
        action_reconstruction_network = tfkl.Concatenate(name='action_reconstruction_input')([
            latent_state, latent_action])
        action_reconstruction_network = action_decoder_network(action_reconstruction_network)
        action_reconstruction_network = tfkl.Dense(
            units=np.prod(action_shape),
            activation=None,
            name='action_reconstruction_network_raw_output'
        )(action_reconstruction_network)
        action_reconstruction_network = tfkl.Reshape(
            target_shape=action_shape,
            name='action_reconstruction_network_output'
        )(action_reconstruction_network)

        super(ActionReconstructionNetwork, self).__init__(
            inputs=[latent_state, latent_action],
            outputs=action_reconstruction_network,
            name='action_reconstruction_network')

    def distribution(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tfd.Distribution:
        return tfd.Deterministic(loc=self([latent_state, latent_action]))


class RewardNetwork(DistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input = None,
            latent_action: tfkl.Input = None,
            next_latent_state: tfkl.Input = None,
            reward_network: tfk.Model = None,
            reward_shape: Union[Tuple[int, ...], tf.TensorShape] = None,
    ):
        _reward_network = tfkl.Concatenate(name='reward_function_input')(
            [latent_state, latent_action, next_latent_state])
        _reward_network = reward_network(_reward_network)
        _reward_network = tfkl.Dense(
            units=np.prod(reward_shape),
            activation=None,
            name='reward_network_raw_output'
        )(_reward_network)
        _reward_network = tfkl.Reshape(reward_shape, name='reward')(_reward_network)
        super(RewardNetwork, self).__init__(
            inputs=[latent_state, latent_action, next_latent_state],
            outputs=_reward_network,
            name='reward_network')

    def distribution(
            self,
            latent_state: Float,
            latent_action: Float,
            next_latent_state: Float,
    ) -> tfd.Distribution:
        return tfd.Deterministic(loc=self([latent_state, latent_action, next_latent_state]))
