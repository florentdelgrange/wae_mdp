from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from layers.base_models import DistributionModel
from util.nn import _get_elem


class StateReconstructionNetwork(DistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input,
            decoder_network: tfk.Model,
            state_shape: Union[Tuple[int, ...], tf.TensorShape, Tuple[Tuple[int, ...]], Tuple[tf.TensorShape, ...]],
            time_stacked_states: bool = False,
            post_processing_net: Optional[Tuple[Union[tfk.Model, tfkl.Layer], ...]] = None,
            time_stacked_lstm_units: Optional[int] = None,
            flatten_output: bool = False
    ):

        decoder = decoder_network(latent_state)

        try:
            # output with multiple components
            self.n_dim = len(state_shape[0])
            if flatten_output:
                # enforce flattening the output
                state_shape = [np.sum([np.prod(shape_i) for shape_i in state_shape])]
                self.n_dim = 1
        except TypeError:
            self.n_dim = 1
            state_shape = [state_shape]

        post_processing_net = tf.nest.flatten(post_processing_net)

        outputs = []
        for i, _state_shape in enumerate(state_shape):
            _decoder = _get_elem(post_processing_net, i)
            if _decoder.inputs is None:
                x = latent_state
            else:
                post_process_is = _decoder.inputs[0].shape[1:]
                x = tfkl.Dense(
                    units=np.prod(post_process_is),
                    activation=tf.nn.sigmoid,
                )(latent_state)
                x = tfkl.Reshape(target_shape=post_process_is)(x)
            _decoder = _decoder(x)

            if time_stacked_states and time_stacked_lstm_units is not None:
                decoder_output = tfkl.Flatten(_decoder)
                time_dimension = _state_shape[0]
                _state_shape = _state_shape[1:]

                if decoder_output.shape[-1] % time_dimension != 0:
                    decoder_output = tfkl.Dense(
                        units=decoder_output.shape[-1] + time_dimension - decoder_output.shape[-1] % time_dimension
                    )(decoder_output)

                decoder_output = tfkl.Reshape(
                    target_shape=(time_dimension, decoder_output.shape[-1] // time_dimension)
                )(decoder_output)
                decoder_output = tfkl.LSTM(
                    units=time_stacked_lstm_units, return_sequences=True
                )(decoder_output)
            else:
                decoder_output = _decoder

            if np.prod(decoder_output.shape[1:]) % np.prod(_state_shape) != 0:
                if len(decoder_output.shape[1:]) > 1:
                    decoder_output = tfkl.Flatten()(decoder_output)
                decoder_output = tfkl.Dense(
                    units=np.prod(_state_shape),
                    activation=None,
                    name='state_decoder_raw_output_{:d}'.format(i)
                )(decoder_output)
            if not flatten_output and decoder_output.shape[1:] != _state_shape:
                decoder_output = tfkl.Reshape(
                    target_shape=_state_shape,
                    name='state_{:d}_decoder_raw_output_reshape'.format(i))

            if time_stacked_states:
                decoder_output = tfkl.TimeDistributed(decoder_output)(_decoder)
            else:
                decoder_output = decoder_output(_decoder)

            outputs.append(decoder_output)

        super(StateReconstructionNetwork, self).__init__(
            inputs=latent_state,
            outputs=outputs,
            name='state_reconstruction_network')
        self.time_stacked_states = time_stacked_states

    def distribution(self, latent_state: Float) -> tfd.Distribution:
        outputs = self(latent_state)
        distributions = []
        for output in outputs:
            if self.time_stacked_states:
                distributions.append(tfd.Independent(tfd.Deterministic(loc=output)))
            else:
                distributions.append(tfd.Deterministic(loc=output))
        if self.n_dim == 1:
            return distributions[0]
        else:
            return tfd.JointDistributionSequential(distributions)

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
