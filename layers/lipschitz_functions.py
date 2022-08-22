from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl

class SteadyStateLipschitzFunction(tfk.Model):

    def __init__(
            self,
            latent_state: tfk.Input,
            next_latent_state: tfk.Input,
            steady_state_lipschitz_network: tfk.Model,
            latent_action: Optional[tfkl.Input] = None,
    ):
        inputs = [latent_state] + ([latent_action] if latent_action is not None else []) + [next_latent_state]
        network_input = tfkl.Concatenate()(inputs)
        _steady_state_lipschitz_network = steady_state_lipschitz_network(network_input)
        _steady_state_lipschitz_network = tfkl.Dense(
            units=1,
            activation=None,
            name='steady_state_lipschitz_network_output'
        )(_steady_state_lipschitz_network)

        super(SteadyStateLipschitzFunction, self).__init__(
            inputs=inputs,
            outputs=_steady_state_lipschitz_network,
            name='steady_state_lipschitz_network')


class TransitionLossLipschitzFunction(tfk.Model):
    def __init__(
            self,
            state: Union[tfkl.Input, Tuple[tfkl.Input, ...]],
            action: tfkl.Input,
            latent_state: tfkl.Input,
            next_latent_state: tfkl.Input,
            transition_loss_lipschitz_network: tfk.Model,
            latent_action: Optional[tfkl.Input] = None,
            flatten_units: int = 64,
    ):
        try:
            no_inputs = len(state)
        except TypeError:
            no_inputs = 1

        if no_inputs > 1:
            components = []
            for state_component in state:
                x = tfkl.Flatten()(state_component)
                x = tfkl.Dense(
                    units=flatten_units,
                    activation='sigmoid'
                )(x)
                components.append(x)
            _state = tfkl.Concatenate()(components)
        else:
            _state = state

        inputs = [state, action, latent_state]

        if latent_action is not None:
            inputs.append(latent_action)
        inputs.append(next_latent_state)
        # combine multiple state-components into _state
        _transition_loss_lipschitz_network = tfkl.Concatenate()([_state] + inputs[1:])
        _transition_loss_lipschitz_network = transition_loss_lipschitz_network(_transition_loss_lipschitz_network)
        _transition_loss_lipschitz_network = tfkl.Dense(
            units=1,
            activation=None,
            name='transition_loss_lipschitz_network_output'
        )(_transition_loss_lipschitz_network)

        super(TransitionLossLipschitzFunction, self).__init__(
            inputs=inputs,
            outputs=_transition_loss_lipschitz_network,
            name='transition_loss_lipschitz_network')
