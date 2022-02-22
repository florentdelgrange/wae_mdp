from typing import Optional

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
            state: tfkl.Input,
            action: tfkl.Input,
            latent_state: tfkl.Input,
            next_latent_state: tfkl.Input,
            transition_loss_lipschitz_network: tfk.Model,
            latent_action: Optional[tfkl.Input] = None,
    ):
        inputs = [state, action, latent_state]
        if latent_action is not None:
            inputs.append(latent_action)
        inputs.append(next_latent_state)
        _transition_loss_lipschitz_network = tfkl.Concatenate()(inputs)
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
