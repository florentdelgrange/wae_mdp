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
            encode_action: bool = True
    ):
        if encode_action and (latent_action is None or next_latent_state is None):
            raise ValueError("The WAE is built to encode actions, so latent actions and next latent states are"
                             "required as input of the steady-state Lipschitz function.")
        if encode_action:
            network_input = tfkl.Concatenate(name='steady-state-lipschitz-fun-input')(
                [latent_state, latent_action, next_latent_state])
        else:
            network_input = tfkl.Concatenate(name='steady-state-lipschitz-fun-input')(
                [latent_state, next_latent_state])
        _steady_state_lipschitz_network = steady_state_lipschitz_network(network_input)
        _steady_state_lipschitz_network = tfkl.Dense(
            units=1,
            activation=None,
            name='steady_state_lipschitz_network_output'
        )(_steady_state_lipschitz_network)

        super(SteadyStateLipschitzFunction, self).__init__(
            inputs=([latent_state, latent_action, next_latent_state]
                    if encode_action else [latent_state, next_latent_state]),
            outputs=_steady_state_lipschitz_network,
            name='steady_state_lipschitz_network')


class TransitionLossLipschitzFunction(tfk.Model):
    def __init__(
            self,
            state: tfkl.Input,
            action: tfkl.Input,
            latent_state: tfkl.Input,
            latent_action: tfkl.Input,
            next_latent_state: tfkl.Input,
            transition_loss_lipschitz_network: tfk.Model,
    ):
        _transition_loss_lipschitz_network = tfkl.Concatenate()([
            state, action, latent_state, latent_action, next_latent_state])
        _transition_loss_lipschitz_network = transition_loss_lipschitz_network(_transition_loss_lipschitz_network)
        _transition_loss_lipschitz_network = tfkl.Dense(
            units=1,
            activation=None,
            name='transition_loss_lipschitz_network_output'
        )(_transition_loss_lipschitz_network)

        super(TransitionLossLipschitzFunction, self).__init__(
            inputs=[state, action, latent_state, latent_action, next_latent_state],
            outputs=_transition_loss_lipschitz_network,
            name='transition_loss_lipschitz_network')
