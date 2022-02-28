import gc
from collections import namedtuple

import tensorflow as tf
from typing import Tuple, Optional, Callable, NamedTuple, List, Union
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras.metrics import Mean, MeanSquaredError
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd

import tf_agents
from tf_agents.typing.types import Float, Int
from tf_agents.environments import tf_py_environment, tf_environment

import variational_action_discretizer
from layers.autoregressive_bernoulli import AutoRegressiveBernoulliNetwork
from layers.latent_policy import LatentPolicyNetwork
from layers.decoders import RewardNetwork, ActionReconstructionNetwork, StateReconstructionNetwork
from layers.encoders import StateEncoderNetwork, ActionEncoderNetwork, AutoRegressiveStateEncoderNetwork, EncodingType
from layers.lipschitz_functions import SteadyStateLipschitzFunction, TransitionLossLipschitzFunction
from layers.steady_state_network import SteadyStateNetwork
from util.io import dataset_generator, scan_model
from variational_mdp import VariationalMarkovDecisionProcess, EvaluationCriterion, debug_gradients, debug, epsilon
from verification.local_losses import estimate_local_losses_from_samples


class WassersteinRegularizerScaleFactor(NamedTuple):
    global_scaling: Optional[Float] = None
    global_gradient_penalty_multiplier: Optional[Float] = None
    steady_state_scaling: Optional[Float] = None
    steady_state_gradient_penalty_multiplier: Optional[Float] = None
    local_transition_loss_scaling: Optional[Float] = None
    local_transition_loss_gradient_penalty_multiplier: Optional[Float] = None

    values = namedtuple('WassersteinRegularizer', ['scaling', 'gradient_penalty_multiplier'])

    def sanity_check(self):
        if self.global_scaling is None and (self.steady_state_scaling is None or
                                            self.local_transition_loss_scaling):
            raise ValueError("Either a global scaling value or a unique scaling value for"
                             "each Wasserstein regularizer should be provided.")

        if self.global_gradient_penalty_multiplier is None and (
                self.steady_state_gradient_penalty_multiplier is None or
                self.local_transition_loss_gradient_penalty_multiplier is None):
            raise ValueError("Either a global gradient penalty multiplier or a unique multiplier for"
                             "each Wasserstein regularizer should be provided.")

    @property
    def stationary(self):
        self.sanity_check()
        return self.values(
            scaling=self.steady_state_scaling if self.steady_state_scaling is not None else self.global_scaling,
            gradient_penalty_multiplier=(self.steady_state_gradient_penalty_multiplier
                                         if self.steady_state_gradient_penalty_multiplier is not None else
                                         self.global_gradient_penalty_multiplier))

    @property
    def local_transition_loss(self):
        self.sanity_check()
        return self.values(
            scaling=(self.local_transition_loss_scaling
                     if self.local_transition_loss_scaling is not None else
                     self.global_scaling),
            gradient_penalty_multiplier=(self.local_transition_loss_gradient_penalty_multiplier
                                         if self.local_transition_loss_gradient_penalty_multiplier is not None else
                                         self.global_gradient_penalty_multiplier))


class WassersteinMarkovDecisionProcess(VariationalMarkovDecisionProcess):
    def __init__(
            self,
            state_shape: Tuple[int, ...],
            action_shape: Tuple[int, ...],
            reward_shape: Tuple[int, ...],
            label_shape: Tuple[int, ...],
            discretize_action_space: bool,
            state_encoder_network: tfk.Model,
            action_decoder_network: tfk.Model,
            transition_network: tfk.Model,
            reward_network: tfk.Model,
            decoder_network: tfk.Model,
            latent_policy_network: tfk.Model,
            steady_state_lipschitz_network: tfk.Model,
            transition_loss_lipschitz_network: tfk.Model,
            latent_state_size: int,
            number_of_discrete_actions: Optional[int] = None,
            action_encoder_network: Optional[tfk.Model] = None,
            state_encoder_pre_processing_network: Optional[tfk.Model] = None,
            state_decoder_pre_processing_network: Optional[tfk.Model] = None,
            time_stacked_states: bool = False,
            state_encoder_temperature: float = 2. / 3,
            state_prior_temperature: float = 1. / 2,
            action_encoder_temperature: Optional[Float] = None,
            latent_policy_temperature: Optional[Float] = None,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=1., global_gradient_penalty_multiplier=1.),
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            pre_loaded_model: bool = False,
            reset_state_label: bool = True,
            autoencoder_optimizer: Optional = None,
            wasserstein_regularizer_optimizer: Optional = None,
            entropy_regularizer_scale_factor: float = 0.,
            entropy_regularizer_decay_rate: float = 0.,
            entropy_regularizer_scale_factor_min_value: float = 0.,
            evaluation_window_size: int = 1,
            evaluation_criterion: EvaluationCriterion = EvaluationCriterion.MAX,
            importance_sampling_exponent: Optional[Float] = 1.,
            importance_sampling_exponent_growth_rate: Optional[Float] = 0.,
            time_stacked_lstm_units: int = 128,
            reward_bounds: Optional[Tuple[float, float]] = None,
            latent_stationary_network: Optional[tfk.Model] = None,
            action_entropy_regularizer_scaling: float = 1.,
            enforce_upper_bound: bool = False,
            squared_wasserstein: bool = False,
            n_critic: int = 5,
            trainable_prior: bool = True,
            state_encoder_type: EncodingType = EncodingType.AUTOREGRESSIVE,
            policy_based_decoding: bool = False,
            deterministic_state_embedding: bool = True,
    ):
        super(WassersteinMarkovDecisionProcess, self).__init__(
            state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
            encoder_network=None, transition_network=None, reward_network=None, decoder_network=None,
            time_stacked_states=time_stacked_states, latent_state_size=latent_state_size,
            encoder_temperature=state_encoder_temperature, prior_temperature=state_prior_temperature,
            encoder_temperature_decay_rate=encoder_temperature_decay_rate,
            prior_temperature_decay_rate=prior_temperature_decay_rate,
            pre_loaded_model=True, optimizer=None,
            reset_state_label=reset_state_label,
            evaluation_window_size=evaluation_window_size,
            evaluation_criterion=evaluation_criterion,
            importance_sampling_exponent=importance_sampling_exponent,
            importance_sampling_exponent_growth_rate=importance_sampling_exponent_growth_rate,
            time_stacked_lstm_units=time_stacked_lstm_units,
            reward_bounds=reward_bounds,
            entropy_regularizer_scale_factor=entropy_regularizer_scale_factor,
            entropy_regularizer_scale_factor_min_value=entropy_regularizer_scale_factor_min_value,
            entropy_regularizer_decay_rate=entropy_regularizer_decay_rate,
            deterministic_state_embedding=deterministic_state_embedding)

        self.wasserstein_regularizer_scale_factor = wasserstein_regularizer_scale_factor
        self.mixture_components = None
        self._autoencoder_optimizer = autoencoder_optimizer
        self._wasserstein_regularizer_optimizer = wasserstein_regularizer_optimizer
        self.action_discretizer = discretize_action_space
        self.policy_based_decoding = policy_based_decoding
        self.action_entropy_regularizer_scaling = action_entropy_regularizer_scaling
        self.enforce_upper_bound = enforce_upper_bound
        self.squared_wasserstein = squared_wasserstein
        self.n_critic = n_critic
        self.trainable_prior = trainable_prior

        if self.action_discretizer:
            self.number_of_discrete_actions = number_of_discrete_actions
        else:
            assert len(action_shape) == 1
            self.number_of_discrete_actions = self.action_shape[0]

        self._action_encoder_temperature = None
        if action_encoder_temperature is None:
            self.action_encoder_temperature = 1. / (self.number_of_discrete_actions - 1)
        else:
            self.action_encoder_temperature = action_encoder_temperature

        self._latent_policy_temperature = None
        if latent_policy_temperature is None:
            self.latent_policy_temperature = self.action_encoder_temperature / 1.5
        else:
            self.latent_policy_temperature = latent_policy_temperature

        self._sample_additional_transition = False
        # softclipping for latent states logits; 3 offers an probability error of about 5e-2
        # scale = 10.
        # self.softclip = tfb.Chain([tfb.Scale(scale), tfb.Tanh(), tfb.Scale(1. / scale)], name="softclip")
        # self.softclip = tfb.SoftClip(low=-scale, high=scale)
        self.softclip = tfb.Identity()

        if not pre_loaded_model:

            state = tfkl.Input(shape=state_shape, name="state")
            action = tfkl.Input(shape=action_shape, name="action")
            latent_state = tfkl.Input(shape=(self.latent_state_size,), name="latent_state")
            latent_action = tfkl.Input(shape=(self.number_of_discrete_actions,), name="latent_action")
            next_latent_state = tfkl.Input(shape=(self.latent_state_size,), name='next_latent_state')

            # state encoder network
            if state_encoder_type is EncodingType.AUTOREGRESSIVE:
                hidden_units, activation = scan_model(state_encoder_network)
                self.state_encoder_network = AutoRegressiveStateEncoderNetwork(
                    state_shape=state_shape,
                    activation=activation,
                    hidden_units=hidden_units,
                    latent_state_size=self.latent_state_size,
                    atomic_props_dims=self.atomic_props_dims,
                    time_stacked_states=self.time_stacked_states,
                    temperature=self.state_encoder_temperature,
                    time_stacked_lstm_units=self.time_stacked_lstm_units,
                    state_encoder_pre_processing_network=state_encoder_pre_processing_network,
                    output_softclip=self.softclip)
            else:
                self.state_encoder_network = StateEncoderNetwork(
                    state=state,
                    state_encoder_network=state_encoder_network,
                    latent_state_size=self.latent_state_size,
                    atomic_props_dims=self.atomic_props_dims,
                    time_stacked_states=self.time_stacked_states,
                    time_stacked_lstm_units=self.time_stacked_lstm_units,
                    state_encoder_pre_processing_network=state_encoder_pre_processing_network,
                    output_softclip=self.softclip,
                    lstm_output=state_encoder_type is EncodingType.LSTM)
            # action encoder network
            if self.action_discretizer and not self.policy_based_decoding:
                self.action_encoder_network = ActionEncoderNetwork(
                    latent_state=latent_state,
                    action=action,
                    number_of_discrete_actions=self.number_of_discrete_actions,
                    action_encoder_network=action_encoder_network,)
            else:
                self.action_encoder_network = None
            # transition network
            hidden_units, activation = scan_model(transition_network)
            self.transition_network = AutoRegressiveBernoulliNetwork(
                event_shape=(self.latent_state_size,),
                activation=activation,
                hidden_units=hidden_units,
                conditional_event_shape=(self.latent_state_size + self.number_of_discrete_actions,),
                temperature=self.state_prior_temperature,
                output_softclip=self.softclip,
                name='autoregressive_transition_network')
            # stationary distribution over latent states
            self.latent_stationary_network: AutoRegressiveBernoulliNetwork = SteadyStateNetwork(
                atomic_props_dims=self.atomic_props_dims,
                latent_state_size=latent_state_size,
                activation=activation,
                hidden_units=hidden_units,
                trainable_prior=trainable_prior,
                temperature=self.state_prior_temperature,
                output_softclip=self.softclip,
                name='latent_stationary_network')
            # latent policy
            self.latent_policy_network = LatentPolicyNetwork(
                latent_state=latent_state,
                latent_policy_network=latent_policy_network,
                number_of_discrete_actions=self.number_of_discrete_actions,)
            # reward function
            self.reward_network = RewardNetwork(
                latent_state=latent_state,
                latent_action=latent_action,
                next_latent_state=next_latent_state,
                reward_network=reward_network,
                reward_shape=self.reward_shape)
            # state reconstruction function
            self.reconstruction_network = StateReconstructionNetwork(
                next_latent_state=next_latent_state,
                decoder_network=decoder_network,
                state_shape=self.state_shape,
                time_stacked_states=self.time_stacked_states,
                state_decoder_pre_processing_network=state_decoder_pre_processing_network,
                time_stacked_lstm_units=self.time_stacked_lstm_units)
            # action reconstruction function
            if self.action_discretizer and not self.policy_based_decoding:
                self.action_reconstruction_network = ActionReconstructionNetwork(
                    latent_state=latent_state,
                    latent_action=latent_action,
                    action_decoder_network=action_decoder_network,
                    action_shape=self.action_shape)
            else:
                self.action_reconstruction_network = None
            # steady state Lipschitz function
            self.steady_state_lipschitz_network = SteadyStateLipschitzFunction(
                latent_state=latent_state,
                latent_action=latent_action if not self.policy_based_decoding else None,
                next_latent_state=next_latent_state,
                steady_state_lipschitz_network=steady_state_lipschitz_network, )
            # transition loss Lipschitz function
            self.transition_loss_lipschitz_network = TransitionLossLipschitzFunction(
                state=state,
                action=action,
                latent_state=latent_state,
                latent_action=latent_action if self.action_discretizer else None,
                next_latent_state=next_latent_state,
                transition_loss_lipschitz_network=transition_loss_lipschitz_network)

            if debug or True:
                self.state_encoder_network.summary()
                if self.action_discretizer and not self.policy_based_decoding:
                    self.action_encoder_network.summary()
                else:
                    print("No action encoder")
                self.transition_network.summary()
                self.latent_stationary_network.summary()
                self.latent_policy_network.summary()
                self.reward_network.summary()
                self.reconstruction_network.summary()
                if self.action_discretizer:
                    self.action_reconstruction_network.summary()
                self.steady_state_lipschitz_network.summary()
                self.transition_loss_lipschitz_network.summary()

        else:
            self.state_encoder_network = state_encoder_network
            self.action_encoder_network = action_encoder_network
            self.transition_network = transition_network
            self.latent_stationary_network = latent_stationary_network
            self.latent_policy_network = latent_policy_network
            self.reward_network = reward_network
            self.reconstruction_network = decoder_network
            self.action_reconstruction_network = action_decoder_network
            self.steady_state_lipschitz_network = steady_state_lipschitz_network
            self.transition_loss_lipschitz_network = transition_loss_lipschitz_network

        self.encoder_network = self.state_encoder_network

        self.loss_metrics = {
            'reconstruction_loss': Mean(name='reconstruction_loss'),
            'state_mse': MeanSquaredError(name='state_mse'),
            'action_mse': MeanSquaredError(name='action_mse'),
            'reward_mse': MeanSquaredError(name='reward_loss'),
            'transition_loss': Mean('transition_loss'),
            'latent_policy_entropy': Mean('latent_policy_entropy'),
            'steady_state_regularizer': Mean('steady_state_wasserstein_regularizer'),
            'gradient_penalty': Mean('gradient_penalty'),
            'marginal_state_encoder_entropy': Mean('marginal_state_encoder_entropy'),
            'state_encoder_entropy': Mean('state_encoder_entropy'),
            'entropy_regularizer': Mean('entropy_regularizer'),
            'transition_log_probs': Mean('transition_log_probs'),
            'binary_encoding_log_probs': Mean('binary_encoding_log_probs'),
        }
        if self.policy_based_decoding:
            self.loss_metrics['marginal_variance'] = Mean(name='marginal_variance')
        elif self.action_discretizer:
            self.loss_metrics.update({
                'marginal_action_encoder_entropy': Mean('marginal_action_encoder_entropy'),
                'action_encoder_entropy': Mean('action_encoder_entropy'),
            })

    def anneal(self):
        super().anneal()
        for var, decay_rate in [
            (self._action_encoder_temperature, self.encoder_temperature_decay_rate),
            (self._latent_policy_temperature, self.prior_temperature_decay_rate),
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

    def attach_optimizer(
            self,
            optimizers: Optional[Union[Tuple, List]] = None,
            autoencoder_optimizer: Optional = None,
            wasserstein_regularizer_optimizer: Optional = None
    ):
        assert optimizers is not None or (
                autoencoder_optimizer is not None and wasserstein_regularizer_optimizer is not None)
        if optimizers is not None:
            assert len(optimizers) == 2
            autoencoder_optimizer, wasserstein_regularizer_optimizer = optimizers
        self._autoencoder_optimizer = autoencoder_optimizer
        self._wasserstein_regularizer_optimizer = wasserstein_regularizer_optimizer

    def detach_optimizer(self):
        autoencoder_optimizer = self._autoencoder_optimizer
        wasserstein_regularizer_optimizer = self._wasserstein_regularizer_optimizer
        self._autoencoder_optimizer = None
        self._wasserstein_regularizer_optimizer = None
        return autoencoder_optimizer, wasserstein_regularizer_optimizer

    def binary_encode_state(self, state: Float, label: Optional[Float] = None) -> tfd.Distribution:
        return self.state_encoder_network.discrete_distribution(
            state=state, label=label)

    def relaxed_state_encoding(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            logistic: bool = False,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.state_encoder_network.relaxed_distribution(
            state=state, temperature=temperature, label=label, logistic=logistic)

    def discrete_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_encoder_network.discrete_distribution(
                latent_state=latent_state, action=action)
        else:
            return tfd.Deterministic(loc=action)

    def relaxed_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
            temperature
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_encoder_network.relaxed_distribution(
                latent_state=latent_state, action=action, temperature=temperature)
        else:
            return tfd.Deterministic(loc=action)

    def decode_state(self, latent_state: tf.Tensor) -> tfd.Distribution:
        return self.reconstruction_network.distribution(latent_state=latent_state)

    def decode_action(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
            *args, **kwargs
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_reconstruction_network.distribution(
                latent_state=latent_state, latent_action=latent_action)
        else:
            return tfd.Deterministic(loc=latent_action)

    def action_generator(
            self,
            latent_state: Float
    ) -> tfd.Distribution:
        if self.action_discretizer:
            batch_size = tf.shape(latent_state)[0]
            loc = self.action_reconstruction_network([
                tf.repeat(latent_state, self.number_of_discrete_actions, axis=0),
                tf.tile(tf.eye(self.number_of_discrete_actions), [batch_size, 1])
            ])
            loc = tf.reshape(
                loc,
                tf.concat([[batch_size], [self.number_of_discrete_actions], self.action_shape], axis=-1))
            return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    logits=self.discrete_latent_policy(latent_state).logits_parameter()),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=tf.ones(tf.shape(loc)) * 1e-6))
        else:
            return self.discrete_latent_policy(latent_state)

    def relaxed_latent_transition(
            self,
            latent_state: Float,
            latent_action: Float,
            temperature: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.transition_network.relaxed_distribution(
            conditional_input=tf.concat([latent_state, latent_action], axis=-1))

    def discrete_latent_transition(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, *args, **kwargs
    ) -> tfd.Distribution:
        return self.transition_network.discrete_distribution(
            conditional_input=tf.concat([latent_state, latent_action], axis=-1))

    def relaxed_markov_chain_latent_transition(
            self, latent_state: tf.Tensor, temperature: float = 1e-5, reparamaterize: bool = True
    ) -> tfd.Distribution:
        return NotImplemented

    def discrete_markov_chain_latent_transition(
            self, latent_state: tf.Tensor
    ) -> tfd.Distribution:
        return NotImplemented

    def relaxed_latent_policy(
            self,
            latent_state: tf.Tensor,
            temperature: Float = 1e-5,
    ) -> tfd.Distribution:
        return self.latent_policy_network.relaxed_distribution(
            latent_state=latent_state, temperature=temperature)

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        return self.latent_policy_network.discrete_distribution(latent_state=latent_state)

    def reward_distribution(
            self,
            latent_state: Float,
            latent_action: Float,
            next_latent_state: Float,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.reward_network.distribution(
            latent_state=latent_state,
            latent_action=latent_action,
            next_latent_state=next_latent_state)

    def markov_chain_reward_distribution(
            self,
            latent_state: Float,
            next_latent_state: Float,
    ) -> tfd.Distribution:
        batch_size = tf.shape(latent_state)[0]
        loc = self.reward_network([
            tf.repeat(latent_state, self.number_of_discrete_actions, axis=0),
            tf.tile(tf.eye(self.number_of_discrete_actions), [batch_size, 1]),
            tf.repeat(next_latent_state, self.number_of_discrete_actions, axis=0),
        ])
        loc = tf.reshape(loc, tf.concat([[batch_size], [self.number_of_discrete_actions], self.reward_shape], axis=-1))
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=self.discrete_latent_policy(latent_state).logits_parameter()),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=tf.ones(tf.shape(loc)) * 1e-6))

    def discrete_latent_steady_state_distribution(
            self,
            batch_size: Optional[int] = None,
            *args, **kwargs) -> tfd.Distribution:
        if batch_size is None:
            return self.latent_stationary_network.discrete_distribution(*args, **kwargs)
        else:
            return tfd.BatchBroadcast(
                self.latent_stationary_network.discrete_distribution(*args, **kwargs),
                with_shape=[batch_size])

    def relaxed_latent_steady_state_distribution(
            self,
            batch_size: Optional[int] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        if batch_size is None:
            return self.latent_stationary_network.relaxed_distribution(*args, **kwargs)
        else:
            return tfd.BatchBroadcast(
                self.latent_stationary_network.relaxed_distribution(*args, **kwargs),
                with_shape=[batch_size])

    def action_embedding_function(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tf.Tensor:

        if self.action_discretizer:
            decoder = self.decode_action(
                latent_state=tf.cast(latent_state, dtype=tf.float32),
                latent_action=tf.cast(
                    tf.one_hot(
                        latent_action,
                        depth=self.number_of_discrete_actions),
                    dtype=tf.float32),)
            if self.deterministic_state_embedding:
                return decoder.mode()
            else:
                return decoder.sample()
        else:
            return latent_action

    @staticmethod
    @tf.function
    def norm(x: Float, axis: int = -1):
        """
        to replace tf.norm(x, order=2, axis) which has numerical instabilities (the derivative can yields NaN).
        """
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + epsilon)

    @tf.function
    def __call__(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_key: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            *args, **kwargs
    ):
        batch_size = tf.shape(state)[0]
        # encoder sampling
        latent_state = self.relaxed_state_encoding(
            state,
            label=label,
            temperature=self.state_encoder_temperature,
        ).sample()
        next_latent_state = self.relaxed_state_encoding(
            next_state,
            label=next_label,
            temperature=self.state_encoder_temperature,
        ).sample()

        if self.policy_based_decoding:
            latent_action = self.relaxed_latent_policy(
                latent_state,
                temperature=self.latent_policy_temperature
            ).sample()
        else:
            latent_action = self.relaxed_action_encoding(
                latent_state,
                action,
                temperature=self.action_encoder_temperature
            ).sample()  # note that latent_action = action when self.action_discretizer is False

        (stationary_latent_state,
         stationary_latent_action,
         next_stationary_latent_state) = tfd.JointDistributionSequential([
            self.relaxed_latent_steady_state_distribution(batch_size=batch_size),
            lambda _latent_state: self.relaxed_latent_policy(
                latent_state=_latent_state,
                temperature=self.latent_policy_temperature),
            lambda _latent_action, _latent_state: self.relaxed_latent_transition(
                _latent_state,
                _latent_action, ),
        ]).sample()

        # next latent state from the latent transition function
        next_transition_latent_state = self.relaxed_latent_transition(
            latent_state,
            latent_action,
        ).sample()

        # reconstruction loss
        # the reward as well as the state and action reconstruction functions are deterministic
        mean_decoder_fn = tfd.JointDistributionSequential([
            self.decode_state(latent_state),
            self.action_generator(latent_state),
            self.markov_chain_reward_distribution(latent_state, next_latent_state),
            self.decode_state(next_latent_state)
        ]).mean

        if not self.policy_based_decoding or self.enforce_upper_bound:
            _state, _action, _reward, _next_state = tfd.JointDistributionSequential([
                self.decode_state(latent_state),
                self.decode_action(
                    latent_state,
                    latent_action),
                self.reward_distribution(
                    latent_state,
                    latent_action,
                    next_latent_state),
                self.decode_state(next_latent_state)
            ]).sample()
        else:
            _state, _action, _reward, _next_state = mean_decoder_fn()

        reconstruction_loss = (
                self.norm(state - _state, axis=1) +
                self.norm(action - _action, axis=1) +
                self.norm(reward - _reward, axis=1) +
                self.norm(next_state - _next_state, axis=1)
        )
        if self.squared_wasserstein or self.policy_based_decoding:
            reconstruction_loss = tf.square(reconstruction_loss)

        if self.policy_based_decoding:
            # marginal variance of the reconstruction
            if self.enforce_upper_bound:
                random_action, random_reward = _action, _reward
                _, _action, _reward, _ = mean_decoder_fn()
            else:
                random_action, random_reward = tfd.JointDistributionSequential([
                    self.decode_action(latent_state, latent_action),
                    self.reward_distribution(latent_state, latent_action, next_latent_state),
                ]).sample()
            y = tf.concat([_state, random_action, random_reward, _next_state], axis=-1)
            mean = tf.concat([_state, _action, _reward, _next_state], axis=-1)
            marginal_variance = (self.norm(y - mean, axis=1) ** 2. +
                                 self.norm(mean - tf.reduce_mean(mean), axis=1) ** 2)

        else:
            random_action = _action
            random_reward = _reward
            marginal_variance = 0.

        # Wasserstein regularizers and Lipschitz constraint
        if self.policy_based_decoding:
            x = [latent_state, next_transition_latent_state]
            y = [stationary_latent_state, next_stationary_latent_state]
        else:
            x = [latent_state, latent_action, next_transition_latent_state]
            y = [stationary_latent_state, stationary_latent_action, next_stationary_latent_state]
        steady_state_regularizer = tf.squeeze(
            self.steady_state_lipschitz_network(x) - self.steady_state_lipschitz_network(y))
        steady_state_gradient_penalty = self.compute_gradient_penalty(
            x=tf.concat(x, axis=-1),
            y=tf.concat(y, axis=-1),
            lipschitz_function=lambda _x: self.steady_state_lipschitz_network(
                [_x[:, :self.latent_state_size, ...]] +
                (
                    [_x[:, self.latent_state_size: self.latent_state_size + self.number_of_discrete_actions, ...]]
                    if not self.policy_based_decoding else
                    []
                ) +
                [_x[:, -self.latent_state_size:, ...]]))

        if self.action_discretizer:
            x = [state, action, latent_state, latent_action, next_latent_state]
            y = [state, action, latent_state, latent_action, next_transition_latent_state]
        else:
            x = [state, action, latent_state, next_latent_state]
            y = [state, action, latent_state, next_transition_latent_state]
        transition_loss_regularizer = tf.squeeze(
            self.transition_loss_lipschitz_network(x) - self.transition_loss_lipschitz_network(y))
        transition_loss_gradient_penalty = self.compute_gradient_penalty(
            x=next_latent_state,
            y=next_transition_latent_state,
            lipschitz_function=lambda _x: self.transition_loss_lipschitz_network(x[:-1] + [_x]))

        logits = self.state_encoder_network.get_logits(state, latent_state)
        entropy_regularizer = self.entropy_regularizer(
            state=state,
            latent_state=latent_state,
            logits=logits,
            action=action if not self.policy_based_decoding else None,
            sample_probability=sample_probability, )

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(reconstruction_loss +
                                          marginal_variance)))

        # loss metrics
        self.loss_metrics['reconstruction_loss'](reconstruction_loss)
        self.loss_metrics['state_mse'](state, _state)
        self.loss_metrics['state_mse'](next_state, _next_state)
        self.loss_metrics['action_mse'](action, random_action)
        self.loss_metrics['reward_mse'](reward, random_reward)
        self.loss_metrics['transition_loss'](transition_loss_regularizer)
        self.loss_metrics['steady_state_regularizer'](steady_state_regularizer)
        self.loss_metrics['gradient_penalty'](
            steady_state_gradient_penalty + transition_loss_gradient_penalty)
        self.loss_metrics['marginal_state_encoder_entropy'](
            self.marginal_state_encoder_entropy(logits=logits, sample_probability=sample_probability))
        self.loss_metrics['state_encoder_entropy'](
            tfd.Independent(
                tfd.Bernoulli(logits=logits),
                reinterpreted_batch_ndims=1
            ).entropy())
        self.loss_metrics['latent_policy_entropy'](
            self.discrete_latent_policy(latent_state).entropy())
        self.loss_metrics['transition_log_probs'](
            self.discrete_latent_transition(
                latent_state=tf.round(latent_state),
                latent_action=tf.one_hot(
                    tf.argmax(latent_action, axis=-1),
                    depth=self.number_of_discrete_actions)
            ).log_prob(tf.round(next_latent_state)))
        self.loss_metrics['binary_encoding_log_probs'](
            self.binary_encode_state(
                state=state
            ).log_prob(tf.round(latent_state)[..., self.atomic_props_dims:]))
        if self.action_discretizer and not self.policy_based_decoding:
            self.loss_metrics['marginal_action_encoder_entropy'](
                self.marginal_action_encoder_entropy(latent_state, action))
            self.loss_metrics['action_encoder_entropy'](
                self.discrete_action_encoding(latent_state, action).entropy())
        elif self.policy_based_decoding:
            self.loss_metrics['marginal_variance'](marginal_variance)
        self.loss_metrics['entropy_regularizer'](entropy_regularizer)

        if debug:
            tf.print("latent_state", latent_state, summarize=-1)
            tf.print("next_latent_state", next_latent_state, summarize=-1)
            tf.print("next_stationary_latent_state", next_stationary_latent_state, summarize=-1)
            tf.print("next_transition_latent_state", next_transition_latent_state, summarize=-1)
            tf.print("latent_action", latent_action, summarize=-1)
            tf.print("loss", tf.stop_gradient(
                reconstruction_loss + marginal_variance +
                self.wasserstein_regularizer_scale_factor.stationary.scaling * steady_state_regularizer +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling * transition_loss_regularizer))

        return {
            'reconstruction_loss': reconstruction_loss + marginal_variance,
            'steady_state_regularizer': steady_state_regularizer,
            'steady_state_gradient_penalty': steady_state_gradient_penalty,
            'transition_loss_regularizer': transition_loss_regularizer,
            'transition_loss_gradient_penalty': transition_loss_gradient_penalty,
            'entropy_regularizer': entropy_regularizer if self.entropy_regularizer_scale_factor > epsilon else 0.,
        }

    def marginal_state_encoder_entropy(
            self,
            state: Optional[Float] = None,
            latent_state: Optional[Float] = None,
            logits: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
    ) -> Float:

        if logits is None:
            if state is None or latent_state is None:
                raise ValueError("A state and its encoding (i.e., as a latent state) "
                                 "should be provided when logits are not.")

            logits = self.state_encoder_network.get_logits(state, latent_state)

        if sample_probability is None:
            regularizer = tf.reduce_mean(
                - tf.sigmoid(logits) * tf.math.log(tf.reduce_mean(tf.sigmoid(logits), axis=0) + epsilon)
                - tf.sigmoid(-logits) * tf.math.log(tf.reduce_mean(tf.sigmoid(-logits), axis=0) + epsilon),
                axis=0)
        else:
            is_weights = (tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability) ** self.is_exponent
            regularizer = tf.reduce_mean(
                - tf.sigmoid(logits) * tf.math.log(
                    tf.reduce_mean(tf.expand_dims(is_weights, -1) * tf.sigmoid(logits), axis=0) + epsilon)
                - tf.sigmoid(-logits) * tf.math.log(
                    tf.reduce_mean(tf.expand_dims(is_weights, -1) * tf.sigmoid(-logits), axis=0) + epsilon),
                axis=0)
        return tf.reduce_sum(regularizer)

    def marginal_action_encoder_entropy(
            self,
            latent_state: Optional[Float] = None,
            action: Optional[Float] = None,
            logits: Optional[Float] = None,
    ) -> Float:
        if logits is None and (latent_state is None or action is None):
            raise ValueError("You should either provide the logits of the action distribution or a latent state"
                             " and an action to compute the marginal entropy")
        if logits is None:
            logits = self.discrete_action_encoding(latent_state, action).logits_parameter()
        batch_size = tf.cast(tf.shape(logits)[0], tf.float32)
        return -1. * tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.softmax(logits) * (
                    tf.reduce_logsumexp(
                        logits - tf.expand_dims(
                            tf.reduce_logsumexp(logits, axis=-1),
                            axis=-1),
                        axis=0) - tf.math.log(batch_size)),
                axis=-1),
            axis=0)


    @tf.function
    def entropy_regularizer(
            self,
            state: tf.Tensor,
            label: Optional[Float] = None,
            latent_state: Optional[Float] = None,
            logits: Optional[Float] = None,
            action: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            include_state_entropy: bool = True,
            include_action_entropy: bool = True,
            *args, **kwargs
    ) -> Float:
        if latent_state is None:
            if label is None:
                raise ValueError("either a latent state or a label should be provided")
            else:
                latent_state = self.relaxed_state_encoding(
                    state, label=label, temperature=self.state_encoder_temperature)

        regularizer = 0.

        if include_state_entropy:
            if logits is None:
                logits = self.state_encoder_network.get_logits(state, latent_state)
            regularizer += self.marginal_state_encoder_entropy(
                logits=logits,
                sample_probability=sample_probability)
            regularizer -= tfd.Independent(
                tfd.Bernoulli(logits=logits),
                reinterpreted_batch_ndims=1
            ).entropy()

        if include_action_entropy:
            if action is None or not self.action_discretizer:
                regularizer += self.action_entropy_regularizer_scaling * tf.reduce_mean(
                    self.discrete_latent_policy(latent_state).entropy(),
                    axis=0)
            else:
                logits = self.discrete_action_encoding(latent_state, action).logits_parameter()
                regularizer += self.action_entropy_regularizer_scaling * (
                               self.marginal_action_encoder_entropy(logits=logits) -
                               tf.reduce_mean(tfd.Categorical(logits=logits).entropy(), axis=0))
        return regularizer

    @tf.function
    def compute_gradient_penalty(
            self,
            x: Float,
            y: Float,
            lipschitz_function: Callable[[Float], Float],
    ):
        noise = tf.random.uniform(shape=(tf.shape(x)[0], 1), minval=0., maxval=1.)
        straight_lines = noise * x + (1. - noise) * y
        gradients = tf.gradients(lipschitz_function(straight_lines), straight_lines)[0]
        return tf.square(self.norm(gradients, axis=1) - 1.)

    def eval(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_probability: Optional[Float] = None,
            additional_transition_batch: Optional[Tuple[Float, ...]] = None,
            *args, **kwargs
    ):
        batch_size = tf.shape(state)[0]
        # sampling
        # encoders
        latent_state = self.binary_encode_state(state, label).sample()
        next_latent_state = self.binary_encode_state(next_state, next_label).sample()
        if self.policy_based_decoding:
            latent_action = tf.cast(self.discrete_latent_policy(latent_state).sample(), tf.float32)
        else:
            latent_action = tf.cast(self.discrete_action_encoding(latent_state, action).sample(), tf.float32)

        # latent steady-state distribution
        stationary_latent_state = self.discrete_latent_steady_state_distribution().sample(batch_size)
        stationary_latent_action = self.discrete_latent_policy(stationary_latent_state).sample()
        next_stationary_latent_state = self.discrete_latent_transition(
            latent_state=stationary_latent_state,
            latent_action=stationary_latent_action
        ).sample()
        next_stationary_latent_state = tf.cast(next_stationary_latent_state, tf.float32)

        # next latent state from the latent transition function
        next_transition_latent_state = self.discrete_latent_transition(
            latent_state,
            latent_action,
        ).sample()

        # reconstruction loss
        # the reward as well as the state and action reconstruction functions are deterministic
        _action, _reward, _next_state = tfd.JointDistributionSequential([
            self.decode_action(
                latent_state,
                latent_action) if not self.policy_based_decoding else
            tfd.Deterministic(loc=self.action_generator(latent_state).mean()),
            self.reward_distribution(
                latent_state,
                latent_action,
                next_latent_state) if not self.policy_based_decoding else
            tfd.Deterministic(loc=self.markov_chain_reward_distribution(latent_state, next_latent_state).mean()),
            self.decode_state(next_latent_state)
        ]).sample()

        reconstruction_loss = (
                tf.norm(action - _action, ord=2, axis=1) +
                tf.norm(reward - _reward, ord=2, axis=1) +
                tf.norm(next_state - _next_state, ord=2, axis=1))
        if self.policy_based_decoding or self.squared_wasserstein:
            reconstruction_loss = tf.square(reconstruction_loss)

        # marginal variance of the reconstruction
        if self.policy_based_decoding:
            random_action, random_reward = tfd.JointDistributionSequential([
                self.decode_action(latent_state, latent_action),
                self.reward_distribution(latent_state, latent_action, next_latent_state),
            ]).sample()
            y = tf.concat([random_action, random_reward, _next_state], axis=-1)
            mean = tf.concat([_action, _reward, _next_state], axis=-1)
            marginal_variance = tf.reduce_sum((y - mean) ** 2. + (mean - tf.reduce_mean(mean)) ** 2., axis=-1)
        else:
            marginal_variance = 0.

        # Wasserstein regularizers and Lipschitz constraint
        if self.policy_based_decoding:
            x = [latent_state, next_transition_latent_state]
            y = [stationary_latent_state, next_stationary_latent_state]
        else:
            x = [latent_state, latent_action, next_transition_latent_state]
            y = [stationary_latent_state, stationary_latent_action, next_stationary_latent_state]
        steady_state_regularizer = tf.squeeze(
            self.steady_state_lipschitz_network(x) - self.steady_state_lipschitz_network(y))

        if self.action_discretizer:
            x = [state, action, latent_state, latent_action, next_latent_state]
            y = [state, action, latent_state, latent_action, next_transition_latent_state]
        else:
            x = [state, action, latent_state, next_latent_state]
            y = [state, action, latent_state, next_transition_latent_state]
        transition_loss_regularizer = tf.squeeze(
            self.transition_loss_lipschitz_network(x) - self.transition_loss_lipschitz_network(y))

        if debug:
            latent_policy = self.discrete_latent_policy(latent_state)
            tf.print("latent policy", latent_policy,
                     '\n latent policy: probs parameter', latent_policy.probs_parameter())
            tf.print("latent action ~ latent policy", latent_policy.sample())
            tf.print("latent_action hist:", tf.cast(tf.argmax(latent_action, axis=1), tf.int64))

        return {
            'reconstruction_loss': reconstruction_loss + marginal_variance,
            'wasserstein_regularizer':
                (self.wasserstein_regularizer_scale_factor.stationary.scaling * steady_state_regularizer +
                 self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling * transition_loss_regularizer),
            'latent_states': tf.concat([tf.cast(latent_state, tf.int64), tf.cast(next_latent_state, tf.int64)], axis=0),
            'latent_actions': (tf.cast(tf.argmax(latent_action, axis=1), tf.int64)
                               if self.action_discretizer else
                               tf.cast(tf.argmax(stationary_latent_action, axis=1), tf.int64))
        }

    @tf.function
    def compute_loss(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
            additional_transition_batch: Optional[Tuple[Float]] = None,
            *args, **kwargs
    ):
        output = self(state, label, action, reward, next_state, next_label,
                      sample_key=sample_key,
                      sample_probability=sample_probability,
                      additional_transition_batch=additional_transition_batch)

        if debug:
            tf.print('call output', output, summarize=-1)

        # Importance sampling weights (is) for prioritized experience replay
        if sample_probability is not None:
            is_weights = (tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability) ** self.is_exponent
        else:
            is_weights = 1.

        reconstruction_loss = output['reconstruction_loss']
        wasserstein_loss = (
                self.wasserstein_regularizer_scale_factor.stationary.scaling *
                output['steady_state_regularizer'] +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling *
                output['transition_loss_regularizer']
        )
        gradient_penalty = (
                self.wasserstein_regularizer_scale_factor.stationary.scaling *
                self.wasserstein_regularizer_scale_factor.stationary.gradient_penalty_multiplier *
                output['steady_state_gradient_penalty'] +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling *
                self.wasserstein_regularizer_scale_factor.local_transition_loss.gradient_penalty_multiplier *
                output['transition_loss_gradient_penalty']
        )
        entropy_regularizer = self.entropy_regularizer_scale_factor * output['entropy_regularizer']

        loss = lambda minimize: tf.reduce_mean(
            (-1.) ** (1. - minimize) * is_weights * (
                    minimize * reconstruction_loss +
                    wasserstein_loss +
                    (minimize - 1.) * gradient_penalty -
                    minimize * entropy_regularizer
            )
        )

        return {'min': loss(1.), 'max': loss(0.)}

    @property
    def state_encoder_temperature(self):
        return self.encoder_temperature

    @property
    def state_prior_temperature(self):
        return self.prior_temperature

    @property
    def action_encoder_temperature(self):
        return self._action_encoder_temperature

    @action_encoder_temperature.setter
    def action_encoder_temperature(self, value):
        self._action_encoder_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='action_encoder_temperature')

    @property
    def latent_policy_temperature(self):
        return self._latent_policy_temperature

    @latent_policy_temperature.setter
    def latent_policy_temperature(self, value):
        self._latent_policy_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='latent_policy_temperature')

    @property
    def inference_variables(self):
        if self.action_discretizer and not self.policy_based_decoding:
            return self.state_encoder_network.trainable_variables + self.action_encoder_network.trainable_variables
        else:
            return self.state_encoder_network.trainable_variables

    @property
    def generator_variables(self):
        variables = self.latent_stationary_network.trainable_variables
        if self.action_discretizer:
            variables += self.action_reconstruction_network.trainable_variables
        for network in [self.transition_network,
                        self.latent_policy_network,
                        self.reward_network,
                        self.reconstruction_network]:
            variables += network.trainable_variables
        return variables

    @property
    def wasserstein_variables(self):
        return (self.steady_state_lipschitz_network.trainable_variables +
                self.transition_loss_lipschitz_network.trainable_variables)

    def _compute_apply_gradients(
            self, state, label, action, reward, next_state, next_label,
            autoencoder_variables=None, wasserstein_regularizer_variables=None,
            sample_key=None, sample_probability=None,
            additional_transition_batch=None,
            step: Int = None,
            *args, **kwargs
    ):
        if autoencoder_variables is None and wasserstein_regularizer_variables is None:
            raise ValueError("Must pass autoencoder and/or wasserstein regularizer variables")
        if step is None:
            step = self.n_critic

        def numerical_error(x, list_of_tensors=False):
            detected = False
            if not list_of_tensors:
                x = [x]
            for value in x:
                if value is not None:
                    detected = detected or tf.reduce_any(tf.logical_or(
                        tf.math.is_nan(value),
                        tf.math.is_inf(value)))
            return detected

        with tf.GradientTape(persistent=True) as tape:
            loss = self.compute_loss(
                state, label, action, reward, next_state, next_label,
                sample_key=sample_key, sample_probability=sample_probability,
                additional_transition_batch=additional_transition_batch)

        for optimization_direction, variables in {
            'max': wasserstein_regularizer_variables, 'min': autoencoder_variables
        }.items():
            if (
                    variables is not None and
                    (not debug or not numerical_error(loss[optimization_direction])) and
                    (optimization_direction == 'max' or
                     (step % self.n_critic == 0 and optimization_direction == 'min'))
            ):
                gradients = tape.gradient(loss[optimization_direction], variables)
                optimizer = {
                    'max': self._wasserstein_regularizer_optimizer,
                    'min': self._autoencoder_optimizer,
                }[optimization_direction]

                if not numerical_error(gradients, list_of_tensors=True):
                    optimizer.apply_gradients(zip(gradients, variables))

                if debug_gradients:
                    for gradient, variable in zip(gradients, variables):
                        tf.print("Gradient for {} (direction={}):".format(variable.name, optimization_direction),
                                 gradient)

        del tape

        return {'loss_minimizer': loss['min'], 'loss_maximizer': loss['max']}

    @tf.function
    def compute_apply_gradients(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_key: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            additional_transition_batch: Optional[Tuple[Float]] = None,
            step: Int = None,
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            autoencoder_variables=self.inference_variables + self.generator_variables,
            wasserstein_regularizer_variables=self.wasserstein_variables,
            sample_key=sample_key, sample_probability=sample_probability,
            additional_transition_batch=additional_transition_batch,
            step=step)

    @tf.function
    def inference_update(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            autoencoder_variables=self.generator_variables,
            wasserstein_regularizer_variables=self.wasserstein_variables,
            sample_key=sample_key, sample_probability=sample_probability)

    @tf.function
    def generator_update(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            autoencoder_variables=self.generator_variables,
            wasserstein_regularizer_variables=self.wasserstein_variables,
            sample_key=sample_key, sample_probability=sample_probability)

    def mean_latent_bits_used(self, inputs, eps=1e-3, deterministic=True):
        state, label, action, reward, next_state, next_label = inputs[:6]
        latent_distribution = self.binary_encode_state(state, label)
        latent_state = latent_distribution.sample()
        if deterministic:
            mean = tf.reduce_mean(latent_distribution.mode(), axis=0)
        else:
            mean = tf.reduce_mean(latent_distribution.mean(), axis=0)
        check = lambda x: 1. if 1. - eps > x > eps else 0.
        mbu = {'mean_state_bits_used': tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()}
        if self.action_discretizer:
            mean = tf.reduce_mean(
                self.discrete_action_encoding(latent_state, action).probs_parameter()
                if not self.policy_based_decoding else
                self.discrete_latent_policy(latent_state).probs_parameter(),
                axis=0)
            check = lambda x: 1 if 1 - eps > x > eps else 0
            mean_bits_used = tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

            mbu.update({'mean_action_bits_used': mean_bits_used})
        return mbu

    def estimate_local_losses_from_samples(
            self,
            environment: tf_py_environment.TFPyEnvironment,
            steps: int,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            estimate_transition_function_from_samples: bool = False,
            assert_estimated_transition_function_distribution: bool = False,
            replay_buffer_max_frames: Optional[int] = int(1e5),
            reward_scaling: Optional[float] = 1.,
    ):
        if self.time_stacked_states:
            labeling_function = lambda x: labeling_function(x)[:, -1, ...]

        class LatentTransitionFunction:
            def __init__(self, discrete_latent_transition, latent_state, latent_action):
                self._distribution = discrete_latent_transition(
                    latent_state=tf.cast(latent_state, tf.float32),
                    latent_action=tf.cast(latent_action, tf.float32))

            def prob(self, label, state_without_label, **kwargs) -> Float:
                return self._distribution.prob(tf.concat([label, state_without_label], axis=-1))

        return estimate_local_losses_from_samples(
            environment=environment,
            latent_policy=self.get_latent_policy(action_dtype=tf.int64),
            steps=steps,
            latent_state_size=self.latent_state_size,
            number_of_discrete_actions=self.number_of_discrete_actions,
            state_embedding_function=self.state_embedding_function,
            probabilistic_state_embedding=None if self.deterministic_state_embedding else self.binary_encode_state,
            action_embedding_function=self.action_embedding_function,
            latent_reward_function=lambda latent_state, latent_action, next_latent_state: (
                self.reward_distribution(
                    latent_state=tf.cast(latent_state, dtype=tf.float32),
                    latent_action=tf.cast(latent_action, dtype=tf.float32),
                    next_latent_state=tf.cast(next_latent_state, dtype=tf.float32),
                ).mode()),
            labeling_function=labeling_function,
            latent_transition_function=lambda latent_state, latent_action: LatentTransitionFunction(
                discrete_latent_transition=self.discrete_latent_transition,
                latent_state=latent_state,
                latent_action=latent_action),
            estimate_transition_function_from_samples=estimate_transition_function_from_samples,
            replay_buffer_max_frames=replay_buffer_max_frames,
            reward_scaling=reward_scaling,
            atomic_prop_dims=self.atomic_props_dims,)

    def eval_and_save(
            self,
            eval_steps: int,
            global_step: tf.Variable,
            dataset: Optional = None,
            dataset_iterator: Optional = None,
            batch_size: Optional[int] = None,
            save_directory: Optional[str] = None,
            log_name: Optional[str] = None,
            train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
            eval_policy_driver: Optional[tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver] = None,
            local_losses_estimator: Optional[Callable] = None,
            *args, **kwargs
    ):

        if (dataset is None) == (dataset_iterator is None or batch_size is None):
            raise ValueError("Must either provide a dataset or a dataset iterator + batch size.")

        if dataset is not None:
            batch_size = eval_steps
            dataset_iterator = iter(dataset.batch(
                batch_size=batch_size,
                drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
            eval_steps = 1 if eval_steps > 0 else 0

        metrics = {
            'eval_loss': tf.metrics.Mean(),
            'eval_reconstruction_loss': tf.metrics.Mean(),
            'eval_wasserstein_regularizer': tf.metrics.Mean(),
        }

        data = {'states': None, 'actions': None}
        avg_rewards = None
        local_losses_metrics = None

        if eval_steps > 0:
            eval_progressbar = Progbar(
                target=(eval_steps + 1) * batch_size, interval=0.1, stateful_metrics=['eval_ELBO'])

            tf.print("\nEvalutation over {} step(s)".format(eval_steps))

            for step in range(eval_steps):
                x = next(dataset_iterator)
                if self._sample_additional_transition:
                    x_prime = next(dataset_iterator)
                else:
                    x_prime = None

                if len(x) >= 8:
                    sample_probability = x[7]
                    # we consider is_exponent=1 for evaluation
                    is_weights = tf.reduce_min(sample_probability) / sample_probability
                else:
                    sample_probability = None
                    is_weights = 1.

                evaluation = self.eval(
                    *(x[:6]), sample_probability=sample_probability, additional_transition_batch=x_prime)
                for value in ('states', 'actions'):
                    latent = evaluation['latent_' + value]
                    data[value] = latent if data[value] is None else tf.concat([data[value], latent], axis=0)
                for value in ('loss', 'reconstruction_loss', 'wasserstein_regularizer'):
                    if value == 'loss':
                        metrics['eval_' + value](tf.reduce_mean(
                            is_weights * (evaluation['reconstruction_loss'] + evaluation['wasserstein_regularizer'])))
                    else:
                        metrics['eval_' + value](tf.reduce_mean(is_weights * evaluation[value]))
                eval_progressbar.add(batch_size, values=[('eval_loss', metrics['eval_loss'].result())])
            tf.print('\n')
        
        if eval_policy_driver is not None:
            avg_rewards = self.eval_policy(
                eval_policy_driver=eval_policy_driver,
                train_summary_writer=train_summary_writer,
                global_step=global_step)

        if local_losses_estimator is not None:
            local_losses_metrics = local_losses_estimator()

        if train_summary_writer is not None and eval_steps > 0:
            with train_summary_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value.result(), step=global_step)
                for value in ('states', 'actions'):
                    if data[value] is not None:
                        if value == 'states':
                            data[value] = tf.reduce_sum(
                                data[value] * 2 ** tf.range(tf.cast(self.latent_state_size, dtype=tf.int64)),
                                axis=-1)
                        tf.summary.histogram('{}_frequency'.format(value[:-1]), data[value],
                                             step=global_step, buckets=32)
                if local_losses_metrics is not None:
                    tf.summary.scalar('local_reward_loss', local_losses_metrics.local_reward_loss, step=global_step)
                    tf.summary.scalar('local_transition_loss',
                            local_losses_metrics.local_transition_loss, step=global_step)
                    if local_losses_metrics.local_transition_loss_transition_function_estimation is not None:
                        tf.summary.scalar('local_transition_loss_empirical_transition_function',
                                local_losses_metrics.local_transition_loss_transition_function_estimation,
                                step=global_step)

        if local_losses_metrics is not None:
            tf.print('Local reward loss: {:.2f}'.format(local_losses_metrics.local_reward_loss))
            tf.print('Local transition loss: {:.2f}'.format(local_losses_metrics.local_transition_loss))
            tf.print('Local transition loss (empirical transition function): {:.2f}'
                        ''.format(local_losses_metrics.local_transition_loss_transition_function_estimation))
            local_losses_metrics.print_time_metrics()

        if eval_steps > 0:
            print('eval loss: ', metrics['eval_loss'].result().numpy())

        if eval_policy_driver is not None or eval_steps > 0:
            self.assign_score(
                score=avg_rewards if avg_rewards is not None else metrics['eval_loss'].result(),
                checkpoint_model=save_directory is not None and log_name is not None,
                save_directory=save_directory,
                model_name=log_name,
                training_step=global_step.numpy())

        gc.collect()

        return metrics['eval_loss'].result()
