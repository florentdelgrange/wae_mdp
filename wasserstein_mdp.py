from collections import namedtuple

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Callable, NamedTuple, List, Union
from tensorflow.python.keras import Model, Input, Sequential
from tensorflow.python.keras.layers import TimeDistributed, LSTM, Dense, Concatenate, Reshape
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd

from variational_mdp import VariationalMarkovDecisionProcess, EvaluationCriterion


class WassersteinRegularizerScaleFactor(NamedTuple):
    global_scaling: Optional[float] = None
    global_gradient_penalty_multiplier: Optional[float] = None
    stationary_scaling: Optional[float] = None
    stationary_gradient_penalty_multiplier: Optional[float] = None
    local_transition_loss_scaling: Optional[float] = None
    local_transition_loss_gradient_penalty_multiplier: Optional[float] = None
    action_successor_scaling: Optional[float] = None
    action_successor_gradient_penalty_multiplier: Optional[float] = None

    values = namedtuple('WassersteinRegularizer', ['scaling', 'gradient_penalty_multiplier'])

    if global_scaling is None and (stationary_scaling is None or
                                   local_transition_loss_scaling is None or
                                   action_successor_scaling is None):
        raise ValueError("Either a global scaling value or a unique scaling value for"
                         "each Wasserstein regularizer should be provided.")

    if global_gradient_penalty_multiplier is None and (stationary_gradient_penalty_multiplier is None or
                                                       local_transition_loss_gradient_penalty_multiplier is None or
                                                       action_successor_gradient_penalty_multiplier is None):
        raise ValueError("Either a global gradient penalty multiplier or a unique multiplier for"
                         "each Wasserstein regularizer should be provided.")

    @property
    def stationary(self):
        return self.values(
            scaling=self.global_scaling if self.global_scaling is not None else self.stationary_scaling,
            gradient_penalty_multiplier=(self.global_gradient_penalty_multiplier
                                         if self.global_gradient_penalty_multiplier is not None else
                                         self.stationary_gradient_penalty_multiplier))

    @property
    def local_transition_loss(self):
        return self.values(
            scaling=self.global_scaling if self.global_scaling is not None else self.local_transition_loss_scaling,
            gradient_penalty_multiplier=(self.global_gradient_penalty_multiplier
                                         if self.global_gradient_penalty_multiplier is not None else
                                         self.local_transition_loss_gradient_penalty_multiplier))

    @property
    def action_successor_loss(self):
        return self.values(
            scaling=self.global_scaling if self.global_scaling is not None else self.action_successor_scaling,
            gradient_penalty_multiplier=(self.global_gradient_penalty_multiplier
                                         if self.global_gradient_penalty_multiplier is not None else
                                         self.action_successor_gradient_penalty_multiplier))


class WassersteinMarkovDecisionProcess(VariationalMarkovDecisionProcess):
    def __init__(
            self,
            state_shape: Tuple[int, ...],
            action_shape: Tuple[int, ...],
            reward_shape: Tuple[int, ...],
            label_shape: Tuple[int, ...],
            discretize_action_space: bool,
            state_encoder_network: Model,
            action_encoder_network: Model,
            action_decoder_network: Model,
            transition_network: Model,
            reward_network: Model,
            decoder_network: Model,
            latent_policy_network: Model,
            steady_state_lipschitz_network: Model,
            transition_loss_lipschitz_network: Model,
            action_successor_lipschitz_network: Model,
            latent_state_size: int,
            number_of_discrete_actions: Optional[int] = None,
            state_encoder_pre_processing_network: Optional[Model] = None,
            state_decoder_pre_processing_network: Optional[Model] = None,
            time_stacked_states: bool = False,
            state_encoder_temperature: float = 2. / 3,
            state_prior_temperature: float = 1. / 2,
            action_encoder_temperature: Optional[float] = None,
            latent_policy_temperature: Optional[float] = None,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=1., global_gradient_penalty_multiplier=1.),
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            pre_loaded_model: bool = False,
            reset_state_label: bool = True,
            generative_optimizer: Optional = None,
            wasserstein_regularizer_optimizer: Optional = None,
            evaluation_window_size: int = 1,
            evaluation_criterion: EvaluationCriterion = EvaluationCriterion.MAX,
            importance_sampling_exponent: Optional[float] = 1.,
            importance_sampling_exponent_growth_rate: Optional[float] = 0.,
            time_stacked_lstm_units: int = 128,
            reward_bounds: Optional[Tuple[float, float]] = None,
            steady_state_logits: Optional[tf.Variable] = None,
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
            reward_bounds=reward_bounds)

        self.wasserstein_regularizer_scale_factor = wasserstein_regularizer_scale_factor
        self.mixture_components = None
        self._autoencoder_optimizer = generative_optimizer
        self._wasserstein_regularizer_optimizer = wasserstein_regularizer_optimizer
        self.action_discretizer = discretize_action_space

        if not self.action_discretizer:
            assert len(action_shape) == 1
            self.number_of_discrete_actions = self.action_shape[0]
        else:
            self.number_of_discrete_actions = number_of_discrete_actions

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

        state = Input(shape=state_shape, name="state")
        action = Input(shape=action_shape, name="action")
        latent_state = Input(shape=(self.latent_state_size,), name="latent_state")
        latent_action = Input(shape=(self.number_of_discrete_actions,), name="latent_action")
        next_latent_state = Input(shape=(self.latent_state_size,), name='next_latent_state')

        if not pre_loaded_model:
            # state encoder network
            self.state_encoder_network = self._initialize_state_encoder_network(
                state, state_encoder_network, state_encoder_pre_processing_network)
            # action encoder network
            if self.action_discretizer:
                self.action_encoder_network = self._initialize_action_encoder_network(
                    latent_state, action, action_encoder_network)
            else:
                self.action_encoder_network = None
            # transition network
            self.transition_network = self._initialize_transition_network(
                latent_state, latent_action, transition_network)
            # stationary distribution over latent states
            self.latent_steady_state_logits = self._initialize_latent_steady_state_logits()
            # latent policy
            self.latent_policy_network = self._initialize_latent_policy_network(latent_state, latent_policy_network)
            # reward function
            self.reward_network = self._initialize_reward_network(latent_state, latent_action, reward_network)
            # state reconstruction function
            self.reconstruction_network = self._initialize_state_reconstruction_network(
                next_latent_state, decoder_network, state_decoder_pre_processing_network)
            # action reconstruction function
            self.action_reconstruction_network = self._initialize_action_reconstruction_network(
                latent_state, latent_action, action_decoder_network)
            # steady state Lipschitz function
            self.steady_state_lipschitz_network = self._initialize_steady_state_lipschitz_function(
                latent_state, steady_state_lipschitz_network)
            # transition loss Lipschitz function
            self.transition_loss_lipschitz_network = self._initialize_transition_loss_lipschitz_function(
                state, action, next_latent_state, transition_loss_lipschitz_network)
            # action-successor Lipschitz function
            self.action_successor_lipschitz_network = self._initialize_action_successor_lipschitz_function(
                latent_state, latent_action, next_latent_state, action_successor_lipschitz_network)

        else:
            self.state_encoder_network = state_encoder_network
            self.action_encoder_network = action_encoder_network
            self.transition_network = transition_network
            self.latent_steady_state_logits = steady_state_logits
            self.latent_policy_network = latent_policy_network
            self.reward_network = reward_network
            self.reconstruction_network = decoder_network
            self.action_reconstruction_network = action_decoder_network
            self.steady_state_lipschitz_network = steady_state_lipschitz_network
            self.transition_loss_lipschitz_network = transition_loss_lipschitz_network
            self.action_successor_lipschitz_network = action_successor_lipschitz_network

    def _initialize_state_encoder_network(
            self,
            state: Input,
            state_encoder_network: Model,
            state_encoder_pre_processing_network: Optional[Model] = None
    ):
        if self.time_stacked_states:
            if state_encoder_pre_processing_network is not None:
                encoder = TimeDistributed(state_encoder_pre_processing_network)(state)
            else:
                encoder = state
            encoder = LSTM(units=self.time_stacked_lstm_units)(encoder)
            encoder = state_encoder_network(encoder)
        else:
            if state_encoder_pre_processing_network is not None:
                _state = state_encoder_pre_processing_network(state)
            else:
                _state = state
            encoder = state_encoder_network(_state)
        logits_layer = Dense(
            units=self.latent_state_size - self.atomic_props_dims,
            # allows avoiding exploding logits values and probability errors after applying a sigmoid
            activation=lambda x: self._encoder_softclip(x),
            name='encoder_latent_distribution_logits'
        )(encoder)

        return Model(
            inputs=state,
            outputs=logits_layer,
            name='state_encoder')

    def _initialize_action_encoder_network(
            self,
            latent_state: Input,
            action: Input,
            action_encoder_network: Model
    ):
        action_encoder = Concatenate(name='action_encoder_input')(
            [latent_state, action])
        action_encoder = action_encoder_network(action_encoder)
        action_encoder = Dense(
            units=self.number_of_discrete_actions,
            activation=None,
            name='action_encoder_exp_one_hot_logits'
        )(action_encoder)

        return Model(
            inputs=[latent_state, action],
            outputs=action_encoder,
            name="action_encoder")

    def _initialize_transition_network(
            self,
            latent_state: Input,
            latent_action: Input,
            transition_network: Model
    ):
        _transition_network = Concatenate(name='transition_network_input')(
            [latent_state, latent_action])
        _transition_network = transition_network(_transition_network)
        _next_label_logits = Dense(
            units=self.atomic_props_dims,
            activation=None,
            name='next_label_logits')(_transition_network)
        _next_latent_state_logits = Dense(
            units=self.latent_state_size - self.atomic_props_dims,
            activation=None,
            name='next_latent_state_logits')(_transition_network)

        return Model(
            inputs=[latent_state, latent_action],
            outputs=[_next_label_logits, _next_latent_state_logits],
            name="transition_network")

    def _initialize_latent_policy_network(
            self,
            latent_state: Input,
            latent_policy_network: Model
    ):
        _latent_policy_network = latent_policy_network(latent_state)
        _latent_policy_network = Dense(
            units=self.number_of_discrete_actions,
            activation=None,
            name='latent_policy_exp_one_hot_logits'
        )(self.latent_policy_network)
        return Model(
            inputs=latent_state,
            outputs=self.latent_policy_network,
            name='latent_policy_network')

    def _initialize_latent_steady_state_logits(self):
        return tf.Variable(
            initial_value=tf.ones(shape=(self.latent_state_size,)),
            trainable=True,
            name='latent_steady_state_distribution_logits')

    def _initialize_reward_network(
            self,
            latent_state: Input,
            latent_action: Input,
            reward_network: Model,
    ):
        _reward_network = Concatenate(name='reward_function_input')([latent_state, latent_action])
        _reward_network = reward_network(_reward_network)
        _reward_network = Dense(
            units=np.prod(self.reward_shape),
            activation=None if self._reward_softclip is None else lambda x: self._reward_softclip(x),
            name='reward_network_raw_output'
        )(_reward_network)
        _reward_network = Reshape(self.reward_shape, name='reward')(_reward_network)
        return Model(
            inputs=[latent_state, latent_action],
            outputs=_reward_network,
            name='reward_network')

    def _initialize_state_reconstruction_network(
            self,
            next_latent_state: Input,
            decoder_network: Model,
            state_decoder_pre_processing_network: Optional[Model] = None
    ):

        decoder = decoder_network(next_latent_state)
        if self.time_stacked_states:
            time_dimension = self.state_shape[0]
            _state_shape = self.state_shape[1:]

            if decoder.shape[-1] % time_dimension != 0:
                decoder = Dense(
                    units=decoder.shape[-1] + time_dimension - decoder.shape[-1] % time_dimension
                )(decoder)

            decoder = Reshape(
                target_shape=(time_dimension, decoder.shape[-1] // time_dimension)
            )(decoder)
            decoder = LSTM(
                units=self.time_stacked_lstm_units, return_sequences=True
            )(decoder)

            if state_decoder_pre_processing_network is not None:
                decoder = TimeDistributed(state_decoder_pre_processing_network)(decoder)

        else:
            if state_decoder_pre_processing_network is not None:
                decoder = state_decoder_pre_processing_network(decoder)
            _state_shape = self.state_shape

        decoder_output = Sequential([
            Dense(
                units=np.prod(_state_shape),
                activation=None,
                name='state_decoder_raw_output'),
            Reshape(
                target_shape=_state_shape,
                name='state_decoder_raw_output_reshape')],
            name="state_decoder")

        if self.time_stacked_states:
            decoder_output = TimeDistributed(decoder_output)(decoder)
        else:
            decoder_output = decoder_output(decoder)

        return Model(
            inputs=next_latent_state,
            outputs=decoder_output,
            name='state_reconstruction_network')

    def _initialize_action_reconstruction_network(
            self,
            latent_state: Input,
            latent_action: Input,
            action_decoder_network: Model
    ):
        action_reconstruction_network = Concatenate('action_reconstruction_input')([
            latent_state, latent_action])
        action_reconstruction_network = action_decoder_network(action_reconstruction_network)
        action_reconstruction_network = Dense(
            units=np.prod(self.action_shape),
            activation=None,
            name='action_reconstruction_network_raw_output'
        )(action_reconstruction_network)
        action_reconstruction_network = Reshape(
            target_shape=self.action_shape,
            name='action_reconstruction_network_output'
        )(action_reconstruction_network)

        return Model(
            inputs=[latent_state, latent_action],
            outputs=action_reconstruction_network,
            name='action_reconstruction_network')

    def _initialize_steady_state_lipschitz_function(
            self,
            latent_state: Input,
            steady_state_lipschitz_network: Model,
    ):
        _steady_state_lipschitz_network = steady_state_lipschitz_network(latent_state)
        _steady_state_lipschitz_network = Dense(
            units=1,
            activation=None,
            name='steady_state_lipschitz_network_output'
        )(_steady_state_lipschitz_network)

        return Model(
            inputs=latent_state,
            outputs=_steady_state_lipschitz_network,
            name='steady_state_lipschitz_network')

    def _initialize_transition_loss_lipschitz_function(
            self,
            state: Input,
            action: Input,
            next_latent_state: Input,
            transition_loss_lipschitz_network: Model,
    ):
        _transition_loss_lipschitz_network = Concatenate()([
            state, action, next_latent_state])
        _transition_loss_lipschitz_network = transition_loss_lipschitz_network(_transition_loss_lipschitz_network)
        _transition_loss_lipschitz_network = Dense(
            units=1,
            activation=None,
            name='transition_loss_lipschitz_network_output'
        )(_transition_loss_lipschitz_network)

        return Model(
            inputs=[state, action, next_latent_state],
            outputs=_transition_loss_lipschitz_network,
            name='transition_loss_lipschitz_network')

    def _initialize_action_successor_lipschitz_function(
            self,
            latent_state: Input,
            latent_action: Input,
            next_latent_state: Input,
            action_successor_lipschitz_network: Model,
    ):
        _action_successor_lipschitz_network = Concatenate()([
            latent_state, latent_action, next_latent_state])
        _action_successor_lipschitz_network = action_successor_lipschitz_network(_action_successor_lipschitz_network)
        _action_successor_lipschitz_network = Dense(
            units=1,
            activation=None,
            name='action_successor_lipschitz_network_output')

        return Model(
            inputs=[latent_state, latent_action, next_latent_state],
            outpus=_action_successor_lipschitz_network,
            name='action_successor_lipschitz_network')

    def attach_optimizer(
            self,
            optimizers: Optional[Union[Tuple, List]] = None,
            autoencoder_optimizer: Optional = None,
            wasserstein_regularize_optimizer: Optional = None
    ):
        assert len(optimizers) == 2
        assert optimizers is not None or (
                autoencoder_optimizer is not None and wasserstein_regularize_optimizer is not None)
        if optimizers is not None:
            [autoencoder_optimizer, wasserstein_regularize_optimizer] = optimizers
        self._autoencoder_optimizer = autoencoder_optimizer
        self._wasserstein_regularizer_optimizer = wasserstein_regularize_optimizer

    def detach_optimizer(self):
        autoencoder_optimizer = self._autoencoder_optimizer
        wasserstein_regularizer_optimizer = self._wasserstein_regularizer_optimizer
        self._autoencoder_optimizer = None
        self._wasserstein_regularizer_optimizer = None
        return autoencoder_optimizer, wasserstein_regularizer_optimizer

    def binary_encode(
            self,
            state: tf.Tensor,
            label: Optional[tf.Tensor] = None,
            temperature: Optional[float, tf.float32, tf.float64] = 0.
    ) -> tfd.Distribution:
        if temperature > 0:
            return self.relaxed_encoding(state, label, temperature)
        else:
            return super().binary_encode(state, label)

    def relaxed_encoding(
            self,
            state: tf.Tensor,
            temperature: Union[float, tf.float32, tf.float64],
            label: Optional[tf.Tensor] = None
    ) -> tfd.Distribution:
        logits = self.encoder_network(state)
        if label is not None:
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        return tfd.Independent(
            tfd.RelaxedBernoulli(
                logits=logits,
                temperature=temperature))

    def encode_action(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
            temperature: Optional[Union[float, tf.float32, tf.float64]] = 0.
    ) -> tfd.Distribution:
        logits = self.action_encoder_network([latent_state, action])
        if temperature > 0:
            return tfd.RelaxedOneHotCategorical(logits=logits, temperature=temperature)
        else:
            return tfd.OneHotCategorical(logits=logits)

    def decode(self, latent_state: tf.Tensor) -> tfd.Distribution:
        if self.time_stacked_states:
            return tfd.Independent(tfd.Deterministic(loc=self.reconstruction_network(latent_state)))
        else:
            return tfd.Deterministic(loc=self.reconstruction_network(latent_state))

    def decode_action(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tfd.Distribution:
        return tfd.Deterministic(loc=self.action_reconstruction_network([latent_state, latent_action]))

    def relaxed_latent_transition(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, next_label: Optional[tf.Tensor] = None,
            temperature: Union[float, tf.float32, tf.float64] = 1e-5
    ) -> tfd.Distribution:
        next_label_logits, next_latent_state_logits = self.transition_network([latent_state, latent_action])
        if next_label is not None:
            return tfd.Independent(
                tfd.RelaxedBernoulli(
                    logits=next_latent_state_logits,
                    temperature=temperature) if temperature > 0. else
                tfd.Bernoulli(logits=next_latent_state_logits))
        else:
            return tfd.JointDistributionSequential([
                tfd.Independent(
                    tfd.RelaxedBernoulli(
                        logits=next_label_logits,
                        temperature=temperature) if temperature > 0. else
                    tfd.Bernoulli(logits=next_label_logits)),
                lambda _next_label: tfd.Independent(
                    tfd.RelaxedBernoulli(
                        logits=next_latent_state_logits,
                        temperature=temperature) if temperature > 0. else
                    tfd.Bernoulli(logits=next_latent_state_logits))
            ])

    def discrete_latent_transition(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, next_label: Optional[tf.Tensor] = None
    ) -> tfd.Distribution:
        return self.relaxed_latent_transition(
            latent_state, latent_action, next_label, temperature=0.)

    def relaxed_markov_chain_latent_transition(
            self, latent_state: tf.Tensor, temperature: float = 1e-5, reparamaterize: bool = True
    ) -> tfd.Distribution:
        batch_size = tf.shape(latent_state)[0]
        latent_state = tf.tile(
            tf.expand_dims(latent_state, 1),
            multiples=[1, self.number_of_discrete_actions, 1])
        latent_action = tf.one_hot(
            tf.tile(
                tf.expand_dims(tf.range(self.number_of_discrete_actions), 0),
                multiples=[batch_size, 1]),
            depth=self.number_of_discrete_actions)

        next_label_logits, next_latent_state_logits = self.transition_network([latent_state, latent_action])
        latent_policy_logits = self.latent_policy_network(latent_state)

        return tfd.JointDistributionSequential([
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=latent_policy_logits),
                components_distribution=tfd.Independent(
                    tfd.RelaxedBernoulli(
                        logits=next_label_logits,
                        temperature=temperature) if temperature > 0. else
                    tfd.Bernoulli(logits=next_label_logits)),
                reparameterize=reparamaterize,
            ), lambda next_label: tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=latent_policy_logits),
                components_distribution=tfd.Independent(
                    tfd.RelaxedBernoulli(
                        logits=next_latent_state_logits,
                        temperature=temperature) if temperature > 0. else
                    tfd.Bernoulli(logits=next_latent_state_logits)),
                reparameterize=reparamaterize)
        ])

    def discrete_markov_chain_latent_transition(
            self, latent_state: tf.Tensor
    ) -> tfd.Distribution:
        return self.relaxed_markov_chain_latent_transition(
            latent_state, temperature=0., reparamaterize=False)

    def relaxed_latent_policy(
            self,
            latent_state: tf.Tensor,
            temperature: Union[float, tf.float32, tf.float64] = 1e-5,
    ) -> tfd.Distribution:
        if temperature > 0:
            return tfd.RelaxedOneHotCategorical(
                logits=self.latent_policy_network(latent_state),
                temperature=temperature)
        else:
            return tfd.OneHotCategorical(logits=self.latent_policy_network(latent_state))

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        return self.relaxed_latent_policy(latent_state, temperature=0.)

    def reward_distribution(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, *args, **kwargs
    ) -> tfd.Distribution:
        return tfd.Deterministic(loc=self.reward_network([latent_state, latent_action]))

    def latent_steady_state_distribution(self, temperature=0.) -> tfd.Distribution:
        if temperature > 0.:
            return tfd.RelaxedOneHotCategorical(logits=self.latent_steady_state_logits, temperature=temperature)
        else:
            return tfd.OneHotCategorical(logits=self.latent_steady_state_logits)

    @property
    def state_encoder_temperature(self):
        return self.encoder_temperature

    @property
    def state_encoder_prior_temperature(self):
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
        return self._action_encoder_temperature

    @latent_policy_temperature.setter
    def latent_policy_temperature(self, value):
        self._action_encoder_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='latent_policy_temperature')
