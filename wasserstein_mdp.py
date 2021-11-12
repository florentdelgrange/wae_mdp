import gc
from collections import namedtuple

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Callable, NamedTuple, List, Union
from tensorflow.python.keras import Model, Input, Sequential
from tensorflow.python.keras.layers import TimeDistributed, LSTM, Dense, Concatenate, Reshape
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras.metrics import Mean, MeanSquaredError
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd

import tf_agents
from tf_agents.typing.types import Float
from tf_agents.environments import tf_py_environment, tf_environment

import variational_action_discretizer
from util.io import dataset_generator
from variational_mdp import VariationalMarkovDecisionProcess, EvaluationCriterion, debug_gradients, debug, epsilon
from verification.local_losses import estimate_local_losses_from_samples


class WassersteinRegularizerScaleFactor(NamedTuple):
    global_scaling: Optional[Float] = None
    global_gradient_penalty_multiplier: Optional[Float] = None
    steady_state_scaling: Optional[Float] = None
    steady_state_gradient_penalty_multiplier: Optional[Float] = None
    local_transition_loss_scaling: Optional[Float] = None
    local_transition_loss_gradient_penalty_multiplier: Optional[Float] = None
    action_successor_scaling: Optional[Float] = None
    action_successor_gradient_penalty_multiplier: Optional[Float] = None

    values = namedtuple('WassersteinRegularizer', ['scaling', 'gradient_penalty_multiplier'])

    def sanity_check(self):
        if self.global_scaling is None and (self.steady_state_scaling is None or
                                            self.local_transition_loss_scaling is None or
                                            self.action_successor_scaling is None):
            raise ValueError("Either a global scaling value or a unique scaling value for"
                             "each Wasserstein regularizer should be provided.")

        if self.global_gradient_penalty_multiplier is None and (
                self.steady_state_gradient_penalty_multiplier is None or
                self.local_transition_loss_gradient_penalty_multiplier is None or
                self.action_successor_gradient_penalty_multiplier is None):
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

    @property
    def action_successor_loss(self):
        self.sanity_check()
        return self.values(
            scaling=(self.action_successor_scaling
                     if self.action_successor_scaling is not None else
                     self.global_scaling),
            gradient_penalty_multiplier=(self.action_successor_gradient_penalty_multiplier
                                         if self.action_successor_gradient_penalty_multiplier is not None else
                                         self.global_gradient_penalty_multiplier))


class WassersteinMarkovDecisionProcess(VariationalMarkovDecisionProcess):
    def __init__(
            self,
            state_shape: Tuple[int, ...],
            action_shape: Tuple[int, ...],
            reward_shape: Tuple[int, ...],
            label_shape: Tuple[int, ...],
            discretize_action_space: bool,
            state_encoder_network: Model,
            action_decoder_network: Model,
            transition_network: Model,
            reward_network: Model,
            decoder_network: Model,
            latent_policy_network: Model,
            steady_state_lipschitz_network: Model,
            transition_loss_lipschitz_network: Model,
            latent_state_size: int,
            number_of_discrete_actions: Optional[int] = None,
            action_encoder_network: Optional[Model] = None,
            state_encoder_pre_processing_network: Optional[Model] = None,
            state_decoder_pre_processing_network: Optional[Model] = None,
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
            evaluation_window_size: int = 1,
            evaluation_criterion: EvaluationCriterion = EvaluationCriterion.MAX,
            importance_sampling_exponent: Optional[Float] = 1.,
            importance_sampling_exponent_growth_rate: Optional[Float] = 0.,
            time_stacked_lstm_units: int = 128,
            reward_bounds: Optional[Tuple[float, float]] = None,
            steady_state_logits: Optional[tf.Variable] = None,
            relaxed_exp_one_hot_action_encoding: bool = True,
            action_entropy_regularizer_scaling: float = 1.,
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
            entropy_regularizer_decay_rate=entropy_regularizer_decay_rate)

        self.wasserstein_regularizer_scale_factor = wasserstein_regularizer_scale_factor
        self.mixture_components = None
        self._autoencoder_optimizer = autoencoder_optimizer
        self._wasserstein_regularizer_optimizer = wasserstein_regularizer_optimizer
        self.action_discretizer = discretize_action_space
        self.relaxed_exp_one_hot_action_encoding = relaxed_exp_one_hot_action_encoding
        self.encode_action = action_encoder_network is not None
        self.action_entropy_regularizer_scaling = action_entropy_regularizer_scaling

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

        self._sample_additional_transition = False
        self.softclip = lambda x: 10. * tf.nn.tanh(x / 10.)

        if not pre_loaded_model:

            state = Input(shape=state_shape, name="state")
            action = Input(shape=action_shape, name="action")
            latent_state = Input(shape=(self.latent_state_size,), name="latent_state")
            latent_action = Input(shape=(self.number_of_discrete_actions,), name="latent_action")
            next_latent_state = Input(shape=(self.latent_state_size,), name='next_latent_state')

            # state encoder network
            self.state_encoder_network = self._initialize_state_encoder_network(
                state, state_encoder_network, state_encoder_pre_processing_network)
            # action encoder network
            if self.action_discretizer and self.encode_action:
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
            if self.action_discretizer:
                self.action_reconstruction_network = self._initialize_action_reconstruction_network(
                    latent_state, latent_action, action_decoder_network)
            else:
                self.action_reconstruction_network = None
            # steady state Lipschitz function
            self.steady_state_lipschitz_network = self._initialize_steady_state_lipschitz_function(
                latent_state, steady_state_lipschitz_network)
            # transition loss Lipschitz function
            self.transition_loss_lipschitz_network = self._initialize_transition_loss_lipschitz_function(
                state, action, latent_state, latent_action, next_latent_state, transition_loss_lipschitz_network)
            # action-successor Lipschitz function

            if debug:
                self.state_encoder_network.summary()
                if self.action_discretizer and self.encode_action:
                    self.action_encoder_network.summary()
                else:
                    print("No action encoder")
                self.transition_network.summary()
                print("latent state stationary logits")
                tf.print(self.latent_steady_state_logits, tf.shape(self.latent_steady_state_logits), summarize=-1)
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
            self.latent_steady_state_logits = steady_state_logits
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
            'steady_state_regularizer': Mean('steady_state_wasserstein_regularizer'),
            'gradient_penalty': Mean('gradient_penalty'),
            'state_encoder_entropy': Mean('state_encoder_entropy'),
            'entropy_regularizer': Mean('entropy_regularizer'),
        }
        if self.encode_action:
            self.loss_metrics['action_encoder_entropy'] = Mean('action_encoder_entropy')
        else:
            self.loss_metrics['marginal_variance'] = Mean(name='marginal_variance')

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
            activation=self.softclip,
            # activation=lambda x: 10 * tf.nn.tanh(x),
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
            activation=self.softclip,
            name='next_label_logits')(_transition_network)
        _next_latent_state_logits = Dense(
            units=self.latent_state_size - self.atomic_props_dims,
            activation=self.softclip,
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
        )(_latent_policy_network)
        return Model(
            inputs=latent_state,
            outputs=_latent_policy_network,
            name='latent_policy_network')

    def _initialize_latent_steady_state_logits(self):
        return tf.Variable(
            initial_value=tf.zeros(shape=(self.latent_state_size,), dtype=tf.float32),
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
            activation=None,
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
        action_reconstruction_network = Concatenate(name='action_reconstruction_input')([
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
            latent_state: Input,
            latent_action: Input,
            next_latent_state: Input,
            transition_loss_lipschitz_network: Model,
    ):
        _transition_loss_lipschitz_network = Concatenate()([
            state, action, latent_state, latent_action, next_latent_state])
        _transition_loss_lipschitz_network = transition_loss_lipschitz_network(_transition_loss_lipschitz_network)
        _transition_loss_lipschitz_network = Dense(
            units=1,
            activation=None,
            name='transition_loss_lipschitz_network_output'
        )(_transition_loss_lipschitz_network)

        return Model(
            inputs=[state, action, latent_state, latent_action, next_latent_state],
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
            name='action_successor_lipschitz_network_output'
        )(_action_successor_lipschitz_network)

        return Model(
            inputs=[latent_state, latent_action, next_latent_state],
            outputs=_action_successor_lipschitz_network,
            name='action_successor_lipschitz_network')

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

    def relaxed_state_encoding(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            logistic: bool = False,
            *args, **kwargs
    ) -> tfd.Distribution:
        if logistic:
            return super().relaxed_state_encoding(state, temperature, label)

        logits = self.encoder_network(state)
        if label is not None:
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        return tfd.Independent(
            tfd.RelaxedBernoulli(
                logits=logits,
                temperature=temperature,
                allow_nan_stats=False))

    def discrete_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
    ) -> tfd.Distribution:
        if self.action_discretizer:
            if self.relaxed_exp_one_hot_action_encoding:
                relaxed_distribution = self.relaxed_action_encoding(latent_state, action, 1e-5)
                log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
                return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False)
            else:
                logits = self.action_encoder_network([latent_state, action])
                return tfd.OneHotCategorical(logits=logits, allow_nan_stats=False)
        else:
            return tfd.Deterministic(loc=action)

    def relaxed_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
            temperature: Optional[Float] = 0.
    ) -> tfd.Distribution:
        logits = self.action_encoder_network([latent_state, action])
        if self.action_discretizer:
            if self.relaxed_exp_one_hot_action_encoding:
                return tfd.ExpRelaxedOneHotCategorical(
                    temperature=temperature,
                    logits=logits,
                    allow_nan_stats=False)
            else:
                return tfd.RelaxedOneHotCategorical(
                    logits=logits,
                    temperature=temperature,
                    allow_nan_stats=False)
        else:
            if self.relaxed_exp_one_hot_action_encoding:
                return tfd.Deterministic(loc=tf.math.log(action + epsilon))
            else:
                return tfd.Deterministic(loc=action)

    def decode_state(self, latent_state: tf.Tensor) -> tfd.Distribution:
        if self.time_stacked_states:
            return tfd.Independent(tfd.Deterministic(loc=self.reconstruction_network(latent_state)))
        else:
            return tfd.Deterministic(loc=self.reconstruction_network(latent_state))

    def decode_action(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
            *args, **kwargs
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return tfd.Deterministic(loc=self.action_reconstruction_network([latent_state, latent_action]))
        else:
            return tfd.Deterministic(loc=latent_action)

    def action_generator(
            self,
            latent_state: Float
    ) -> tfd.Distribution:
        def _ground_action_per_latent_action(latent_state: Float):
            return self.action_reconstruction_network(
                [tf.tile(tf.expand_dims(latent_state, 0), multiples=[self.number_of_discrete_actions, 1]),
                 tf.eye(num_rows=self.number_of_discrete_actions)])
        x = tf.map_fn(fn=_ground_action_per_latent_action, elems=latent_state)
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.discrete_latent_policy(latent_state).logits_parameter()),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=x, scale_diag=tf.ones(tf.shape(x)) * 1e-6))

    def relaxed_latent_transition(
            self,
            latent_state: Float,
            latent_action: Float,
            next_label: Optional[Float] = None,
            temperature: Float = 1e-5,
            logistic: bool = False,
            *args, **kwargs
    ) -> tfd.Distribution:
        next_label_logits, next_latent_state_logits = self.transition_network([latent_state, latent_action])

        if logistic:
            next_latent_state_distribution = tfd.Independent(
                tfd.Logistic(
                    loc=next_latent_state_logits / temperature,
                    scale=1. / temperature,
                    allow_nan_stats=False))
        else:
            next_latent_state_distribution = tfd.Independent(
                tfd.RelaxedBernoulli(
                    logits=next_latent_state_logits,
                    temperature=temperature,
                    allow_nan_stats=False))

        if next_label is not None:
            return next_latent_state_distribution
        else:
            if logistic:
                next_label_distribution = tfd.Independent(
                    tfd.Logistic(
                        loc=next_label_logits / temperature,
                        scale=1. / temperature,
                        allow_nan_stats=False))
            else:
                next_label_distribution = tfd.Independent(
                    tfd.RelaxedBernoulli(
                        logits=next_label_logits,
                        temperature=temperature,
                        allow_nan_stats=False))

            return tfd.JointDistributionSequential([
                next_label_distribution,
                next_latent_state_distribution,
            ])

    def discrete_latent_transition(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, next_label: Optional[tf.Tensor] = None
    ) -> tfd.Distribution:
        next_label_logits, next_latent_state_logits = self.transition_network([latent_state, latent_action])
        if next_label is not None:
            return tfd.Independent(
                tfd.Bernoulli(
                    logits=next_latent_state_logits,
                    allow_nan_stats=False))
        else:
            return tfd.JointDistributionSequential([
                tfd.Independent(
                    tfd.Bernoulli(
                        logits=next_label_logits,
                        allow_nan_stats=False)),
                lambda _next_label: tfd.Independent(
                    tfd.Bernoulli(
                        logits=next_latent_state_logits,
                        allow_nan_stats=False))
            ])

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
        if self.relaxed_exp_one_hot_action_encoding:
            return tfd.ExpRelaxedOneHotCategorical(
                temperature=temperature,
                logits=self.latent_policy_network(latent_state),
                allow_nan_stats=False)
        else:
            return tfd.RelaxedOneHotCategorical(
                logits=self.latent_policy_network(latent_state),
                temperature=temperature,
                allow_nan_stats=False)

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        if self.relaxed_exp_one_hot_action_encoding:
            relaxed_distribution = self.relaxed_latent_policy(latent_state, temperature=1e-5)
            log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
            return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False)
        else:
            return tfd.OneHotCategorical(logits=self.latent_policy_network(latent_state))

    def reward_distribution(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, *args, **kwargs
    ) -> tfd.Distribution:
        return tfd.Deterministic(loc=self.reward_network([latent_state, latent_action]))

    def markov_chain_reward_distribution(
            self,
            latent_state: Float
    ) -> tfd.Distribution:
        def _reward_per_action(latent_state: Float):
            return self.reward_network(
                [tf.tile(tf.expand_dims(latent_state, 0), multiples=[self.number_of_discrete_actions, 1]),
                 tf.eye(num_rows=self.number_of_discrete_actions)])
        x = tf.map_fn(fn=_reward_per_action, elems=latent_state)
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.discrete_latent_policy(latent_state).logits_parameter()),
            components_distribution=tfd.MultivariateNormalDiag(loc=x, scale_diag=tf.ones(tf.shape(x)) * 1e-6))

    def discrete_latent_steady_state_distribution(self) -> tfd.Distribution:
        return tfd.Independent(
            tfd.Bernoulli(logits=self.softclip(self.latent_steady_state_logits)))

    def relaxed_latent_steady_state_distribution(
            self,
            temperature: Float,
            logistic: bool = False,
    ) -> tfd.Distribution:
        if logistic:
            return tfd.Independent(
                tfd.Logistic(
                    loc=self.softclip(self.latent_steady_state_logits) / temperature,
                    scale=1. / temperature,
                    allow_nan_stats=False))
        else:
            return tfd.Independent(
                tfd.RelaxedBernoulli(
                    logits=self.softclip(self.latent_steady_state_logits),
                    temperature=temperature,
                    allow_nan_stats=False))

    def discrete_marginal_state_encoder_distribution(
            self,
            states: Float,
            labels: Optional[Float] = None,
            is_weights: Optional[Float] = None,
    ):
        logits = self.state_encoder_network(states)
        batch_size = tf.shape(logits)[0]

        if is_weights is None:
            mixture_distribution = tfd.Categorical(logits=tf.ones(shape=(batch_size,)))
        else:
            mixture_distribution = tfd.Categorical(
                    logits=tf.math.log(
                        tf.pow(tf.cast(batch_size, tf.float32), -1.) * is_weights + epsilon),
                    allow_nan_stats=False)

        latent_state_distribution = tfd.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            components_distribution=tfd.Independent(
                tfd.Bernoulli(
                    logits=logits,
                    allow_nan_stats=False)))

        if labels is not None:
            return tfd.JointDistributionSequential([
                tfd.MixtureSameFamily(
                    mixture_distribution=mixture_distribution,
                    components_distribution=tfd.Independent(
                        tfd.Bernoulli(
                            logits=(labels * 2. - 1) * 1e2,
                            allow_nan_stats=False)
                    ),
                ),
                latent_state_distribution
            ])
        else:
            return latent_state_distribution

    def relaxed_marginal_action_encoder_distribution(
            self,
            latent_states: Float,
            actions: Float,
            temperature: Float = 1e-5,
            is_weights: Optional[Float] = None,
    ) -> tfd.Distribution:
        if self.relaxed_exp_one_hot_action_encoding:
            logits = tf.math.log(
                self.relaxed_action_encoding(
                    latent_states, actions, temperature=temperature
                ).probs_parameter() + epsilon)
        else:
            logits = self.action_encoder_network([latent_states, actions])

        batch_size = tf.shape(logits)[0]

        if debug:
            tf.print('relaxed marginal actions logits:', logits, summarize=-1)

        if is_weights is None:
            mixture_distribution = tfd.Categorical(logits=tf.ones(shape=(batch_size,)), allow_nan_stats=False)
        else:
            mixture_distribution = tfd.Categorical(
                logits=tf.math.log(
                    tf.pow(tf.cast(batch_size, tf.float32), -1.) * is_weights + epsilon),
                allow_nan_stats=False)

        return tfd.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            components_distribution=tfd.RelaxedOneHotCategorical(
                logits=logits,
                temperature=temperature,
                allow_nan_stats=False),
            allow_nan_stats=False)

    def relaxed_marginal_state_encoder_distribution(
            self,
            states: Float,
            labels: Optional[Float] = None,
            temperature: Float = 1e-5,
            reparameterize: bool = True,
            logistic: bool = True,
            is_weights: Optional[Float] = None,
    ) -> tfd.Distribution:
        logits = self.state_encoder_network(states)
        batch_size = tf.shape(logits)[0]
        reparameterize = reparameterize and labels is None

        if debug:
            tf.print('relaxed marginal logits:', logits, summarize=-1)

        if is_weights is None:
            mixture_distribution = tfd.Categorical(logits=tf.ones(shape=(batch_size,)), allow_nan_stats=False)
        else:
            mixture_distribution = tfd.Categorical(
                logits=tf.math.log(tf.pow(tf.cast(batch_size, tf.float32), -1.) * is_weights),
                allow_nan_stats=False)

        if logistic:
            latent_state_distribution = tfd.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                components_distribution=tfd.Independent(
                    tfd.Logistic(
                        loc=logits / temperature,
                        scale=1. / temperature,
                        allow_nan_stats=False)
                ),
                reparameterize=reparameterize,
                allow_nan_stats=False)
        else:
            latent_state_distribution = tfd.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                components_distribution=tfd.Independent(
                    tfd.RelaxedBernoulli(
                        logits=logits,
                        temperature=temperature,
                        validate_args=False,
                        allow_nan_stats=False)
                ),
                reparameterize=reparameterize,
                allow_nan_stats=False)

        if labels is not None:
            return tfd.JointDistributionSequential([
                tfd.MixtureSameFamily(
                    mixture_distribution=mixture_distribution,
                    components_distribution=tfd.Independent(
                        tfd.Bernoulli(
                            logits=(labels * 2. - 1) * 1e2,
                            allow_nan_stats=False)
                    ),
                    allow_nan_stats=False
                ), latent_state_distribution
            ])
        else:
            return latent_state_distribution

    def action_embedding_function(
            self,
            state: tf.Tensor,
            latent_action: tf.Tensor,
            label: Optional[tf.Tensor] = None,
            labeling_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None
    ) -> tf.Tensor:
        if (label is None) == (labeling_function is None):
            raise ValueError("Must either pass a label or a labeling_function")

        if labeling_function is not None:
            label = labeling_function(state)

        if self.action_discretizer:
            return self.decode_action(
                latent_state=tf.cast(self.state_embedding_function(state, label=label), dtype=tf.float32),
                latent_action=tf.cast(
                    tf.one_hot(
                        latent_action,
                        depth=self.number_of_discrete_actions),
                    dtype=tf.float32),
            ).mode()
        else:
            return latent_action

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
            additional_transition_batch: Optional[Tuple[Float, ...]] = None,
            *args, **kwargs
    ):
        batch_size = tf.shape(state)[0]
        # encoder sampling
        latent_state = tf.concat([
            label,
            tfd.TransformedDistribution(
                distribution=self.relaxed_state_encoding(state, self.state_encoder_temperature, logistic=True),
                bijector=tfb.Sigmoid(),
            ).sample(),
        ], axis=-1)
        next_latent_state = tf.concat([
            next_label,
            tfd.TransformedDistribution(
                distribution=self.relaxed_state_encoding(next_state, self.state_encoder_temperature, logistic=True),
                bijector=tfb.Sigmoid(),
            ).sample()
        ], axis=-1)

        if self.encode_action:
            latent_action = tfd.TransformedDistribution(
                distribution=self.relaxed_action_encoding(
                    latent_state,
                    action,
                    temperature=self.action_encoder_temperature),
                bijector=(tfb.Exp() if self.relaxed_exp_one_hot_action_encoding else
                          tfb.Identity())
            ).sample()
        else:
            latent_action = tfd.TransformedDistribution(
                distribution=self.relaxed_latent_policy(
                    latent_state,
                    temperature=self.latent_policy_temperature),
                bijector=(tfb.Exp() if self.relaxed_exp_one_hot_action_encoding else
                          tfb.Identity())
            ).sample()

        # latent steady-state distribution
        (stationary_latent_state,
         stationary_latent_action,
         next_stationary_latent_state) = tfd.JointDistributionSequential([
            tfd.TransformedDistribution(
                distribution=self.relaxed_latent_steady_state_distribution(
                    temperature=self.encoder_temperature,
                    logistic=True),
                bijector=tfb.Sigmoid()),
            lambda _latent_state: tfd.TransformedDistribution(
                distribution=self.relaxed_latent_policy(
                    latent_state=_latent_state,
                    temperature=self.latent_policy_temperature),
                bijector=(tfb.Exp() if self.relaxed_exp_one_hot_action_encoding else
                          tfb.Identity())),
            lambda _latent_action, _latent_state: self.relaxed_latent_transition(
                    _latent_state,
                    _latent_action,
                    temperature=self.state_prior_temperature,
                    logistic=True),
        ]).sample(sample_shape=(batch_size,))
        # next_stationary latent state is a logistic sample of the form
        # (<batch_size, label>,<batch_size, next_latent_state_wo_label>)
        next_stationary_latent_state = tf.nn.sigmoid(tf.concat(next_stationary_latent_state, axis=-1))

        # next latent state from the latent transition function
        next_transition_latent_state = tf.sigmoid(  # we generate logistic samples instead of relaxed Bernoulli
            tf.concat(
                self.relaxed_latent_transition(
                    latent_state,
                    latent_action,
                    temperature=self.state_prior_temperature,
                    logistic=True,
                ).sample(),
                axis=-1)
        )  # the sample is of the form (<batch_size, label>,<batch_size, next_latent_state_wo_label>)

        # reconstruction loss
        # the reward as well as the state and action reconstruction functions are deterministic
        _action, _reward, _next_state = tfd.JointDistributionSequential([
            self.decode_action(
                latent_state,
                latent_action) if self.encode_action else
            tfd.Deterministic(loc=self.action_generator(latent_state).mean()),
            self.reward_distribution(
                latent_state,
                latent_action) if self.encode_action else
            tfd.Deterministic(loc=self.markov_chain_reward_distribution(latent_state).mean()),
            self.decode_state(next_latent_state)
        ]).sample()

        reconstruction_loss = (
            tf.norm(action - _action, ord=1, axis=1) +
            tf.norm(reward - _reward, ord=1, axis=1) +  # local reward loss
            tf.norm(next_state - _next_state, ord=1, axis=1))

        if not self.encode_action:
            reconstruction_loss = reconstruction_loss ** 2

            # marginal variance of the reconstruction
            random_action, random_reward = tfd.JointDistributionSequential([
                self.decode_action(latent_state, latent_action),
                self.reward_distribution(latent_state, latent_action),
            ]).sample()
            y = tf.concat([random_action, random_reward, _next_state], axis=-1)
            mean = tf.concat([_action, _reward, _next_state], axis=-1)
            marginal_variance = tf.reduce_sum((y - mean) ** 2. + (mean - tf.reduce_mean(mean)) ** 2., axis=-1)

        else:
            random_action = _action
            random_reward = _reward
            marginal_variance = 0.

        # Wasserstein regularizers
        steady_state_regularizer = tf.squeeze(
            self.steady_state_lipschitz_network(next_transition_latent_state) -
            self.steady_state_lipschitz_network(next_stationary_latent_state))
        transition_loss_regularizer = tf.squeeze(
            self.transition_loss_lipschitz_network(
                [state, action, latent_state, latent_action, next_latent_state]) -
            self.transition_loss_lipschitz_network(
                [state, action, latent_state, latent_action, next_transition_latent_state]))

        # Lipschitz constraints
        steady_state_gradient_penalty = self.compute_gradient_penalty(
            x=next_transition_latent_state,
            y=next_stationary_latent_state,
            lipschitz_function=self.steady_state_lipschitz_network)
        transition_loss_gradient_penalty = self.compute_gradient_penalty(
            x=next_latent_state,
            y=next_transition_latent_state,
            lipschitz_function=lambda _x: self.transition_loss_lipschitz_network(
                [state, action, latent_state, latent_action, _x]))
        
        # entropy_regularizer = 0.
        entropy_regularizer = self.entropy_regularizer(
            state=state,
            use_marginal_encoder_entropy=True,
            action=action if self.encode_action else None,
            sample_probability=sample_probability,
            latent_states=latent_state,
            latent_actions=latent_action,
            discrete_distribution=False,
            logistic=False)

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(reconstruction_loss +
                                          marginal_variance +
                                          steady_state_regularizer +
                                          transition_loss_regularizer)))

        # loss metrics
        self.loss_metrics['reconstruction_loss'](reconstruction_loss)
        self.loss_metrics['state_mse'](next_state, _next_state)
        self.loss_metrics['action_mse'](action, random_action)
        self.loss_metrics['reward_mse'](reward, random_reward)
        self.loss_metrics['transition_loss'](transition_loss_regularizer)
        self.loss_metrics['steady_state_regularizer'](steady_state_regularizer)
        self.loss_metrics['gradient_penalty'](
            steady_state_gradient_penalty + transition_loss_gradient_penalty)
        self.loss_metrics['state_encoder_entropy'](self.binary_encode_state(state).entropy())
        if self.encode_action:
            self.loss_metrics['action_encoder_entropy'](
                    self.discrete_action_encoding(latent_state, action).entropy())
        else:
            self.loss_metrics['marginal_variance'](marginal_variance)
        self.loss_metrics['entropy_regularizer'](entropy_regularizer)

        if debug:
            tf.print("latent_state", latent_state, summarize=-1)
            tf.print("next_latent_state", next_latent_state, summarize=-1)
            tf.print("next_stationary_latent_state", next_stationary_latent_state, summarize=-1)
            tf.print("next_transition_latent_state", next_transition_latent_state, summarize=-1)
            tf.print("latent_action", latent_action, summarize=-1)
            tf.print("loss", tf.stop_gradient(reconstruction_loss +
                                              marginal_variance +
                                              steady_state_regularizer +
                                              transition_loss_regularizer))

        return {
            'reconstruction_loss': reconstruction_loss + marginal_variance,
            'steady_state_regularizer': steady_state_regularizer,
            'steady_state_gradient_penalty': steady_state_gradient_penalty,
            'transition_loss_regularizer': transition_loss_regularizer,
            'transition_loss_gradient_penalty': transition_loss_gradient_penalty,
            'entropy_regularizer': -1. * entropy_regularizer if self.entropy_regularizer_scale_factor > epsilon else 0.,
        }

    @tf.function
    def entropy_regularizer(
            self,
            state: tf.Tensor,
            use_marginal_encoder_entropy: bool = True,
            action: Optional[Float] = None,
            latent_states: Optional[Float] = None,
            latent_actions: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            logistic: bool = False,
            discrete: bool = False,
            *args, **kwargs
    ):
        if sample_probability is None or not use_marginal_encoder_entropy:
            is_weights = None
        else:
            is_weights = tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability

        if logistic:
            latent_state = super().relaxed_state_encoding(state, self.state_encoder_temperature).sample()
        elif latent_states is None:
            latent_state = self.relaxed_state_encoding(state, self.state_encoder_temperature).sample()
        else:
            latent_state = latent_states[:, self.atomic_props_dims:, ...]

        if not use_marginal_encoder_entropy:
            regularizer = tf.reduce_mean(-1. * self.discrete_latent_steady_state_distribution().entropy())
        else:
            if discrete:
                marginal_state_encoder_distribution = self.discrete_marginal_state_encoder_distribution(
                    states=state,
                    is_weights=is_weights)
            else:
                marginal_state_encoder_distribution = self.relaxed_marginal_state_encoder_distribution(
                    states=state,
                    temperature=self.encoder_temperature,
                    reparameterize=False,
                    logistic=logistic,
                    is_weights=is_weights)

            clip = lambda x: tf.clip_by_value(
                x,
                clip_value_min=-10. if logistic else 1e-6,
                clip_value_max=10. if logistic else 1. - 1e-6)

            if debug:
                tf.print("clipped_latent_state: ", clip(latent_state), summarize=-1)
                tf.print("regularizer:", -1. * tf.math.log(marginal_state_encoder_distribution.prob(clip(latent_state))),
                         summarize=-1)

            regularizer = tf.reduce_mean(marginal_state_encoder_distribution.log_prob(clip(latent_state)))

        if action is not None and latent_actions is None:
            if self.encode_action:
                latent_action = tfd.TransformedDistribution(
                    distribution=self.relaxed_action_encoding(
                        latent_state, action, temperature=self.action_encoder_temperature),
                    bijector=tfb.Exp() if self.relaxed_exp_one_hot_action_encoding else tfb.Identity()
                ).sample()
            else:
                latent_action = tfd.TransformedDistribution(
                    distribution=self.relaxed_latent_policy(
                        latent_state, temperature=self.latent_policy_temperature),
                    bijector=tfb.Exp() if self.relaxed_exp_one_hot_action_encoding else tfb.Identity()
                ).sample()
        else:
            latent_action = latent_actions

        if latent_states is not None:
            if action is not None and use_marginal_encoder_entropy:
                marginal_action_encoder_distribution = self.relaxed_marginal_action_encoder_distribution(
                    latent_states=latent_states,
                    actions=action,
                    temperature=self.action_encoder_temperature,
                    is_weights=is_weights)
                regularizer += (self.action_entropy_regularizer_scaling *
                                tf.reduce_mean(marginal_action_encoder_distribution.log_prob(latent_action)))
            else:
                regularizer += (-1. * self.action_entropy_regularizer_scaling *
                                self.discrete_latent_policy(latent_states).entropy())

        return regularizer

    @tf.function
    def compute_gradient_penalty(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            lipschitz_function: Callable[[tf.Tensor], tf.Tensor],
    ):
        noise = tf.random.uniform(shape=(tf.shape(x)[0], 1), minval=0., maxval=1.)
        straight_lines = noise * x + (1. - noise) * y
        gradients = tf.gradients(lipschitz_function(straight_lines), straight_lines)[0]
        return tf.square(tf.norm(gradients, ord=2, axis=1) - 1.)

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
        latent_state_distribution = self.binary_encode_state(state)
        latent_state = tf.concat([
            label, tf.cast(latent_state_distribution.sample(), tf.float32)
        ], axis=-1)
        next_latent_state = tf.concat([
            next_label, tf.cast(self.binary_encode_state(next_state).sample(), tf.float32)
        ], axis=-1)
        if self.encode_action:
            latent_action = self.discrete_action_encoding(latent_state, action).sample()
        else:
            latent_action = tf.cast(self.discrete_latent_policy(latent_state).sample(), tf.float32)

        # latent steady-state distribution
        (stationary_latent_state,
         stationary_latent_action,
         next_stationary_latent_state) = tfd.JointDistributionSequential([
            self.discrete_latent_steady_state_distribution(),
            lambda _latent_state: self.discrete_latent_policy(_latent_state),
            lambda _latent_action, _latent_state: self.discrete_latent_transition(
                _latent_state,
                _latent_action),
        ]).sample(sample_shape=(batch_size,))
        next_stationary_latent_state = tf.concat(next_stationary_latent_state, axis=-1)
        next_stationary_latent_state = tf.cast(next_stationary_latent_state, tf.float32)

        # next latent state from the latent transition function
        next_transition_latent_state = tf.concat(
            self.discrete_latent_transition(
                latent_state,
                latent_action,
            ).sample(),
            axis=-1)

        # reconstruction loss
        # the reward as well as the state and action reconstruction functions are deterministic
        _action, _reward, _next_state = tfd.JointDistributionSequential([
            self.decode_action(
                latent_state,
                latent_action) if self.encode_action else
            tfd.Deterministic(loc=self.action_generator(latent_state).mean()),
            self.reward_distribution(
                latent_state,
                latent_action) if self.encode_action else
            tfd.Deterministic(loc=self.markov_chain_reward_distribution(latent_state).mean()),
            self.decode_state(next_latent_state)
        ]).sample()

        reconstruction_loss = (
            tf.norm(action - _action, ord=1, axis=1) +
            tf.norm(reward - _reward, ord=1, axis=1) +  # local reward loss
            tf.norm(next_state - _next_state, ord=1, axis=1))

        # marginal variance of the reconstruction
        if self.encode_action:
            marginal_variance = 0.
        else:
            reconstruction_loss = reconstruction_loss ** 2.
            random_action, random_reward = tfd.JointDistributionSequential([
                self.decode_action(latent_state, latent_action),
                self.reward_distribution(latent_state, latent_action),
            ]).sample()
            y = tf.concat([random_action, random_reward, _next_state], axis=-1)
            mean = tf.concat([_action, _reward, _next_state], axis=-1)
            marginal_variance = tf.reduce_sum((y - mean) ** 2. + (mean - tf.reduce_mean(mean)) ** 2., axis=-1)

        # Wasserstein regularizers
        steady_state_regularizer = tf.squeeze(
            self.steady_state_lipschitz_network(next_transition_latent_state) -
            self.steady_state_lipschitz_network(next_stationary_latent_state))
        transition_loss_regularizer = tf.squeeze(
            self.transition_loss_lipschitz_network(
                [state, action, latent_state, latent_action, next_latent_state]) -
            self.transition_loss_lipschitz_network(
                [state, action, latent_state, latent_action, next_transition_latent_state]))

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
            'latent_actions': tf.cast(tf.argmax(latent_action, axis=1), tf.int64)
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
                self.wasserstein_regularizer_scale_factor.stationary.gradient_penalty_multiplier *
                output['steady_state_gradient_penalty'] +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.gradient_penalty_multiplier *
                output['transition_loss_gradient_penalty']
        )
        entropy_regularizer = self.entropy_regularizer_scale_factor * output['entropy_regularizer']

        loss = lambda minimize: tf.reduce_mean(
            (-1.) ** (1. - minimize) * is_weights * (
                    minimize * reconstruction_loss +
                    wasserstein_loss +
                    (minimize - 1.) * gradient_penalty +
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
        if self.action_discretizer and self.encode_action:
            return self.state_encoder_network.trainable_variables + self.action_encoder_network.trainable_variables
        else:
            return self.state_encoder_network.trainable_variables

    @property
    def generator_variables(self):
        variables = [self.latent_steady_state_logits]
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
                self.transition_loss_lipschitz_network.trainable_variables)  # +

    def _compute_apply_gradients(
            self, state, label, action, reward, next_state, next_label,
            autoencoder_variables=None, wasserstein_regularizer_variables=None,
            sample_key=None, sample_probability=None,
            additional_transition_batch=None,
            *args, **kwargs
    ):
        if autoencoder_variables is None and wasserstein_regularizer_variables is None:
            raise ValueError("Must pass autoencoder and/or wasserstein regularizer variables")
        with tf.GradientTape(persistent=True) as tape:
            loss = self.compute_loss(
                state, label, action, reward, next_state, next_label,
                sample_key=sample_key, sample_probability=sample_probability,
                additional_transition_batch=additional_transition_batch)

        for optimization_direction, variables in {
            'min': autoencoder_variables, 'max': wasserstein_regularizer_variables
        }.items():
            if variables is not None:
                gradients = tape.gradient(loss[optimization_direction], variables)
                optimizer = {
                    'min': self._autoencoder_optimizer,
                    'max': self._wasserstein_regularizer_optimizer
                }[optimization_direction]
                optimizer.apply_gradients(zip(gradients, variables))

                if debug_gradients:
                    for gradient, variable in zip(gradients, variables):
                        tf.print("Gradient for {} (direction={}):".format(variable.name, optimization_direction), gradient)

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
            additional_transition_batch: Optional[Tuple[Float]] = None
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            autoencoder_variables=self.inference_variables + self.generator_variables,
            wasserstein_regularizer_variables=self.wasserstein_variables,
            sample_key=sample_key, sample_probability=sample_probability,
            additional_transition_batch=additional_transition_batch)

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
        latent_state = tf.cast(self.binary_encode_state(state, label).sample(), tf.float32)
        mean = tf.reduce_mean(
                self.discrete_action_encoding(latent_state, action).probs_parameter() if self.encode_action else
                self.discrete_latent_policy(latent_state).probs_parameter(),
                axis=0)
        check = lambda x: 1 if 1 - eps > x > eps else 0
        mean_bits_used = tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

        mbu = {'mean_action_bits_used': mean_bits_used}
        mbu.update(super().mean_latent_bits_used(inputs, eps, deterministic))
        return mbu

    def wrap_tf_environment(
            self,
            tf_env: tf_environment.TFEnvironment,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            *args,
            **kwargs
    ) -> tf_environment.TFEnvironment:
        return variational_action_discretizer.VariationalTFEnvironmentDiscretizer(
            variational_action_discretizer=self,
            tf_env=tf_env,
            labeling_function=labeling_function)

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
        _labeling_function = dataset_generator.ergodic_batched_labeling_function(labeling_function)

        return estimate_local_losses_from_samples(
            environment=environment,
            latent_policy=self.get_latent_policy(),
            steps=steps,
            number_of_discrete_actions=self.number_of_discrete_actions,
            state_embedding_function=self.state_embedding_function,
            action_embedding_function=lambda state, latent_action: self.action_embedding_function(
                state, latent_action, labeling_function=_labeling_function),
            latent_reward_function=lambda latent_state, latent_action, _: (
                self.reward_distribution(
                    latent_state=tf.cast(latent_state, dtype=tf.float32),
                    latent_action=tf.cast(latent_action, dtype=tf.float32)).mode()),
            labeling_function=labeling_function,
            latent_transition_function=lambda latent_state, latent_action: self.discrete_latent_transition(
                latent_state=tf.cast(latent_state, tf.float32),
                latent_action=tf.cast(latent_action, tf.float32)),
            estimate_transition_function_from_samples=estimate_transition_function_from_samples,
            replay_buffer_max_frames=replay_buffer_max_frames,
            reward_scaling=reward_scaling)

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
                    if (local_losses_metrics.local_transition_loss_transition_function_estimation is not None and
                            local_losses_metrics.local_transition_loss_transition_function_estimation <
                            local_losses_metrics.local_transition_loss):
                        local_transition_loss = \
                            local_losses_metrics.local_transition_loss_transition_function_estimation
                        local_transition_loss_time = local_losses_metrics.time_metrics[
                            'local_transition_loss_transition_function_estimation']
                    else:
                        local_transition_loss = local_losses_metrics.local_transition_loss
                        local_transition_loss_time = local_losses_metrics.time_metrics['local_transition_loss']
                    tf.summary.scalar('local_transition_loss', local_transition_loss, step=global_step)
                    tf.summary.scalar(
                        'local_losses_computation_time',
                        local_losses_metrics.time_metrics['local_reward_loss'] + local_transition_loss_time,
                        step=global_step)
                    tf.print('Local reward loss: {:.2f}'.format(local_losses_metrics.local_reward_loss))
                    tf.print('Local transition loss: {:.2f}'.format(local_losses_metrics.local_transition_loss))
                    tf.print('Local transition loss (empirical transition function): {:.2f}'
                             ''.format(local_losses_metrics.local_transition_loss_transition_function_estimation))

            print('eval loss: ', metrics['eval_loss'].result().numpy())

        if eval_policy_driver is not None or eval_steps > 0:
            self.assign_score(
                score=avg_rewards if eval_policy_driver is not None else metrics['eval_elbo'].result(),
                checkpoint_model=save_directory is not None and log_name is not None,
                save_directory=save_directory,
                model_name=log_name,
                training_step=global_step.numpy())

        gc.collect()

        return metrics['eval_loss'].result()
