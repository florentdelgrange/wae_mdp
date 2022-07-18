from typing import Optional, Text, Callable, cast

import tensorflow as tf
from tensorflow.python.framework.indexed_slices import tensor_spec

from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import network
from tf_agents.typing import types
from tf_agents.typing.types import Float
from tf_agents.utils import common, eager_utils

from layers.encoders import TFAgentEncodingNetworkWrapper
from wasserstein_mdp import WassersteinMarkovDecisionProcess


class WaeDqnAgent(dqn_agent.DqnAgent):

    def __init__(
            self,
            time_step_spec: ts.TimeStep,
            latent_time_step_spec: ts.TimeStep,
            action_spec: types.NestedTensorSpec,
            q_network: network.Network,
            optimizer: types.Optimizer,
            wae_mdp: WassersteinMarkovDecisionProcess,
            labeling_fn: Callable[[Float], Float],
            observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
            epsilon_greedy: Optional[types.FloatOrReturningFloat] = 0.1,
            n_step_update: int = 1,
            boltzmann_temperature: Optional[types.FloatOrReturningFloat] = None,
            emit_log_probability: bool = False,
            target_q_network: Optional[network.Network] = None,
            target_update_tau: types.Float = 1.0,
            target_update_period: int = 1,
            td_errors_loss_fn: Optional[types.LossFn] = None,
            gamma: types.Float = 1.0,
            reward_scale_factor: types.Float = 1.0,
            gradient_clipping: Optional[types.Float] = None,
            debug_summaries: bool = False,
            summarize_grads_and_vars: bool = False,
            train_step_counter: Optional[tf.Variable] = None,
            name: Optional[Text] = None,
    ):
        super().__init__(latent_time_step_spec, action_spec, q_network, optimizer,
                         observation_and_action_constraint_splitter,
                         epsilon_greedy, n_step_update, boltzmann_temperature, emit_log_probability, target_q_network,
                         target_update_tau, target_update_period, td_errors_loss_fn, gamma, reward_scale_factor,
                         gradient_clipping, debug_summaries, summarize_grads_and_vars, train_step_counter, name)

        self._state_embedding = TFAgentEncodingNetworkWrapper(
            wae_mdp.state_encoder_network,
            wae_mdp.state_encoder_temperature,
            name='CriticStateEmbedding')
        self._state_embedding.create_variables(time_step_spec.observation)
        self._target_state_embedding = common.maybe_copy_target_network_with_checks(
            self._state_embedding, None, input_spec=time_step_spec.observation,
            name='TargetStateEmbedding')
        self._labeling_fn = labeling_fn

    def _initialize(self):
        super(WaeDqnAgent, self)._initialize()
        common.soft_variables_update(
            self._state_embedding.variables, self._target_state_embedding.variables, tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        with tf.name_scope('update_targets'):
            def update():
                return common.soft_variables_update(
                    self._q_network.variables + self._state_embedding.variables,
                    self._target_q_network.variables + self._target_state_embedding.variables,
                    tau,
                    tau_non_trainable=1.0)

            return common.Periodically(update, period, 'periodic_update_targets')

    def _train(self, experience, weights):
        # copy pasta from DQN TFAgent
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = self._q_network.trainable_weights + \
                             self._state_embedding.trainable_weights  # changes here
        non_trainable_weights = self._q_network.non_trainable_weights + \
                                self._state_embedding.non_trainable_weights  # changes here
        assert list(variables_to_train), "No variables in the agent's q_network."
        grads = tape.gradient(loss_info.loss, variables_to_train)
        grads_and_vars = list(zip(grads, variables_to_train))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)

        if self._summarize_grads_and_vars:
            grads_and_vars_with_non_trainable = (
                    grads_and_vars + [(None, v) for v in non_trainable_weights])
            eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)
        self._optimizer.apply_gradients(grads_and_vars)
        self.train_step_counter.assign_add(1)

        self._update_target()

        return loss_info

    def _compute_q_values(self, time_steps, actions, training=False):
        # copy pasta from TFAgent DQN
        network_observation = time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        # changes here
        embedded_observation, _ = self._state_embedding(
            [network_observation, self._labeling_fn(network_observation)],
            training=training)

        q_values, _ = self._q_network(embedded_observation,
                                      step_type=time_steps.step_type,
                                      training=training)
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        action_spec = cast(tensor_spec.BoundedTensorSpec, self._action_spec)
        multi_dim_actions = action_spec.shape.rank > 0
        return common.index_with_actions(
            q_values,
            tf.cast(actions, dtype=tf.int32),
            multi_dim_actions=multi_dim_actions)

    def _compute_next_q_values(self, next_time_steps, info):
        # copy pasta from TFAgent DQN
        del info
        network_observation = next_time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        # changes here
        embedded_observation, _ = self._target_state_embedding(
            [network_observation, self._labeling_fn(network_observation)],
            training=False)

        next_target_q_values, _ = self._target_q_network(
            network_observation, step_type=next_time_steps.step_type)
        batch_size = (
                next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
        dummy_state = self._policy.get_initial_state(batch_size)
        # Find the greedy actions using our greedy policy. This ensures that action
        # constraints are respected and helps centralize the greedy logic.
        best_next_actions = self._policy.action(next_time_steps, dummy_state).action

        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.rank > 0
        return common.index_with_actions(
            next_target_q_values,
            best_next_actions,
            multi_dim_actions=multi_dim_actions)
