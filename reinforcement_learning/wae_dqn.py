import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')


import random

import numpy as np
from tf_agents import specs
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.typing import types
from tf_agents.typing.types import Int, Float
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import Progbar

import tf_agents
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment, TimeLimit
from tf_agents.metrics import tf_metrics, tf_metric, py_metrics
from tf_agents.networks import q_network, categorical_q_network
from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer, reverb_replay_buffer, reverb_utils
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.trajectory import experience_to_transitions
from tf_agents.utils import common
from tf_agents.policies import policy_saver, categorical_q_policy, boltzmann_policy, q_policy, py_tf_eager_policy, \
    greedy_policy, tf_policy
import tf_agents.trajectories.time_step as ts
from reinforcement_learning.environments import EnvironmentLoader

import reinforcement_learning
import wasserstein_mdp
from layers.encoders import EncodingType
from policies.latent_policy import LatentPolicyOverRealStateSpace
from policies.one_hot_categorical import OneHotTFPolicyWrapper
from reinforcement_learning.agents.wae_agent import WaeDqnAgent
from reinforcement_learning.environments.latent_environment import LatentEmbeddingTFEnvironmentWrapper
from reinforcement_learning.environments.perturbed_env import PerturbedEnvironment
from util.io.dataset_generator import map_rl_trajectory_to_vae_input, ergodic_batched_labeling_function
from util.nn import ModelArchitecture
from policies.saved_policy import SavedTFPolicy
from wasserstein_mdp import WassersteinMarkovDecisionProcess, WassersteinRegularizerScaleFactor
from flags import FLAGS

from typing import Tuple, Callable, Optional, List
import functools
import threading
import datetime

try:
    import reverb
except ImportError as ie:
    print(ie, "Reverb is not installed on your system, "
              "meaning prioritized experience replay cannot be used.")

from absl import app

default_architecture = ModelArchitecture(hidden_units=(256, 256), activation='relu')


class WaeDqnLearner:
    def __init__(
            self,
            env_name: str,
            env_suite,
            labeling_fn: Callable[[Float], Float],
            latent_state_size: int,
            num_iterations: int = int(2e5),
            initial_collect_steps: int = int(1e4),
            collect_steps_per_iteration: int = 1,
            replay_buffer_capacity: int = int(1e6),
            network_fc_layer_params: ModelArchitecture = default_architecture,
            state_encoder_temperature: Float = 2. / 3,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=20.,
                global_gradient_penalty_multiplier=10.),
            gamma: float = 0.99,
            minimizer_learning_rate: float = 3e-4,
            maximizer_learning_rate: float = 3e-4,
            encoder_learning_rate: Optional[float] = None,
            dqn_learning_rate: float = 3e-4,
            log_interval: int = 200,
            num_eval_episodes: int = 30,
            eval_interval: int = int(1e4),
            num_parallel_environments: int = 4,
            wae_batch_size: int = 128,
            dqn_batch_size: int = 64,
            n_wae_critic: Int = 5,
            n_wae_updates: Int = 5,
            save_directory_location: str = '.',
            prioritized_experience_replay: bool = False,
            priority_exponent: float = 0.6,
            wae_eval_steps: Int = int(1e4),
            seed: Optional[int] = 42,
            epsilon_greedy: Optional[types.FloatOrReturningFloat] = 0.1,
            boltzmann_temperature: Optional[types.FloatOrReturningFloat] = None,
            target_update_period: int = 20,
            target_update_tau: types.Float = 1.0,
            reward_scale_factor: types.Float = 1.0,
            gradient_clipping: Optional[types.Float] = None,
            env_time_limit: Optional[Int] = None,
            env_perturbation: Optional[Float] = .75,
            summarize_grads_and_vars: bool = False,
    ):
        self.parallel_envs = num_parallel_environments > 1 and not prioritized_experience_replay

        if collect_steps_per_iteration is None:
            collect_steps_per_iteration = dqn_batch_size
        if self.parallel_envs:
            replay_buffer_capacity = replay_buffer_capacity // num_parallel_environments
            collect_steps_per_iteration = max(1, collect_steps_per_iteration // num_parallel_environments)

        self.env_name = env_name
        self.env_suite = env_suite
        self.num_iterations = num_iterations
        self.initial_collect_steps = initial_collect_steps
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_capacity = replay_buffer_capacity
        self.gamma = gamma
        self.log_interval = log_interval
        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval
        self.wae_eval_steps = wae_eval_steps
        self.num_parallel_environments = num_parallel_environments
        self.dqn_batch_size = dqn_batch_size
        self.wae_batch_size = wae_batch_size
        self.prioritized_experience_replay = prioritized_experience_replay
        self.n_wae_critic = n_wae_critic
        self.n_wae_updates = n_wae_updates
        self.labeling_fn = labeling_fn

        # set the wae network components to the same architecture
        state_encoder_network = transition_network = reward_network = decoder_network = \
            steady_state_lipschitz_network = transition_loss_lipschitz_network = network_fc_layer_params

        # set up the environment loader
        env_loader = EnvironmentLoader(env_suite, seed=seed)
        env_wrappers = []
        if env_time_limit is not None:
            env_wrappers.append(
                lambda env: TimeLimit(env, env_time_limit))
        # recursive perturbation trick to enforce ergodicity
        if env_perturbation > 0.:
            env_wrappers.append(
                lambda env: PerturbedEnvironment(
                    env,
                    perturbation=env_perturbation,
                    recursive_perturbation=True))

        # load the environment
        if self.parallel_envs:
            self.tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_loader.load(env_name, env_wrappers)] * num_parallel_environments))
            _obs = self.tf_env.reset().observation
            self.py_env = env_suite.load(env_name)
            self.py_env.reset()
            # self.eval_env = tf_py_environment.TFPyEnvironment(self.py_env)
        else:
            self.py_env = env_loader.load(env_name, env_wrappers)
            self.py_env.reset()
            self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
            _obs = self.tf_env.reset().observation
            # self.eval_env = tf_py_environment.TFPyEnvironment(env_suite.load(env_name))

        self.observation_spec = self.tf_env.observation_spec()
        self.latent_observation_spec = specs.BoundedTensorSpec(
            shape=(latent_state_size,),
            dtype=tf.float32,
            minimum=0.,
            maximum=1.,
            name='latent_state')
        self.action_spec = self.tf_env.action_spec()

        self.q_network = q_network.QNetwork(
            # Q-network inputs are latent states
            self.latent_observation_spec,
            self.tf_env.action_spec(),
            fc_layer_params=network_fc_layer_params.hidden_units)
        # Q-policy is a Categorical distribution policy with logits inferred based on the Q-network
        policy = q_policy.QPolicy(
            time_step_spec=ts.time_step_spec(self.latent_observation_spec),
            action_spec=self.tf_env.action_spec(),
            q_network=self.q_network, )

        # policy that can be fed as input of the WAE-MDP
        wae_policy = greedy_policy.GreedyPolicy(
            OneHotTFPolicyWrapper(
                policy,
                time_step_spec=policy.time_step_spec,
                action_spec=policy.action_spec))

        # WAE-MDP optimizers
        wae_mdp_minimizer = tf.keras.optimizers.Adam(learning_rate=minimizer_learning_rate)
        wae_mdp_maximizer = tf.keras.optimizers.Adam(learning_rate=maximizer_learning_rate)
        if encoder_learning_rate is not None:
            encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_learning_rate)
        else:
            encoder_optimizer = None
        # DQN optimizer
        dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=dqn_learning_rate)

        self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name="global_step")
        self.dqn_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name="dqn_step")
        self.wae_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name="wae_step")

        # initialize WAE-MDP
        state_shape, reward_shape = (
            shape if shape != () else (1,) for shape in [
                self.tf_env.observation_spec().shape,
                self.tf_env.time_step_spec().reward.shape,])
        self.wae_mdp = WassersteinMarkovDecisionProcess(
            state_shape=state_shape,
            action_shape=(self.tf_env.action_spec().maximum + 1,),
            reward_shape=reward_shape,
            label_shape=labeling_fn(_obs).shape[1:],
            discretize_action_space=False,
            state_encoder_network=state_encoder_network,
            state_encoder_temperature=state_encoder_temperature,
            latent_policy_network=None,
            action_decoder_network=None,
            latent_state_size=latent_state_size,
            transition_network=transition_network,
            reward_network=reward_network,
            decoder_network=decoder_network,
            steady_state_lipschitz_network=steady_state_lipschitz_network,
            transition_loss_lipschitz_network=transition_loss_lipschitz_network,
            n_critic=n_wae_critic,
            external_latent_policy=wae_policy,
            minimizer=wae_mdp_minimizer,
            maximizer=wae_mdp_maximizer,
            encoder_optimizer=encoder_optimizer,
            wasserstein_regularizer_scale_factor=wasserstein_regularizer_scale_factor,
            reset_state_label=env_perturbation > 0.,
            state_encoder_type=EncodingType.DETERMINISTIC,
            deterministic_state_embedding=True,
            trainable_prior=False)

        # initialize WAE-DQN Agent
        self.tf_agent = WaeDqnAgent(
            time_step_spec=self.tf_env.time_step_spec(),
            latent_time_step_spec=ts.time_step_spec(self.latent_observation_spec),
            action_spec=self.tf_env.action_spec(),
            label_spec=tf.TensorSpec(self.wae_mdp.label_shape),
            q_network=self.q_network,
            optimizer=dqn_optimizer,
            encoder_optimizer=encoder_optimizer,
            epsilon_greedy=epsilon_greedy,
            boltzmann_temperature=boltzmann_temperature,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.dqn_step,
            target_update_period=target_update_period,
            target_update_tau=target_update_tau,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            emit_log_probability=True,
            summarize_grads_and_vars=summarize_grads_and_vars,
            labeling_fn=labeling_fn if env_perturbation <= 0. else ergodic_batched_labeling_function(labeling_fn),
            wae_mdp=self.wae_mdp, )
        self.tf_agent.initialize()

        # The collect policy first embeds the original observation to the latent space,
        # then execute the action based on the tf_agent collect policy
        self.collect_policy = LatentPolicyOverRealStateSpace(
            time_step_spec=self.tf_env.time_step_spec(),
            labeling_function=labeling_fn,
            latent_policy=self.tf_agent.collect_policy,
            # change to the following to explore based on the relaxed states instead of discrete ones
            # state_embedding_function=lambda _state, _label: self.wae_mdp.relaxed_state_encoding(
            #     _state, label=_label, temperature=self.wae_mdp.state_encoder_temperature,
            # ).sample()
            state_embedding_function=lambda _state, _label: self.wae_mdp.state_embedding_function(
                state=_state,
                label=_label,
                dtype=tf.float32)
        )

        # Experience Replay
        self.max_priority = tf.Variable(0., trainable=False, name='max_priority', dtype=tf.float64)
        trajectory_spec = trajectory.from_transition(
            time_step=self.tf_env.time_step_spec(),
            action_step=self.collect_policy.policy_step_spec,
            next_time_step=self.tf_env.time_step_spec())
        if self.prioritized_experience_replay:
            checkpoint_path = os.path.join(save_directory_location, 'saves', env_name, 'reverb')
            reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(checkpoint_path)

            table_name = 'prioritized_replay_buffer'
            table = reverb.Table(
                table_name,
                max_size=replay_buffer_capacity,
                sampler=reverb.selectors.Prioritized(priority_exponent=priority_exponent),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1))

            reverb_server = reverb.Server([table], checkpointer=reverb_checkpointer)

            self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                data_spec=trajectory_spec,
                sequence_length=2,
                table_name=table_name,
                local_server=reverb_server)

            _add_trajectory = reverb_utils.ReverbAddTrajectoryObserver(
                py_client=self.replay_buffer.py_client,
                table_name=table_name,
                sequence_length=2,
                stride_length=1,
                priority=self.max_priority)

            self.num_episodes = py_metrics.NumberOfEpisodes()
            self.env_steps = py_metrics.EnvironmentSteps()
            self.avg_return = py_metrics.AverageReturnMetric()
            observers = [self.num_episodes, self.env_steps, self.avg_return, _add_trajectory]

            self.driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=observers,
                max_steps=collect_steps_per_iteration)
            self.initial_collect_driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=[_add_trajectory],
                max_steps=initial_collect_steps)

        else:
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=trajectory_spec,
                batch_size=self.tf_env.batch_size,
                max_length=replay_buffer_capacity)

            self.num_episodes = tf_metrics.NumberOfEpisodes()
            self.env_steps = tf_metrics.EnvironmentSteps()
            self.avg_return = tf_metrics.AverageReturnMetric(batch_size=self.tf_env.batch_size)

            observers = [self.num_episodes, self.env_steps] if not self.parallel_envs else []
            observers += [self.avg_return, self.replay_buffer.add_batch]
            # A driver executes the agent's exploration loop and allows the observers to collect exploration information
            self.driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env, self.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
            self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env,
                self.collect_policy,
                observers=[self.replay_buffer.add_batch],
                num_steps=initial_collect_steps)

        # Dataset for WAE-DQN
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=num_parallel_environments,
            sample_batch_size=self.dqn_batch_size,
            num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

        # Dataset for WAE-MDP
        def dataset_generator(generator_fn):
            return self.replay_buffer.as_dataset(
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                num_steps=2
            ).map(
                map_func=generator_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                #  deterministic=False  # TF version >= 2.2.0
            )

        self.wae_dataset = dataset_generator(
            lambda trajectory, buffer_info: map_rl_trajectory_to_vae_input(
                trajectory=trajectory,
                labeling_function=ergodic_batched_labeling_function(labeling_fn),
                discrete_action=True,
                num_discrete_actions=self.tf_env.action_spec().maximum + 1, ))
        self.wae_iterator = iter(
            self.wae_dataset.batch(
                batch_size=self.wae_batch_size,
                drop_remainder=True
            ).prefetch(tf.data.experimental.AUTOTUNE))

        # checkpointing
        self.checkpoint_dir = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn_training_checkpoint')
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.tf_agent,
            policy=self.collect_policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
            dqn_step=self.dqn_step,
            wae_step=self.wae_step,
        )
        self.policy_dir = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn', 'policy')
        self.policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)

        # logs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(
            save_directory_location, 'logs', 'gradient_tape', env_name, 'wae_dqn_agent_training', current_time)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_directory_location = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn')

        if os.path.exists(self.checkpoint_dir):
            self.train_checkpointer.initialize_or_restore()
            print("Checkpoint loaded! global_step={:d}; dqn_step={:d}; wae_step={:d}".format(
                self.global_step.numpy(), self.dqn_step.numpy(), self.wae_step.numpy()))
        if not os.path.exists(self.policy_dir):
            os.makedirs(self.policy_dir)

    def update_progress_bar(self, progressbar, wae_mdp_loss, dqn_loss, num_steps=1):
        log_values = [
            ('dqn_loss', dqn_loss),
            ('replay_buffer_frames', self.replay_buffer.num_frames()),
            ('training_avg_returns', self.avg_return.result()),
        ]
        if not self.parallel_envs:
            log_values += [
                ('num_episodes', self.num_episodes.result()),
                ('env_steps', self.env_steps.result())
            ]

        log_values += \
            [('wae_step', self.wae_step.numpy()), ('dqn_step', self.dqn_step.numpy())] + \
            [(key, value) for key, value in wae_mdp_loss.items()] + \
            [(key, value.result()) for key, value in self.wae_mdp.loss_metrics.items()] + \
            [(key, value) for key, value in self.wae_mdp.temperature_metrics.items()]

        progressbar.add(num_steps, log_values)

    def train_and_eval(self, display_progressbar: bool = True, display_interval: float = 0.1):

        # Optimize by wrapping some of the code in a graph using TF function.
        self.tf_agent.train = common.function(self.tf_agent.train)
        if not self.prioritized_experience_replay:
            self.driver.run = common.function(self.driver.run)

        metrics = ['eval_avg_returns', 'avg_eval_episode_length', 'replay_buffer_frames', 'training_avg_returns',
                   'wae_step', 'dqn_step', 't_1', 't_2', 't_2_action', 'entropy_regularizer_scale_factor',
                   "num_episodes", "env_steps", "replay_buffer_frames", 'kl_annealing_scale_factor', 'state_rate',
                   "state_distortion", 'action_rate', 'action_distortion', 'mean_state_bits_used', 'wis_exponent',
                   'priority_logistic_smoothness', 'priority_logistic_mean',
                   'priority_logistic_max', 'priority_logistic_min', 'dynamic_reward_scaling'
                   ] + list(self.wae_mdp.loss_metrics.keys())

        if not self.parallel_envs:
            metrics += ['num_episodes', 'env_steps']

        dqn_loss = 0.

        if display_progressbar:
            progressbar = Progbar(target=self.num_iterations, interval=display_interval, stateful_metrics=metrics)
            progressbar.update(self.global_step.numpy())
            print('\n')
        else:
            progressbar = None

        env = self.tf_env if not self.prioritized_experience_replay else self.py_env

        if tf.math.less(self.replay_buffer.num_frames(), self.initial_collect_steps):
            print("Initialize replay buffer...")
            self.initial_collect_driver.run(env.current_time_step())

        print("Start training...")

        for _ in range(self.global_step.numpy(), self.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.driver.run(env.current_time_step())

            # WAE update
            wae_mdp_loss = self.wae_mdp.training_step(
                dataset=None,
                dataset_iterator=self.wae_iterator,
                batch_size=self.wae_batch_size,
                annealing_period=1,
                global_step=self.wae_step,
                display_progressbar=False,
                progressbar=None,
                eval_and_save_model_interval=np.inf,
                eval_steps=self.wae_eval_steps,
                save_directory=None,
                log_name='wae_mdp',
                train_summary_writer=None,
                log_interval=np.inf,
                start_annealing_step=0, )

            if self.global_step.numpy() % (self.n_wae_updates * self.n_wae_critic) == 0:

                # Use data from the buffer and update the agent's network.
                experience, info = next(self.iterator)
                if self.prioritized_experience_replay:
                    is_weights = tf.cast(
                        tf.stop_gradient(tf.reduce_min(info.probability[:, 0, ...])) / info.probability[:, 0, ...],
                        dtype=tf.float32)
                    loss_info = self.tf_agent.train(experience, weights=is_weights)
                    dqn_loss = loss_info.loss

                    priorities = tf.cast(tf.abs(loss_info.extra.td_error), tf.float64)
                    self.replay_buffer.update_priorities(keys=info.key[:, 0, ...], priorities=priorities)
                    if tf.reduce_max(priorities) > self.max_priority:
                        self.max_priority.assign(tf.reduce_max(priorities))
                else:
                    with self.train_summary_writer.as_default():
                        loss_info = self.tf_agent.train(experience)
                    dqn_loss = loss_info.loss

            self.update_progress_bar(progressbar, wae_mdp_loss=wae_mdp_loss, dqn_loss=dqn_loss)

            if self.global_step.numpy() % self.log_interval == 0:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('dqn_loss', dqn_loss, step=self.dqn_step)
                    tf.summary.scalar('training average returns', self.avg_return.result(), step=self.dqn_step)
                    for key, value in self.wae_mdp.loss_metrics.items():
                        tf.summary.scalar(key, value.result(), step=self.wae_step)
                # reset accumulators after logging
                self.wae_mdp.reset_metrics()

            if self.global_step.numpy() % self.eval_interval == 0 and self.global_step.numpy() != 0:
                self.train_checkpointer.save(self.global_step)
                if self.prioritized_experience_replay:
                    self.replay_buffer.py_client.checkpoint()
                self.policy_saver.save(self.policy_dir)
                self.wae_mdp.save(self.save_directory_location, 'model')
                eval_thread = threading.Thread(
                    target=self.eval,
                    args=(self.dqn_step.numpy(), progressbar),
                    daemon=False,
                    name='eval')
                eval_thread.start()
                #  self.eval(self.dqn_step.numpy(), progressbar)

            self.global_step.assign_add(1)

    def eval(self, step: int = 0, progressbar: Optional = None):
        avg_eval_return = tf_metrics.AverageReturnMetric()
        avg_eval_episode_length = tf_metrics.AverageEpisodeLengthMetric()
        saved_policy = SavedTFPolicy(self.policy_dir)
        wae_mdp = wasserstein_mdp.load(model_path=os.path.join(self.save_directory_location, 'model'))
        wae_mdp.external_latent_policy = OneHotTFPolicyWrapper(
                saved_policy,
                time_step_spec=saved_policy.time_step_spec,
                action_spec=saved_policy.action_spec)
        eval_env = wae_mdp.wrap_tf_environment(
            tf_env=tf_py_environment.TFPyEnvironment(
                EnvironmentLoader(self.env_suite).load(self.env_name)),
            labeling_function=self.labeling_fn)
        latent_policy = eval_env.wrap_latent_policy(
            saved_policy,
            observation_dtype=saved_policy.time_step_spec.observation.dtype)
        
        eval_env.reset()

        dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            latent_policy,
            [avg_eval_return, avg_eval_episode_length],
            num_episodes=self.num_eval_episodes
        ).run()

        log_values = [
            ('eval_avg_returns', avg_eval_return.result()),
            ('avg_eval_episode_length', avg_eval_episode_length.result()),
        ]
        if progressbar is not None:
            progressbar.add(0, log_values)
        else:
            print('Evaluation')
            for key, value in log_values:
                print(key, '=', value.numpy())
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Average returns', avg_eval_return.result(), step=step)
            tf.summary.scalar('Average episode length', avg_eval_episode_length.result(), step=step)

        if wae_mdp.assign_score(
            score={'eval_policy': avg_eval_return.result().numpy()},
            model_name='best_model',
            checkpoint_model=True,
            training_step=step,
            save_directory=self.save_directory_location,
        ):
            policy_saver.PolicySaver(saved_policy).save(
                os.path.join(self.save_directory_location, 'best_model', 'policy'))

        del wae_mdp
        del saved_policy

def load(model_path: str):
    wae_mdp = wasserstein_mdp.load(model_path)
    if os.path.exists(os.path.join(model_path, 'policy')):
        saved_policy = SavedTFPolicy(os.path.join(model_path, 'policy'))
        wae_mdp.external_latent_policy = OneHotTFPolicyWrapper(
            saved_policy,
            time_step_spec=saved_policy.time_step_spec,
            action_spec=saved_policy.action_spec)
    return wae_mdp


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()

    # set seed
    seed = params['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.random.set_seed(params['seed'])

    try:
        import importlib
        for module in params['import']:
            importlib.import_module(module)
    except BaseException as err:
        serr = str(err)
        print("Error to load module: " + serr)
        return -1

    learner = WaeDqnLearner(
        env_name=params['env_name'],
        env_suite=importlib.import_module('tf_agents.environments.' + params['env_suite']),
        labeling_fn=reinforcement_learning.labeling_functions[params['env_name']],
        latent_state_size=params['latent_state_size'],
        num_iterations=params['steps'],
        initial_collect_steps=params['initial_collect_steps'],
        collect_steps_per_iteration=params['collect_steps_per_iteration'],
        replay_buffer_capacity=params['replay_buffer_size'],
        network_fc_layer_params=ModelArchitecture(
            hidden_units=params['network_layers'],
            activation=params['activation']),
        state_encoder_temperature=params['state_encoder_temperature'],
        wasserstein_regularizer_scale_factor=WassersteinRegularizerScaleFactor(
            global_gradient_penalty_multiplier=params['gradient_penalty_scale_factor'],
            steady_state_scaling=params['steady_state_regularizer_scale_factor'],
            local_transition_loss_scaling=params['transition_regularizer_scale_factor'], ),
        gamma=params['gamma'],
        minimizer_learning_rate=params['wae_minimizer_learning_rate'],
        maximizer_learning_rate=params['wae_maximizer_learning_rate'],
        encoder_learning_rate=params['encoder_learning_rate'],
        dqn_learning_rate=params['policy_learning_rate'],
        log_interval=params['log_interval'],
        num_eval_episodes=params['n_eval_episodes'],
        eval_interval=params['n_eval_interval'],
        num_parallel_environments=params['n_parallel_envs'],
        wae_batch_size=params['wae_batch_size'],
        dqn_batch_size=params['policy_batch_size'],
        n_wae_critic=params['n_wae_critic'],
        n_wae_updates=params['n_wae_updates'],
        save_directory_location=params['save_dir'],
        prioritized_experience_replay=params['prioritized_experience_replay'],
        priority_exponent=params['priority_exponent'],
        wae_eval_steps=params['wae_eval_steps'],
        seed=params['seed'],
        epsilon_greedy=params['epsilon_greedy'],
        boltzmann_temperature=params['boltzmann_temperature'],
        target_update_period=params['target_update_period'],
        target_update_tau=params['target_update_scale'],
        reward_scale_factor=params['reward_scaling'],
        gradient_clipping=params['policy_gradient_clipping'],
        env_time_limit=params['env_time_limit'],
        env_perturbation=params['env_perturbation'],
        summarize_grads_and_vars=params['log_grads_and_vars'],)

    learner.train_and_eval()

    return 0


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, main))
    # app.run(main)
