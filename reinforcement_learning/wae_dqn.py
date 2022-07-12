import math
import os
import sys

from tf_agents import specs
from tf_agents.typing.types import Int, Float

import wasserstein_mdp
from reinforcement_learning.agents.wae_agent import WaeDqnAgent
from reinforcement_learning.environments.latent_environment import LatentEmbeddingTFEnvironmentWrapper
from util.io.dataset_generator import map_rl_trajectory_to_vae_input, ergodic_batched_labeling_function
from util.nn import ModelArchitecture
from wasserstein_mdp import WassersteinMarkovDecisionProcess, WassersteinRegularizerScaleFactor

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')

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
from absl import flags

import PIL

import tensorflow as tf
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.utils.generic_utils import Progbar
from tf_agents.agents import CategoricalDqnAgent
from tf_agents.agents.dqn import dqn_agent

import tf_agents
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.metrics import tf_metrics, tf_metric, py_metrics
from tf_agents.networks import q_network, categorical_q_network
from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer, reverb_replay_buffer, reverb_utils
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.trajectory import experience_to_transitions
from tf_agents.utils import common
from tf_agents.policies import policy_saver, categorical_q_policy, boltzmann_policy, q_policy, py_tf_eager_policy, \
    greedy_policy
import tf_agents.trajectories.time_step as ts
from reinforcement_learning.environments import EnvironmentLoader

flags.DEFINE_string(
    'env_name', help='Name of the environment', default='CartPole-v0'
)
flags.DEFINE_string(
    'env_suite', help='Environment suite', default='suite_gym'
)
flags.DEFINE_integer(
    'steps', help='Number of iterations', default=int(2e5)
)
flags.DEFINE_integer(
    'num_parallel_env', help='Number of parallel environments', default=1
)
flags.DEFINE_integer(
    'seed', help='set seed', default=42
)
flags.DEFINE_string(
    'save_dir', help='Save directory location', default='.'
)
flags.DEFINE_multi_integer(
    'network_layers',
    help='number of units per MLP layers',
    default=[100, 50]
)
flags.DEFINE_integer(
    'batch_size',
    help='batch_size',
    default=64
)
flags.DEFINE_float(
    'learning_rate',
    help='learning rate',
    default=1e-3
)
flags.DEFINE_integer(
    'collect_steps_per_iteration',
    help='Collect steps per iteration',
    default=1
)
flags.DEFINE_integer(
    'target_update_period',
    help="Period for update of the target networks",
    default=20
)
flags.DEFINE_float(
    'gamma',
    help='discount_factor',
    default=0.99
)
flags.DEFINE_bool(
    'prioritized_experience_replay',
    help="use priority-based replay buffer (with Deepmind's reverb)",
    default=False
)
flags.DEFINE_float(
    'priority_exponent',
    help='priority exponent for computing the probabilities of the samples from the prioritized replay buffer',
    default=0.6
)
flags.DEFINE_multi_string(
    'import',
    help='list of modules to additionally import',
    default=[]
)
FLAGS = flags.FLAGS

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
            state_encoder_temperature: Float = 2./3,
            state_encoder_network: ModelArchitecture = default_architecture,
            transition_network: ModelArchitecture = default_architecture,
            reward_network: ModelArchitecture = default_architecture,
            decoder_network: ModelArchitecture = default_architecture,
            steady_state_lipschitz_network: ModelArchitecture = default_architecture,
            transition_loss_lipschitz_network: ModelArchitecture = default_architecture,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=1., global_gradient_penalty_multiplier=1.),
            gamma: float = 0.99,
            target_update_period: int = 20,
            autoencoder_learning_rate: float = 3e-4,
            wasserstein_learning_rate: float = 3e-4,
            dqn_learning_rate: float = 3e-4,
            log_interval: int = 2500,
            num_eval_episodes: int = 30,
            eval_interval: int = int(1e4),
            parallelization: bool = True,
            num_parallel_environments: int = 4,
            wae_batch_size: int = 128,
            dqn_batch_size: int = 64,
            n_wae_critic: Int = 5,
            n_dqn: Int = 5,
            save_directory_location: str = '.',
            prioritized_experience_replay: bool = False,
            priority_exponent: float = 0.6,
            wae_eval_steps: Int = int(1e4),
            seed: Optional[int] = 42,
    ):
        self.parallelization = parallelization and not prioritized_experience_replay

        if collect_steps_per_iteration is None:
            collect_steps_per_iteration = dqn_batch_size
        if parallelization:
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

        self.parallelization = parallelization
        self.num_parallel_environments = num_parallel_environments

        self.dqn_batch_size = dqn_batch_size
        self.wae_batch_size = wae_batch_size

        self.prioritized_experience_replay = prioritized_experience_replay
        self.n_wae_critic = n_wae_critic
        self.n_dqn = n_dqn
        self.wae_eval_steps = None

        env_loader = EnvironmentLoader(env_suite, seed=seed)

        if parallelization:
            self.tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_loader.load(env_name)] * num_parallel_environments))
            _obs = self.tf_env.reset().observation
            self.py_env = env_suite.load(env_name)
            self.py_env.reset()
            # self.eval_env = tf_py_environment.TFPyEnvironment(self.py_env)
        else:
            self.py_env = env_loader.load(env_name)
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
            self.latent_observation_spec,
            self.tf_env.action_spec(),
            fc_layer_params=network_fc_layer_params)
        policy = q_policy.QPolicy(
            time_step_spec=self.latent_observation_spec,
            action_spec=self.tf_env.action_spec(),
            q_network=self.q_network, )
        policy = greedy_policy.GreedyPolicy(policy)

        ae_optimizer = tf.keras.optimizers.Adam(learning_rate=autoencoder_learning_rate)
        wasserstein_optimizer = tf.keras.optimizer.Adam(learning_rate=wasserstein_learning_rate)
        dqn_optimizer = tf.keras.optimizer.Adam(learning_rate=dqn_learning_rate)

        self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name="global_step")
        self.dqn_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name="dqn_step")
        self.wae_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name="wae_step")

        self.wae_mdp = WassersteinMarkovDecisionProcess(
            state_shape=self.tf_env.observation_spec().shape,
            action_shape=self.tf_env.action_spec().shape,
            reward_shape=self.tf_env.time_step_spec().reward.shape,
            label_shape=labeling_fn(_obs).shape[1:],
            discretize_action_space=True,
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
            external_latent_policy=policy,
            autoencoder_optimizer=ae_optimizer,
            wasserstein_optimizer=wasserstein_optimizer,
            wasserstein_regularizer_scale_factor=wasserstein_regularizer_scale_factor,)

        self.tf_agent = WaeDqnAgent(
            time_step_spec=self.tf_env.time_step_spec(),
            action_spec=self.tf_env.action_spec(),
            q_network=self.q_network,
            optimizer=dqn_optimizer,
            epsilon_greedy=epsilon_greedy,
            boltzmann_temperature=boltzmann_temperature,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.dqn_step,
            target_update_period=target_update_period,
            target_update_tau=target_update_tau,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            # emit_log_probability=True
        )

        self.tf_agent.initialize()

        # define the policy from the learning agent
        self.collect_policy = self.tf_agent.collect_policy

        self.max_priority = tf.Variable(0., trainable=False, name='max_priority', dtype=tf.float64)
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
                data_spec=self.tf_agent.collect_data_spec,
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
                data_spec=self.tf_agent.collect_data_spec,
                batch_size=self.tf_env.batch_size,
                max_length=replay_buffer_capacity)

            self.num_episodes = tf_metrics.NumberOfEpisodes()
            self.env_steps = tf_metrics.EnvironmentSteps()
            self.avg_return = tf_metrics.AverageReturnMetric(batch_size=self.tf_env.batch_size)
            #  self.safety_violations = NumberOfSafetyViolations(self.labeling_function)

            observers = [self.num_episodes, self.env_steps] if not parallelization else []
            observers += [self.avg_return, self.replay_buffer.add_batch]
            # A driver executes the agent's exploration loop and allows the observers to collect exploration information
            self.driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env, self.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
            self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env,
                self.collect_policy,
                observers=[self.replay_buffer.add_batch],
                num_steps=initial_collect_steps)

        # Dataset generates trajectories with shape [Bx2x...]
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=num_parallel_environments,
            sample_batch_size=self.dqn_batch_size,
            num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

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
                num_discrete_actions=self.tf_env.action_spec().shape[0], ))
        self.wae_iterator = iter(
            self.wae_dataset.batch(
                batch_size=self.wae_batch_size,
                drop_remainder=True
            ).prefetch(tf.data.experimental.AUTOTUNE))

        self.checkpoint_dir = os.path.join(save_directory_location, 'saves', env_name, 'dqn_training_checkpoint')
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
        self.policy_dir = os.path.join(save_directory_location, 'saves', env_name, 'dqn_policy')
        self.policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(
            save_directory_location, 'logs', 'gradient_tape', env_name, 'dqn_agent_training', current_time)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_directory_location = os.path.join(save_directory_location, 'saves', env_name)

        if os.path.exists(self.checkpoint_dir):
            self.train_checkpointer.initialize_or_restore()
            self.global_step = tf.compat.v1.train.get_global_step()
            print("Checkpoint loaded! global_step={:d}; dqn_step={:d}; wae_step={:d}".format(
                self.global_step.numpy(), self.dqn_step.numpy(), self.wae_step.numpy()))
        if not os.path.exists(self.policy_dir):
            os.makedirs(self.policy_dir)

    def train_and_eval(self, display_progressbar: bool = True, display_interval: float = 0.1):

        # Optimize by wrapping some of the code in a graph using TF function.
        self.tf_agent.train = common.function(self.tf_agent.train)
        if not self.prioritized_experience_replay:
            self.driver.run = common.function(self.driver.run)

        metrics = [
            'eval_avg_returns',
            'avg_eval_episode_length',
            'replay_buffer_frames',
            'training_avg_returns'
        ]
        if not self.parallelization:
            metrics += ['num_episodes', 'env_steps']

        train_loss = 0.

        # load the checkpoint

        def update_progress_bar(num_steps=1):
            if display_progressbar:
                log_values = [
                    ('loss', train_loss),
                    ('replay_buffer_frames', self.replay_buffer.num_frames()),
                    ('training_avg_returns', self.avg_return.result()),
                ]
                if not self.parallelization:
                    log_values += [
                        ('num_episodes', self.num_episodes.result()),
                        ('env_steps', self.env_steps.result())
                    ]
                progressbar.add(num_steps, log_values)

        if display_progressbar:
            progressbar = Progbar(target=self.num_iterations, interval=display_interval, stateful_metrics=metrics)
        else:
            progressbar = None

        env = self.tf_env if not self.prioritized_experience_replay else self.py_env

        if tf.math.less(self.replay_buffer.num_frames(), self.initial_collect_steps):
            print("Initialize replay buffer...")
            self.initial_collect_driver.run(env.current_time_step())

        print("Start training...")

        update_progress_bar(self.global_step.numpy())

        for _ in range(self.global_step.numpy(), self.num_iterations):
            step = self.global_step.numpy()

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.driver.run(env.current_time_step())

            # WAE update
            self.wae_mdp.training_step(
                dataset=self.wae_dataset,
                dataset_iterator=self.wae_iterator,
                batch_size=self.wae_batch_size,
                annealing_period=1,
                global_step=self.global_step,
                display_progressbar=True,
                progressbar=progressbar,
                eval_and_save_model_interval=self.eval_interval,
                eval_steps=self.wae_eval_steps,
                save_directory=self.save_directory_location,
                log_name='wae_mdp',
                train_summary_writer=self.train_summary_writer,
                log_interval=self.log_interval,
                start_annealing_step=0,)

            if step % self.n_dqn == 0:
                # Use data from the buffer and update the agent's network.
                # experience = replay_buffer.gather_all()
                experience, info = next(self.iterator)
                if self.prioritized_experience_replay:
                    is_weights = tf.cast(
                        tf.stop_gradient(tf.reduce_min(info.probability[:, 0, ...])) / info.probability[:, 0, ...],
                        dtype=tf.float32)
                    loss_info = self.tf_agent.train(experience, weights=is_weights)
                    train_loss = loss_info.loss

                    priorities = tf.cast(tf.abs(loss_info.extra.td_error), tf.float64)
                    self.replay_buffer.update_priorities(keys=info.key[:, 0, ...], priorities=priorities)
                    if tf.reduce_max(priorities) > self.max_priority:
                        self.max_priority.assign(tf.reduce_max(priorities))
                else:
                    loss_info = self.tf_agent.train(experience)
                    train_loss = loss_info.loss

                update_progress_bar()

            if step % self.log_interval == 0:
                self.train_checkpointer.save(self.global_step)
                if self.prioritized_experience_replay:
                    self.replay_buffer.py_client.checkpoint()
                self.policy_saver.save(self.policy_dir)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss, step=step)
                    tf.summary.scalar('training average returns', self.avg_return.result(), step=step)

            if step % self.eval_interval == 0:
                eval_thread = threading.Thread(target=self.eval, args=(step, progressbar), daemon=True, name='eval')
                eval_thread.start()

    def eval(self, step: int = 0, progressbar: Optional = None):
        avg_eval_return = tf_metrics.AverageReturnMetric()
        avg_eval_episode_length = tf_metrics.AverageEpisodeLengthMetric()
        saved_policy = tf.compat.v2.saved_model.load(self.policy_dir)
        eval_env = tf_py_environment.TFPyEnvironment(
            EnvironmentLoader(self.env_suite).load(self.env_name))
        wae_mdp = wasserstein_mdp.load(model_path=os.path.join(self.save_directory_location, 'wae_mdp'))
        eval_env.reset()

        dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            saved_policy,
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


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()
    print(params)
    tf.random.set_seed(params['seed'])
    try:
        import importlib
        env_suite = importlib.import_module('tf_agents.environments.' + params['env_suite'])
        for module in params['import']:
            importlib.import_module(module)
    except BaseException as err:
        serr = str(err)
        print("Error to load module: " + serr)
        return -1
    learner = DQNLearner(
        env_name=params['env_name'],
        env_suite=env_suite,
        num_iterations=params['steps'],
        num_parallel_environments=params['num_parallel_env'],
        save_directory_location=params['save_dir'],
        learning_rate=params['learning_rate'],
        network_fc_layer_params=params['network_layers'],
        batch_size=params['batch_size'],
        parallelization=params['num_parallel_env'] > 1,
        target_update_period=params['target_update_period'],
        gamma=params['gamma'],
        collect_steps_per_iteration=params['collect_steps_per_iteration'],
        prioritized_experience_replay=params['prioritized_experience_replay'],
        priority_exponent=params['priority_exponent'],
        seed=params['seed']
    )
    learner.train_and_eval()
    return 0


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, main))
    # app.run(main)
