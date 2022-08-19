from absl import flags

# =========================================================
# RL Flags
# =========================================================
flags.DEFINE_string(
    'env_name', help='Name of the environment', default='CartPole-v0'
)
flags.DEFINE_string(
    'env_suite', help='Environment suite', default='suite_gym'
)
flags.DEFINE_integer(
    'n_parallel_envs',
    help='Number of parallel environments',
    default=4
)
flags.DEFINE_integer(
    'env_time_limit',
    default=None,
    help='(Optional) enforce the environment to reset after the input time limit'
)
flags.DEFINE_float(
    'env_perturbation',
    default=.75,
    help="Probability of the recursive environment perturbation. "
         "If < 1, the environment is recursively perturbed at when reset; in that case, the input value corresponds to "
         "the probability of going to the initial state. This enforces the ergodicity of the environment"
)
flags.DEFINE_multi_integer(
    'network_layers',
    help='number of units per fully connected Dense layers',
    default=[128, 128]
)
flags.DEFINE_integer(
    'policy_batch_size',
    help='Batch size for learning the policy',
    default=64
)
flags.DEFINE_float(
    'policy_learning_rate',
    help='learning rate',
    default=3e-4
)
flags.DEFINE_integer(
    'collect_steps_per_iteration',
    help='Collect steps per iteration',
    default=1
)
flags.DEFINE_integer(
    'initial_collect_steps',
    help="Number of collect steps to perform in the environment before performing an update",
    default=int(1e3)
)
flags.DEFINE_integer(
    'target_update_period',
    help="Period for update of the target networks",
    default=20
)
flags.DEFINE_float(
    'target_update_scale',
    help='Weights scaling for the target network updates. '
         'Set to 1 to perform hard updates and < 1 for soft updates',
    default=1.
)
flags.DEFINE_float(
    'gamma',
    help='discount_factor',
    default=0.99
)
flags.DEFINE_integer(
    'replay_buffer_size',
    help='Replay buffer maximum capacity',
    default=int(1e6)
)
flags.DEFINE_bool(
    'prioritized_experience_replay',
    help="Use priority-based replay buffer (via Deepmind reverb)",
    default=False
)
flags.DEFINE_float(
    'priority_exponent',
    help='priority exponent for computing the probabilities of the samples from the prioritized replay buffer',
    default=0.6
)
flags.DEFINE_integer(
    'n_eval_episodes',
    help='Number of episodes to perform for evaluating the policy',
    default=30
)
flags.DEFINE_integer(
    'n_eval_interval',
    help='Number of steps to perform before evaluating the policy',
    default=int(1e4)
)
flags.DEFINE_float(
    'epsilon_greedy',
    help='Epsilon value for the epsilon greedy based exploration policy',
    default=0.1
)
flags.DEFINE_float(
    'boltzmann_temperature',
    help='(Optional) softmax temperature for a Boltzmann exploration policy',
    default=None,
)
flags.DEFINE_float(
    'reward_scaling',
    help='Scale factor for the rewards',
    default=1.
)
flags.DEFINE_float(
    'policy_gradient_clipping',
    help='(Optional) norm length to clip gradients',
    default=None
)
# =========================================================
# WAE-MDP Flags
# =========================================================
flags.DEFINE_string(
    'activation',
    help='Activation function for the fully connected Dense hidden layers of the WAE-MDP model',
    default='relu'
)
flags.DEFINE_integer(
    'wae_batch_size',
    help='Batch size for the WAE',
    default=128,
)
flags.DEFINE_integer(
    'n_wae_critic',
    help="number of Wasserstein critic (discriminators) updates before performing a full WAE-MDP update",
    default=5,
)
flags.DEFINE_integer(
    'n_wae_updates',
    help='number of WAE-MDP updates before performing a policy update',
    default=5,
)
flags.DEFINE_integer(
    'latent_state_size',
    help='Number of bits to use to represent the state space',
    default=10,
)
flags.DEFINE_float(
    'state_encoder_temperature',
    help='Temperature of the state encoder (encoding each original states to relaxed Bernoulli)',
    default=2. / 3.,
    lower_bound=0.,
    upper_bound=1.,
)
flags.DEFINE_float(
    'transition_regularizer_scale_factor',
    help='Scale factor for the WAE-MDP transition regularizer',
    default=60.
)
flags.DEFINE_float(
    'steady_state_regularizer_scale_factor',
    help='Scale factor for the WAE-MDP steady-state regularizer',
    default=60.
)
flags.DEFINE_float(
    'gradient_penalty_scale_factor',
    help='Scale factor for penalizing the gradients of the WAE-MDP regularizers',
    default=10.,
)
flags.DEFINE_float(
    'auto_encoder_learning_rate',
    help='Learning rate for the WAE-MDP autoencoder part (for the min operation)',
    default=3e-4
)
flags.DEFINE_float(
    'wasserstein_learning_rate',
    help='Learning rate for the WAE-MDP regularizers (for the max operation)',
    default=3e-4
)
flags.DEFINE_integer(
    'wae_eval_steps',
    help='Number of steps to perform to evaluate the WAE loss (with discrete latent spaces, i.e., via the zero limit '
         'temperature)',
    default=int(1e4)
)
# =========================================================
# Utils
# =========================================================
flags.DEFINE_integer(
    'steps', help='Total number of iterations', default=int(2e5)
)
flags.DEFINE_multi_string(
    'import',
    help='list of modules to additionally import',
    default=[]
)
flags.DEFINE_integer(
    'seed', help='set seed', default=42
)
flags.DEFINE_string(
    'save_dir', help='Save directory location', default='.'
)
flags.DEFINE_integer(
    'log_interval',
    help='Number of global steps before logging',
    default=200
)
FLAGS = flags.FLAGS
