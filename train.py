import os

import tensorflow as tf
from absl import app
from absl import flags
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Sequential

import reinforcement_learning
import variational_action_discretizer
import variational_mdp
from util.io import dataset_generator

flags.DEFINE_string(
    "dataset_path",
    help="Path of the directory containing the dataset files in hdf5 format.",
    default='')
flags.DEFINE_integer("batch_size", default=128, help="Batch size.")
flags.DEFINE_integer(
    "mixture_components",
    default=1,
    help="Number of gaussian mixture components used to model the reconstruction distributions.")
flags.DEFINE_integer(
    "action_mixture_components",
    default=0,
    help="Number of gaussian mixture components used to model the action reconstruction distribution (optional). If not"
         "set, all mixture distributions take the same value obtained via --mixture_components.")
flags.DEFINE_bool(
    "full_covariance",
    default=False,
    help="If set, the states and rewards reconstruction distributions will use a full covariance matrix instead of"
         "a diagonal matrix."
)
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer("latent_size", default=17, help='Number of bits used for the discrete latent state space.')
flags.DEFINE_float(
    "max_state_decoder_variance",
    default="0.",
    help='Maximum variance allowed for the state decoder.'
)
flags.DEFINE_float(
    "encoder_temperature",
    default=-1.,
    help="Temperature of the relaxation of the discrete encoder distribution."
)
flags.DEFINE_float(
    "prior_temperature",
    default=-1.,
    help="Temperature of relaxation of the discrete prior distribution over latent variables."
)
flags.DEFINE_float(
    "relaxed_state_encoder_temperature",
    default=-1.,
    help="Temperature of the binary concrete relaxation encoder distribution over latent states."
)
flags.DEFINE_float(
    "relaxed_state_prior_temperature",
    default=-1.,
    help="Temperature of the binary concrete relaxation prior distribution over latent states."
)
flags.DEFINE_float(
    "encoder_temperature_decay_rate",
    default=1e-6,
    help="Decay rate used to anneal the temperature of the encoder distribution over latent states."
)
flags.DEFINE_float(
    "prior_temperature_decay_rate",
    default=2e-6,
    help="Decay rate used to anneal the temperature of the prior distribution over latent states."
)
flags.DEFINE_float(
    "regularizer_scale_factor",
    default=0.,
    help="Cross-entropy regularizer scale factor."
)
flags.DEFINE_float(
    "regularizer_decay_rate",
    default=0.,
    help="Cross-entropy regularizer decay rate."
)
flags.DEFINE_float(
    "kl_annealing_scale_factor",
    default=1.,
    help='Scale factor of the KL terms of the ELBO.'
)
flags.DEFINE_float(
    "kl_annealing_growth_rate",
    default=0.,
    help='Annealing growth rate of the ELBO KL terms scale factor.'
)
flags.DEFINE_integer(
    "start_annealing_step",
    default=int(1e4),
    help="Step from which temperatures and scale factors start to be annealed."
)
flags.DEFINE_integer(
    "max_steps",
    default=int(1e6),
    help="Maximum number of training steps."
)
flags.DEFINE_string(
    "save_dir",
    default=".",
    help="Checkpoints and models save directory."
)
flags.DEFINE_bool(
    "display_progressbar",
    default=False,
    help="Display progressbar."
)
flags.DEFINE_bool(
    "action_discretizer",
    default=False,
    help="Discretize the action space via a VAE already trained. Require the flag --load_vae to be set."
)
flags.DEFINE_bool(
    "one_output_per_action",
    default=False,
    help="Set whether discrete action networks use one output per action or use the latent action as input."
)
flags.DEFINE_boolean(
    "do_not_eval",
    default=False,
    help="Set this flag to not perform an evaluation of the ELBO (using discrete latent variables) during training."
)
flags.DEFINE_bool(
    "full_vae_optimization",
    default=False,
    help='Set whether the ELBO is optimized over the whole VAE or if the optimization is only focused on the'
         'state or action discretizer part of the VAE.'
)
flags.DEFINE_bool(
    'relaxed_state_encoding',
    default=False,
    help='Use a relaxed encoding of states to optimize the action discretizer part of the VAE.'
)
flags.DEFINE_integer(
    "number_of_discrete_actions",
    default=16,
    help='Number of discrete actions per states to learn.'
)
flags.DEFINE_string(
    "load_vae",
    default='',
    help='Path of a VAE model already trained to load (saved via the tf.saved_model function).'
)
flags.DEFINE_multi_integer(
    "encoder_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the encoder.'
)
flags.DEFINE_multi_integer(
    "decoder_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the decoder.'
)
flags.DEFINE_multi_integer(
    "transition_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the transition network.'
)
flags.DEFINE_multi_integer(
    "reward_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the reward network.'
)
flags.DEFINE_multi_integer(
    "discrete_policy_layers",
    default=[256, 256],
    help="Number of units to use for each layer of the simplified policy network."
)
flags.DEFINE_string(
    "policy_path",
    default='',
    help="Path of a policy in tf.saved_model format."
)
flags.DEFINE_string(
    "environment",
    default='',
    help="Name of the agent's environment."
)
flags.DEFINE_string(
    "env_suite",
    default='suite_gym',
    help='Name of the tf_agents environment suite.'
)
flags.DEFINE_integer(
    "parallel_env",
    default=1,
    help='Number of parallel environments to be used during training.'
)
flags.DEFINE_float(
    'state_scaling',
    default=1.,
    help='Scaler for the input states of the environment.'
)
flags.DEFINE_integer(
    'annealing_period',
    default=1,
    help='annealing period'
)
flags.DEFINE_bool(
    'aggressive_training',
    default=False,
    help='Set whether to perform aggressive inference optimizations.'
)
flags.DEFINE_integer(
    'initial_collect_steps',
    default=int(1e4),
    help='Number of frames to be collected in the replay buffer before starting the training.'
)
flags.DEFINE_float(
    'seed', help='set seed', default=42
)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()

    tf.random.set_seed(params['seed'])

    def check_missing_argument(name: str):
        if params[name] == '':
            raise RuntimeError('Missing argument: --{}'.format(name))

    if params['dataset_path'] == '':
        for param in ('policy_path', 'environment'):
            check_missing_argument(param)

    relaxed_state_encoder_temperature = params['relaxed_state_encoder_temperature']
    relaxed_state_prior_temperature = params['relaxed_state_prior_temperature']
    if params['encoder_temperature'] < 0.:
        if params['action_discretizer']:
            params['encoder_temperature'] = 1. / (params['number_of_discrete_actions'] - 1)
        else:
            params['encoder_temperature'] = 0.99
    if params['prior_temperature'] < 0.:
        if params['action_discretizer']:
            params['prior_temperature'] = params['encoder_temperature'] / 1.5
        else:
            params['prior_temperature'] = 0.95
    if relaxed_state_encoder_temperature < 0:
        relaxed_state_encoder_temperature = params['encoder_temperature']
    if relaxed_state_prior_temperature < 0:
        relaxed_state_prior_temperature = params['prior_temperature']

    dataset_path = params['dataset_path']
    environment_name = params['environment']

    batch_size = params['batch_size']
    mixture_components = params['mixture_components']
    latent_state_size = params['latent_size']  # depends on the number of bits reserved for labels

    if params['load_vae'] != '':
        name_list = params['load_vae'].split(os.path.sep)
        if 'models' in name_list and name_list.index('models') < len(name_list) - 1:
            base_model_name = os.path.join(*name_list[name_list.index('models') + 1:])
        else:
            base_model_name = os.path.split(params['load_vae'])[-1]

        if params['load_vae'][-1] == os.path.sep:
            params['load_vae'] = params['load_vae'][:-1]
    else:
        base_model_name = ''

    if params['policy_path'] != '' and params['policy_path'][-1] == os.path.sep:
        params['policy_path'] = params['policy_path'][:-1]

    vae_name = ''
    if not params['action_discretizer'] or params['full_vae_optimization']:
        vae_name = 'vae_LS{}_MC{}_CER{}-decay={:g}_KLA{}-growth={:g}_TD{:.2f}-{:.2f}_{}-{}'.format(
            latent_state_size,
            mixture_components,
            params['regularizer_scale_factor'],
            params['regularizer_decay_rate'],
            params['kl_annealing_scale_factor'],
            params['kl_annealing_growth_rate'],
            relaxed_state_encoder_temperature,
            relaxed_state_prior_temperature,
            params['encoder_temperature_decay_rate'],
            params['prior_temperature_decay_rate'])
    if params['action_discretizer']:
        if vae_name != '':
            base_model_name = vae_name
        vae_name = os.path.join(
            base_model_name,
            os.path.split(params['policy_path'])[-1],
            'action_discretizer',
            'LA{}_MC{}_CER{}-decay={:g}_KLA{}-growth={:g}_TD{:.2f}-{:.2f}_{}-{}'.format(
                params['number_of_discrete_actions'],
                mixture_components,
                params['regularizer_scale_factor'],
                params['regularizer_decay_rate'],
                params['kl_annealing_scale_factor'],
                params['kl_annealing_growth_rate'],
                params['encoder_temperature'],
                params['prior_temperature'],
                params['encoder_temperature_decay_rate'],
                params['prior_temperature_decay_rate']
            )
        )
    if params['max_state_decoder_variance'] > 0:
        vae_name += '_max_state_decoder_variance={:g}'.format(params['max_state_decoder_variance'])
    if params['state_scaling'] != 1.:
        vae_name += '_state_scaling={:g}'.format(params['state_scaling'])

    additional_parameters = [
        'one_output_per_action', 'full_vae_optimization', 'relaxed_state_encoding', 'full_covariance'
    ]
    nb_additional_params = sum(
        map(lambda x: params[x], additional_parameters))
    if nb_additional_params > 0:
        vae_name += ('_params={}' + '-{}' * (nb_additional_params - 1)).format(
            *filter(lambda x: params[x], additional_parameters))

    cycle_length = batch_size // 2
    block_length = batch_size // cycle_length
    activation = getattr(tf.nn, params["activation"])

    def generate_dataset():
        return dataset_generator.create_dataset(
            hdf5_files_path=dataset_path,
            cycle_length=cycle_length,
            block_length=block_length)

    dataset_size = -1

    def generate_network_components(name=''):

        if name != '':
            name += '_'

        # Encoder body
        q = Sequential(name="{}encoder_network_body".format(name))
        for i, units in enumerate(params['encoder_layers']):
            q.add(Dense(units, activation=activation, name="{}encoder_{}".format(name, i)))

        # Transition network body
        p_t = Sequential(name="{}transition_network_body".format(name))
        for i, units in enumerate(params['transition_layers']):
            p_t.add(Dense(units, activation=activation, name='{}transition_{}'.format(name, i)))

        # Reward network body
        p_r = Sequential(name="{}reward_network_body".format(name))
        for i, units in enumerate(params['reward_layers']):
            p_r.add(Dense(units, activation=activation, name='{}reward_{}'.format(name, i)))

        # Decoder network body
        p_decode = Sequential(name="{}decoder_network_body".format(name))
        for i, units in enumerate(params['decoder_layers']):
            p_decode.add(Dense(units, activation=activation, name='{}decoder_{}'.format(name, i)))

        # Policy network body
        discrete_policy = Sequential(name="{}policy_network_body".format(name))
        for i, units in enumerate(params['discrete_policy_layers']):
            discrete_policy.add(Dense(units, activation=activation, name='{}discrete_policy_{}'.format(name, i)))

        return q, p_t, p_r, p_decode, discrete_policy

    if params['env_suite'] != '':
        try:
            import importlib
            environment_suite = importlib.import_module('tf_agents.environments.' + params['env_suite'])
        except BaseException as err:
            serr = str(err)
            print("Error to load the module '" + params['env_suite'] + "': " + serr)
    else:
        environment_suite = None

    if params['dataset_path'] != '':
        dummy_dataset = generate_dataset()
        dataset_size = dataset_generator.get_num_samples(dataset_path, batch_size=batch_size, drop_remainder=True)

        state_shape, action_shape, reward_shape, _, label_shape = [
            tuple(spec.shape.as_list()[1:]) for spec in dummy_dataset.element_spec
        ]

        del dummy_dataset

    else:
        environment = environment_suite.load(environment_name)

        state_shape, action_shape, reward_shape, label_shape = (
            shape if shape != () else (1,) for shape in (
            environment.observation_spec().shape,
            environment.action_spec().shape,
            environment.time_step_spec().reward.shape,
            tuple(reinforcement_learning.labeling_functions[environment_name](
                environment.reset().observation).shape)
        )
        )
    if params['load_vae'] == '':
        q, p_t, p_r, p_decode, _ = generate_network_components(name='state')
        vae_mdp_model = variational_mdp.VariationalMarkovDecisionProcess(
            state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
            encoder_network=q, transition_network=p_t, reward_network=p_r, decoder_network=p_decode,
            latent_state_size=latent_state_size,
            mixture_components=mixture_components,
            encoder_temperature=relaxed_state_encoder_temperature,
            prior_temperature=relaxed_state_prior_temperature,
            encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
            prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
            regularizer_scale_factor=params['regularizer_scale_factor'],
            regularizer_decay_rate=params['regularizer_decay_rate'],
            kl_scale_factor=params['kl_annealing_scale_factor'],
            kl_annealing_growth_rate=params['kl_annealing_growth_rate'],
            multivariate_normal_full_covariance=params['full_covariance'],
            max_decoder_variance=(
                None if params['max_state_decoder_variance'] == 0. else params['max_state_decoder_variance']
            ),
            state_scaler=lambda x: x * params['state_scaling'],
        )
    else:
        vae_mdp_model = variational_mdp.load(params['load_vae'])
        vae_mdp_model.encoder_temperature = relaxed_state_encoder_temperature
        vae_mdp_model.prior_temperature = relaxed_state_prior_temperature

    if params['action_discretizer']:
        if params['full_vae_optimization'] and params['load_vae'] != '':
            vae_mdp_model = variational_action_discretizer.load(params['load_vae'], full_optimization=True)
        else:
            q, p_t, p_r, p_decode, discrete_policy = generate_network_components(name='action')
            vae_mdp_model = variational_action_discretizer.VariationalActionDiscretizer(
                vae_mdp=vae_mdp_model,
                number_of_discrete_actions=params['number_of_discrete_actions'],
                action_encoder_network=q, transition_network=p_t, reward_network=p_r, action_decoder_network=p_decode,
                simplified_policy_network=discrete_policy,
                encoder_temperature=params['encoder_temperature'],
                prior_temperature=params['prior_temperature'],
                encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
                prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
                one_output_per_action=params['one_output_per_action'],
                relaxed_state_encoding=params['relaxed_state_encoding'],
                full_optimization=params['full_vae_optimization'],
                reconstruction_mixture_components=(
                    mixture_components if params['action_mixture_components'] == 0
                    else params['action_mixture_components']
                ),
            )
        vae_mdp_model.kl_scale_factor = params['kl_annealing_scale_factor']
        vae_mdp_model.kl_growth_rate = params['kl_annealing_growth_rate']
        vae_mdp_model.regularizer_scale_factor = params['regularizer_scale_factor']
        vae_mdp_model.regularizer_decay_rate = params['regularizer_decay_rate']

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    step = tf.compat.v1.train.get_or_create_global_step()
    checkpoint_directory = os.path.join(params['save_dir'], 'saves', environment_name, 'training_checkpoints', vae_name)
    print("checkpoint path:", checkpoint_directory)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vae_mdp_model, step=step)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_directory, max_to_keep=1)

    if base_model_name != '':
        vae_name_list = vae_name.split(os.path.sep)
        vae_name_list[0] = '_'.join(base_model_name.split(os.path.sep))
        vae_name = os.path.join(*vae_name_list)

    if dataset_path == '':
        policy = tf.compat.v2.saved_model.load(params['policy_path'])

        vae_mdp_model.train_from_policy(policy=policy, environment_suite=environment_suite,
                                        env_name=environment_name,
                                        labeling_function=reinforcement_learning.labeling_functions[environment_name],
                                        batch_size=batch_size, optimizer=optimizer, checkpoint=checkpoint,
                                        manager=manager, log_name=vae_name,
                                        start_annealing_step=params['start_annealing_step'],
                                        logs=True,
                                        num_iterations=params['max_steps'],
                                        display_progressbar=params['display_progressbar'],
                                        save_directory=params['save_dir'],
                                        parallelization=params['parallel_env'] > 1,
                                        num_parallel_call=params['parallel_env'],
                                        eval_steps=int(1e3) if not params['do_not_eval'] else 0,
                                        #  get_policy_evaluation=(
                                        #      None if not params['action_discretizer'] else
                                        #      vae_mdp_model.get_abstract_policy),
                                        #  wrap_eval_tf_env=(
                                        #      None if not params['action_discretizer'] else
                                        #      lambda tf_env: vae_mdp_model.wrap_tf_environment(
                                        #          tf_env, reinforcement_learning.labeling_functions[environment_name]
                                        #      )
                                        #  ),
                                        annealing_period=params['annealing_period'],
                                        aggressive_training=params['aggressive_training'],
                                        initial_collect_steps=params['initial_collect_steps'])
    else:
        vae_mdp_model.train_from_dataset(dataset_generator=generate_dataset,
                                         batch_size=batch_size, optimizer=optimizer, checkpoint=checkpoint,
                                         manager=manager, dataset_size=dataset_size,
                                         annealing_period=params['annealing_period'],
                                         start_annealing_step=params['start_annealing_step'],
                                         log_name=vae_name, logs=True, max_steps=params['max_steps'],
                                         display_progressbar=params['display_progressbar'],
                                         save_directory=params['save_dir'],
                                         eval_ratio=int(1e3) if not params['do_not_eval'] else 0, )

    return 0


if __name__ == '__main__':
    app.run(main)
