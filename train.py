import functools
import importlib
import json
import os
import random
import sys
from collections import namedtuple
import datetime

import numpy as np
import tensorflow as tf
import tf_agents
from absl import app, flags
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import tensorflow_probability as tfp
import tensorflow_probability.python.bijectors as tfb
from tf_agents import specs
from tf_agents.environments import tf_py_environment
from tf_agents.environments.wrappers import HistoryWrapper
from tf_agents.specs import tensor_spec
import tf_agents.trajectories.time_step as ts

import hyperparameter_search
import policies
import policies.saved_policy
import reinforcement_learning
import variational_mdp
import variational_action_discretizer
import wasserstein_mdp
import reinforcement_learning.environments
from layers.encoders import EncodingType

FLAGS = flags.FLAGS
default_flags = FLAGS.flag_values_dict()


def generate_network_components(params, name='', wasserstein_networks=False):
    if hasattr(tf.nn, params["activation"]):
        activation = getattr(tf.nn, params["activation"])
    elif hasattr(tfb, params["activation"]):
        activation = getattr(tfb, params["activation"])()
    else:
        other_activations = {
            'smooth_elu': lambda x: tf.nn.softplus(2. * x + 2.) / 2. - 1.,
            'SmoothELU': tfb.Chain([tfb.Shift(-1.), tfb.Scale(.5), tfb.Softplus(), tfb.Shift(2.), tfb.Scale(2.)])
        }
        activation = other_activations.get(
            params["activation"],
            ValueError("activation {} unknown".format(params["activation"])))

    component_names = ['encoder', 'transition', 'label_transition', 'reward', 'decoder', 'discrete_policy',
                       'state_encoder_pre_processing', 'state_decoder_pre_processing']
    wasserstein_component_names = ['steady_state', 'local_transition_loss']

    if wasserstein_networks:
        component_names += wasserstein_component_names
    network_components = []

    if name != '':
        name += '_'

    for component_name in component_names:
        x = Sequential(name="{}{}_network_body".format(name, component_name))

        if params["global_network_layers"] is not None:
            params[component_name + "_layers"] = params["global_network_layers"]

        for i, units in enumerate(params[component_name + '_layers']):
            x.add(Dense(
                units,
                activation=activation,
                name="{}{}_{}".format(name, component_name, i)))
        network_components.append(x)

    return namedtuple("{}AEArchitecture".format('V' if not wasserstein_networks else 'W'),
                      component_names)(*network_components)


def generate_vae_name(params):
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
    if not params['action_discretizer'] or params['full_vae_optimization'] or params['decompose_training']:
        vae_name = 'vae_LS{}_ER{}-decay={:g}-min={:g}_KLA{}-' \
                   'growth={:g}_TD{:.2f}-{:.2f}_activation={}_lr={:g}_seed={:d}'.format(
            params['latent_size'],
            params['entropy_regularizer_scale_factor'],
            params['entropy_regularizer_decay_rate'],
            params['entropy_regularizer_scale_factor_min_value'],
            params['kl_annealing_scale_factor'],
            params['kl_annealing_growth_rate'],
            params['state_encoder_temperature'],
            params['state_prior_temperature'],
            params['activation'],
            params['learning_rate'],
            int(params['seed']))
    if params['action_discretizer']:
        if vae_name != '':
            base_model_name = vae_name
        vae_name = os.path.join(
            base_model_name,
            os.path.split(params['policy_path'])[-1],
            'action_discretizer',
            'LA{}_ER{}-decay={:g}-min={:g}_KLA{}-growth={:g}_TD{:.2f}-{:.2f}'.format(
                params['number_of_discrete_actions'],
                params['entropy_regularizer_scale_factor'] * params['action_entropy_regularizer_scaling'],
                params['entropy_regularizer_decay_rate'],
                params['entropy_regularizer_scale_factor_min_value'],
                params['kl_annealing_scale_factor'],
                params['kl_annealing_growth_rate'],
                params['action_encoder_temperature'],
                params['latent_policy_temperature'],
            )
        )
    if params['prioritized_experience_replay']:
        vae_name += '_PER-P_exp={:g}-WIS_exponent={:g}-WIS_growth={:g}'.format(
            params['priority_exponent'],
            params['importance_sampling_exponent'],
            params['importance_sampling_exponent_growth_rate'])
        if params['buckets_based_priority']:
            vae_name += '_buckets_based'
        else:
            vae_name += '_loss_based'
    if params['max_state_decoder_variance'] > 0:
        vae_name += '_max_state_decoder_variance={:g}'.format(params['max_state_decoder_variance'])
    if params['epsilon_greedy'] > 0:
        vae_name += '_epsilon_greedy={:g}-decay={:g}'.format(params['epsilon_greedy'],
                                                             params['epsilon_greedy_decay_rate'])
    if params['marginal_entropy_regularizer_ratio'] > 0:
        vae_name += '_marginal_state_entropy_ratio={:g}'.format(params['marginal_entropy_regularizer_ratio'])
    if params['time_stacked_states'] > 1:
        vae_name += '_time_stacked_states={}'.format(params['time_stacked_states'])

    additional_parameters = [
        'one_output_per_action',
        # 'full_vae_optimization',
        # 'relaxed_state_encoding',
        'full_covariance',
        'latent_policy',
        'decompose_training',
    ]
    nb_additional_params = sum(
        map(lambda x: params[x], additional_parameters))
    if nb_additional_params > 0:
        vae_name += ('_params={}' + '-{}' * (nb_additional_params - 1)).format(
            *filter(lambda x: params[x], additional_parameters))
    if not params['label_transition_function']:
        vae_name += '_no_label_net'

    return vae_name


def generate_wae_name(params, wasserstein_regularizer: wasserstein_mdp.WassersteinRegularizerScaleFactor):
    base_model_name = ''

    if params['policy_path'][-1] == os.path.sep:
        params['policy_path'] = params['policy_path'][:-1]

    wae_name = 'wae_LS{}_TD{:.2f}-{:.2f}_activ={}_opt={}_lr={:g}_seed={:d}' \
               '_ER={:g}_decay={:g}' \
               '_SR={:g}_GP={:g}' \
               '_TL={:g}_GP={:g}_n_critic={:d}' \
               '_encoding_type={}'.format(
        params['latent_size'],
        params['state_encoder_temperature'],
        params['state_prior_temperature'],
        params['activation'],
        params['optimizer'],
        params['learning_rate'],
        int(params['seed']),
        params['entropy_regularizer_scale_factor'],
        params['entropy_regularizer_decay_rate'],
        wasserstein_regularizer.stationary.scaling,
        wasserstein_regularizer.stationary.gradient_penalty_multiplier,
        wasserstein_regularizer.local_transition_loss.scaling,
        wasserstein_regularizer.local_transition_loss.gradient_penalty_multiplier,
        params['n_critic'],
        params['state_encoder_type'])
    if params['wasserstein_optimizer'] is not None:
        wae_name += '_wopt={}_lr={:g}'.format(
            params['wasserstein_optimizer'], params['wasserstein_learning_rate'])
    if params['squared_wasserstein']:
        wae_name += '_W2'
    if not params['trainable_prior']:
        wae_name += '_prior_fixed'
    if params['action_discretizer']:
        if wae_name != '':
            base_model_name = wae_name
        wae_name = os.path.join(
            base_model_name,
            os.path.split(params['policy_path'])[-1],
            'action_discretizer',
            'LA{}_ER{}_TD{:.2f}-{:.2f}'.format(
                params['number_of_discrete_actions'],
                params['entropy_regularizer_scale_factor'] * params['action_entropy_regularizer_scaling'],
                params['action_encoder_temperature'],
                params['latent_policy_temperature'],
                str(params['policy_based_decoding']))
        )
    if params['policy_based_decoding']:
        wae_name += '_policy_based_decoding'
        if params['enforce_upper_bound']:
            wae_name += '_UB'

    if params['deterministic_state_embedding']:
        wae_name += '_deterministic_embedding'
    else:
        wae_name += '_stochastic_embedding'

    if params['prioritized_experience_replay']:
        wae_name += '_PER-P_exp={:g}-WIS_exponent={:g}-WIS_growth={:g}'.format(
            params['priority_exponent'],
            params['importance_sampling_exponent'],
            params['importance_sampling_exponent_growth_rate'])
        if params['buckets_based_priority']:
            wae_name += '_buckets_based'
        else:
            wae_name += '_loss_based'
    if params['max_state_decoder_variance'] > 0:
        wae_name += '_max_state_decoder_variance={:g}'.format(params['max_state_decoder_variance'])
    if params['epsilon_greedy'] > 0:
        wae_name += '_epsilon_greedy={:g}-decay={:g}'.format(params['epsilon_greedy'],
                                                             params['epsilon_greedy_decay_rate'])
    if params['marginal_entropy_regularizer_ratio'] > 0:
        wae_name += '_marginal_state_entropy_ratio={:g}'.format(params['marginal_entropy_regularizer_ratio'])
    if params['time_stacked_states'] > 1:
        wae_name += '_time_stacked_states={}'.format(params['time_stacked_states'])

    additional_parameters = [
        'one_output_per_action',
        'latent_policy',
    ]
    nb_additional_params = sum(
        map(lambda x: params[x], additional_parameters))
    if nb_additional_params > 0:
        wae_name += ('_params={}' + '-{}' * (nb_additional_params - 1)).format(
            *filter(lambda x: params[x], additional_parameters))
    if not params['label_transition_function']:
        wae_name += '_no_label_net'

    return wae_name


def get_environment_specs(
        environment_suite,
        environment_name: str,
        discrete_action_space: bool,
        time_stacked_states: int = 1
):
    if time_stacked_states > 1:
        environment = tf_py_environment.TFPyEnvironment(
            tf_agents.environments.parallel_py_environment.ParallelPyEnvironment(
                [lambda: HistoryWrapper(
                    env=environment_suite.load(environment_name),
                    history_length=time_stacked_states)]))
    else:
        environment = tf_py_environment.TFPyEnvironment(
            tf_agents.environments.parallel_py_environment.ParallelPyEnvironment(
                [lambda: environment_suite.load(environment_name)]))

    if time_stacked_states > 1:
        label_shape = reinforcement_learning.labeling_functions[environment_name](
            environment.reset().observation[:, -1, ...]).shape[1:]
    else:
        label_shape = reinforcement_learning.labeling_functions[environment_name](
            environment.reset().observation).shape[1:]

    state_shape, action_shape, reward_shape, label_shape = (
        shape if shape != () else (1,) for shape in [
        environment.observation_spec().shape,
        environment.action_spec().shape,
        environment.time_step_spec().reward.shape,
        label_shape])

    time_step_spec = tensor_spec.from_spec(environment.time_step_spec())
    if time_stacked_states > 1:
        observation_spec = specs.BoundedTensorSpec(
            shape=time_step_spec.observation.shape[1:],  # remove the time dimension
            dtype=time_step_spec.observation.dtype,
            name=time_step_spec.observation.name,
            minimum=time_step_spec.observation.minimum,
            maximum=time_step_spec.observation.maximum)
        time_step_spec = ts.TimeStep(
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            observation=observation_spec)

    action_spec = tensor_spec.from_spec(environment.action_spec())
    if discrete_action_space:
        # one hot encoding
        action_shape = (environment.action_spec().maximum + 1,)

    environment.close()
    del environment

    return namedtuple(
        typename='EnvironmentSpecs',
        field_names=['state_shape', 'action_shape', 'reward_shape', 'label_shape', 'time_step_spec', 'action_spec'])(
        state_shape, action_shape, reward_shape, label_shape, time_step_spec, action_spec)


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()

    # set seed
    seed = params['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if params['hyperparameter_search']:
        hyperparameter_search.search(
            fixed_parameters=params,
            num_steps=params['max_steps'],
            study_name=params['environment'] + '_seed={}'.format(params['seed']),
            n_trials=params['hyperparameter_search_trials'],
            wall_time=None if params['wall_time'] == '.' else params['wall_time'])
        return 0

    def check_missing_argument(name: str):
        if params[name] == '':
            raise RuntimeError('Missing argument: --{}'.format(name))

    if params['collect_steps_per_iteration'] <= 0:
        params['collect_steps_per_iteration'] = params['batch_size'] // 8

    environment_name = params['environment']
    if params['env_suite'] != '':
        try:
            environment_suite = importlib.import_module('tf_agents.environments.' + params['env_suite'])
        except BaseException as err:
            serr = str(err)
            print("An error occurred when loading the module '" + params['env_suite'] + "': " + serr)
    else:
        environment_suite = None

    specs = get_environment_specs(
        environment_suite=environment_suite,
        environment_name=environment_name,
        discrete_action_space=params['latent_policy'] and not params['action_discretizer'],
        time_stacked_states=params['time_stacked_states'])

    state_shape, action_shape, reward_shape, label_shape, time_step_spec, action_spec = \
        specs.state_shape, specs.action_shape, specs.reward_shape, specs.label_shape, \
        specs.time_step_spec, specs.action_spec

    state_encoder_temperature = params['state_encoder_temperature']
    state_prior_temperature = params['state_prior_temperature']
    if params['action_encoder_temperature'] < 0.:
        if params['action_discretizer']:
            params['action_encoder_temperature'] = 1. / (params['number_of_discrete_actions'] - 1)
        else:
            params['action_encoder_temperature'] = 0.99
    if params['latent_policy_temperature'] < 0.:
        if params['action_discretizer']:
            params['latent_policy_temperature'] = params['action_encoder_temperature'] / 1.5
        else:
            params['latent_policy_temperature'] = 2. / (3 * (action_shape[0] - 1))
    if state_encoder_temperature < 0:
        params['state_encoder_temperature'] = 2. / 3.
    if state_prior_temperature < 0:
        params['state_prior_temperature'] = 1. / 2.

    batch_size = params['batch_size']
    mixture_components = params['mixture_components']
    latent_state_size = params['latent_size']  # depends on the number of bits reserved for labels

    vae_name = generate_vae_name(params)

    if params['reward_lower_bound'] is None or params['reward_upper_bound'] is None:
        reward_bounds = None
    else:
        reward_bounds = (params['reward_lower_bound'], params['reward_upper_bound'])

    def build_vae_model():
        if params['load_vae'] == '':
            network = generate_network_components(params, name='state')
            return variational_mdp.VariationalMarkovDecisionProcess(
                state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
                encoder_network=network.encoder,
                state_encoder_pre_processing_network=(network.state_encoder_pre_processing
                                                      if params['state_encoder_pre_processing_network'] else None),
                state_decoder_pre_processing_network=(network.state_decoder_pre_processing
                                                      if params['state_decoder_pre_processing_network'] else None),
                transition_network=network.transition,
                label_transition_network=(network.label_transition if params['label_transition_function'] else None),
                reward_network=network.reward,
                decoder_network=network.decoder,
                latent_policy_network=(network.discrete_policy if params['latent_policy'] else None),
                time_stacked_states=params['time_stacked_states'] > 1,
                latent_state_size=latent_state_size,
                mixture_components=mixture_components,
                encoder_temperature=state_encoder_temperature,
                prior_temperature=state_prior_temperature,
                encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
                prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
                entropy_regularizer_scale_factor=params['entropy_regularizer_scale_factor'],
                entropy_regularizer_decay_rate=params['entropy_regularizer_decay_rate'],
                entropy_regularizer_scale_factor_min_value=params['entropy_regularizer_scale_factor_min_value'],
                marginal_entropy_regularizer_ratio=params['marginal_entropy_regularizer_ratio'],
                kl_scale_factor=params['kl_annealing_scale_factor'],
                kl_annealing_growth_rate=params['kl_annealing_growth_rate'],
                multivariate_normal_full_covariance=params['full_covariance'],
                max_decoder_variance=(
                    None if params['max_state_decoder_variance'] == 0. else params['max_state_decoder_variance']
                ),
                full_optimization=not params['decompose_training'] and params['latent_policy'],
                importance_sampling_exponent=params['importance_sampling_exponent'],
                importance_sampling_exponent_growth_rate=params['importance_sampling_exponent_growth_rate'],
                evaluation_window_size=params['evaluation_window_size'],
                reward_bounds=reward_bounds, )
        else:
            vae = variational_mdp.load(params['load_vae'])
            vae.encoder_temperature = state_encoder_temperature
            vae.prior_temperature = state_prior_temperature
            return vae

    def build_action_discretizer_vae_model(vae_mdp_model, full_optimization=True):
        if params['full_vae_optimization'] and params['load_vae'] != '':
            vae = variational_action_discretizer.load(params['load_vae'], full_optimization=True)
        else:
            network = generate_network_components(params, name='action')
            vae = variational_action_discretizer.VariationalActionDiscretizer(
                vae_mdp=vae_mdp_model,
                number_of_discrete_actions=params['number_of_discrete_actions'],
                action_encoder_network=network.encoder,
                transition_network=network.transition,
                action_label_transition_network=(
                    network.label_transition if params['label_transition_function'] else None),
                reward_network=network.reward, action_decoder_network=network.decoder,
                latent_policy_network=network.discrete_policy,
                encoder_temperature=params['action_encoder_temperature'],
                prior_temperature=params['latent_policy_temperature'],
                encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
                prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
                one_output_per_action=params['one_output_per_action'],
                relaxed_state_encoding=params['relaxed_state_encoding'],
                full_optimization=full_optimization,
                reconstruction_mixture_components=(
                    mixture_components if params['action_mixture_components'] == 0
                    else params['action_mixture_components']
                ),
                action_entropy_regularizer_scaling=params["action_entropy_regularizer_scaling"],
            )
            vae.kl_scale_factor = params['kl_annealing_scale_factor']
            vae.kl_growth_rate = params['kl_annealing_growth_rate']
            vae.entropy_regularizer_scale_factor = params['entropy_regularizer_scale_factor']
            vae.entropy_regularizer_decay_rate = params['entropy_regularizer_decay_rate']
        return vae

    if not params['wae']:
        models = [build_vae_model()]
        if params['action_discretizer']:
            if not params['decompose_training']:
                models[0] = build_action_discretizer_vae_model(
                    models[0], full_optimization=params['full_vae_optimization'])
            else:
                models.append(build_action_discretizer_vae_model(models[0], full_optimization=False))
        else:
            if params['decompose_training']:
                models.append(models[0])

        optimizer = getattr(tf.optimizers, params['optimizer'])(learning_rate=params['learning_rate'])
    else:
        wasserstein_regularizer_scale_factor = wasserstein_mdp.WassersteinRegularizerScaleFactor(
            global_scaling=params['global_wasserstein_regularizer_scale_factor'],
            global_gradient_penalty_multiplier=params["global_gradient_penalty_scale_factor"],
            steady_state_scaling=params["steady_state_wasserstein_regularizer_scale_factor"],
            steady_state_gradient_penalty_multiplier=params["steady_state_gradient_penalty_multiplier"],
            local_transition_loss_scaling=params["local_transition_loss_regularizer_scale_factor"],
            local_transition_loss_gradient_penalty_multiplier=params[
                "local_transition_loss_gradient_penalty_multiplier"],
        )
        vae_name = generate_wae_name(params=params, wasserstein_regularizer=wasserstein_regularizer_scale_factor)
        autoencoder_optimizer = getattr(tf.optimizers, params['optimizer'])(learning_rate=params['learning_rate'])
        if params['wasserstein_optimizer'] is None:
            wasserstein_optimizer = autoencoder_optimizer
        else:
            wasserstein_optimizer = getattr(tf.optimizers, params['wasserstein_optimizer'])(
                learning_rate=params['wasserstein_learning_rate'])
        optimizer = [autoencoder_optimizer, wasserstein_optimizer]
        network = generate_network_components(params, wasserstein_networks=True)
        action_network = generate_network_components(params, name='action')
        wae_mdp = wasserstein_mdp.WassersteinMarkovDecisionProcess(
            state_shape=state_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
            label_shape=label_shape,
            discretize_action_space=params['action_discretizer'],
            state_encoder_network=network.encoder,
            action_encoder_network=action_network.encoder if not params['policy_based_decoding'] else None,
            policy_based_decoding=params['policy_based_decoding'],
            action_decoder_network=action_network.decoder,
            transition_network=network.transition,
            reward_network=network.reward,
            decoder_network=network.decoder,
            latent_policy_network=network.discrete_policy,
            steady_state_lipschitz_network=network.steady_state,
            transition_loss_lipschitz_network=network.local_transition_loss,
            latent_state_size=latent_state_size,
            number_of_discrete_actions=params['number_of_discrete_actions'],
            state_encoder_pre_processing_network=(network.state_encoder_pre_processing
                                                  if params['state_encoder_pre_processing_network'] else None),
            state_decoder_pre_processing_network=(network.state_decoder_pre_processing
                                                  if params['state_decoder_pre_processing_network'] else None),
            time_stacked_states=params['time_stacked_states'] > 1,
            state_encoder_temperature=state_encoder_temperature,
            state_prior_temperature=state_prior_temperature,
            action_encoder_temperature=params['action_encoder_temperature'],
            latent_policy_temperature=params['latent_policy_temperature'],
            wasserstein_regularizer_scale_factor=wasserstein_regularizer_scale_factor,
            encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
            prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
            importance_sampling_exponent=params['importance_sampling_exponent'],
            importance_sampling_exponent_growth_rate=params['importance_sampling_exponent_growth_rate'],
            evaluation_window_size=params['evaluation_window_size'],
            reward_bounds=reward_bounds,
            entropy_regularizer_scale_factor=params['entropy_regularizer_scale_factor'],
            entropy_regularizer_decay_rate=params['entropy_regularizer_decay_rate'],
            entropy_regularizer_scale_factor_min_value=params["entropy_regularizer_scale_factor_min_value"],
            action_entropy_regularizer_scaling=params["action_entropy_regularizer_scaling"],
            enforce_upper_bound=params['enforce_upper_bound'],
            squared_wasserstein=params['squared_wasserstein'],
            n_critic=params['n_critic'],
            trainable_prior=params['trainable_prior'],
            state_encoder_type={
                'autoregressive': EncodingType.AUTOREGRESSIVE,
                'lstm': EncodingType.LSTM,
                'independent': EncodingType.INDEPENDENT}[params['state_encoder_type']],
            deterministic_state_embedding=params['deterministic_state_embedding'],
        )
        models = [wae_mdp]
    step = tf.Variable(0, trainable=False, dtype=tf.int64)

    for phase, vae_mdp_model in enumerate(models):
        checkpoint_directory = os.path.join(
            params['save_dir'], 'saves', environment_name, 'training_checkpoints', vae_name)
        if params['checkpoint']:
            print("checkpoint path:", checkpoint_directory)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vae_mdp_model, step=step)
            manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_directory, max_to_keep=1)
        else:
            checkpoint = manager = None

        if phase == 1 and not params['action_discretizer'] and params['latent_policy']:
            vae_mdp_model.latent_policy_training_phase = True

        policy = policies.saved_policy.SavedTFPolicy(params['policy_path'], time_step_spec, action_spec)

        if params['logs']:
            # initialize logs
            train_log_dir = os.path.join(params['logdir'], environment_name, vae_name)
            print('log path:', train_log_dir)
            if not os.path.exists(train_log_dir):
                os.makedirs(train_log_dir)
            with open(os.path.join(train_log_dir, 'parameters.json'), 'w+') as fp:
                json.dump(params, fp)

            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            with train_summary_writer.as_default():
                hyperparameters = [
                    tf.convert_to_tensor([k, str(v)])
                    for k, v in {
                        key: value for key, value in params.items()
                        if key not in default_flags
                    }.items()
                ]
                tf.summary.text('hyperparameters', tf.stack(hyperparameters), step=step)
                tf.summary.text('tf version', tf.__version__, step=step)
                tf.summary.text('tf_agent version', tf_agents.__version__, step=step)
                tf.summary.text('tf probability version', tfp.__version__, step=step)
                tf.summary.text('python version', sys.version, step=step)

                try:
                    import git
                    repo = git.Repo('.')
                    tf.summary.text('git head', str(repo.head.commit), step=step)
                except Exception as exc:
                    print(exc)
        else:
            train_summary_writer = None

        vae_mdp_model.train_from_policy(
            policy=policy,
            environment_suite=environment_suite,
            environment_seed=params['seed'],
            env_name=environment_name,
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            epsilon_greedy=params['epsilon_greedy'] if phase == 0 else 0.,
            epsilon_greedy_decay_rate=params['epsilon_greedy_decay_rate'],
            batch_size=batch_size, optimizer=optimizer, checkpoint=checkpoint,
            manager=manager,
            train_summary_writer=train_summary_writer,
            start_annealing_step=(
                params['start_annealing_step'] + params['max_steps'] // 2
                if phase == 1 and params['action_discretizer'] else
                params['start_annealing_step']),
            reset_kl_scale_factor=(
                params['kl_annealing_scale_factor'] if phase == 1 and (
                        params['action_discretizer'] or
                        params['latent_policy']) else None),
            reset_entropy_regularizer=(
                params['entropy_regularizer_scale_factor'] if phase == 1 and (
                        params['action_discretizer'] or
                        params['latent_policy']) else None),
            training_steps=(
                params['max_steps'] if not params['decompose_training'] or phase == 1
                else params['max_steps'] // 2),
            display_progressbar=params['display_progressbar'],
            save_directory=params['save_dir'] if params['checkpoint'] else None,
            parallel_environments=params['parallel_env'] > 1,
            num_parallel_environments=params['parallel_env'],
            eval_steps=int(1e3) if not params['do_not_eval'] else 0,
            eval_and_save_model_interval=params['evaluation_interval'],
            policy_evaluation_num_episodes=(
                0 if not (params['action_discretizer'] or params['latent_policy'])
                     or (phase == 0 and len(models) > 1) else params['num_eval_episodes']),
            policy_evaluation_env_name=params['policy_environment'],
            annealing_period=params['annealing_period'],
            aggressive_training=params['aggressive_training'],
            initial_collect_steps=params['initial_collect_steps'],
            discrete_action_space=(
                    not params['action_discretizer'] and params['latent_policy']),
            use_prioritized_replay_buffer=params['prioritized_experience_replay'],
            priority_exponent=params['priority_exponent'],
            buckets_based_priorities=params['buckets_based_priority'],
            collect_steps_per_iteration=params['collect_steps_per_iteration'],
            wall_time=params['wall_time'] if params['wall_time'] != '.' else None,
            memory_limit=params['memory'] if params['memory'] > 0. else None,
            local_losses_evaluation=params['local_losses_evaluation'],
            local_losses_eval_steps=params['local_losses_evaluation_steps'],
            local_losses_eval_replay_buffer_size=params['local_losses_replay_buffer_size'],
            local_losses_reward_scaling=reinforcement_learning.reward_scaling.get(environment_name, 1.),
            embed_video_evaluation=params['generate_videos'],
            environment_perturbation=params['environment_perturbation'],
            recursive_environment_perturbation=params['recursive_environment_perturbation'])

    return 0


if __name__ == '__main__':
    flags.DEFINE_integer(
        "batch_size",
        default=128,
        help="Batch size.")
    flags.DEFINE_integer(
        "mixture_components",
        default=1,
        help="Number of gaussian mixture components used to model the reconstruction distributions.")
    flags.DEFINE_integer(
        "action_mixture_components",
        default=0,
        help="Number of gaussian mixture components used to model the action reconstruction distribution (optional). "
             "If not "
             "set, all mixture distributions take the same value obtained via --mixture_components.")
    flags.DEFINE_bool(
        "full_covariance",
        default=False,
        help="If set, the states and rewards reconstruction distributions will use a full covariance matrix instead of"
             "a diagonal matrix."
    )
    flags.DEFINE_string(
        "activation",
        default="relu",
        help="Activation function for all hidden layers.")
    flags.DEFINE_multi_integer(
        "global_network_layers",
        default=None,
        help="If provided, set the number of units per layer of all neural networks components."
    )
    flags.DEFINE_integer(
        "latent_size",
        default=12,
        help='Number of bits used for the discrete latent state space.')
    flags.DEFINE_float(
        "max_state_decoder_variance",
        default="0.",
        help='Maximum variance allowed for the state decoder.'
    )
    flags.DEFINE_float(
        "action_encoder_temperature",
        default=-1.,
        help="Temperature of the relaxation of the discrete encoder distribution."
    )
    flags.DEFINE_float(
        "latent_policy_temperature",
        default=-1.,
        help="Temperature of relaxation of the discrete prior distribution over latent actions "
             "(i.e., the latent policy to be distilled)."
    )
    flags.DEFINE_float(
        "state_encoder_temperature",
        default=-1.,
        help="Temperature of the binary concrete relaxation encoder distribution over latent states."
    )
    flags.DEFINE_float(
        "state_prior_temperature",
        default=-1.,
        help="Temperature of the binary concrete relaxation prior distribution over latent states."
    )
    flags.DEFINE_bool(
        "latent_policy",
        default=False,
        help="If set, VAEs for state discretization will learn an abstraction of the input policy conditioned on"
             "latent states."
             "Only works for environment with discrete actions."
             "Remark: action discretizer VAEs always learn a latent policy."
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
        "entropy_regularizer_scale_factor",
        default=0.,
        help="Entropy regularizer scale factor."
    )
    flags.DEFINE_float(
        "entropy_regularizer_decay_rate",
        default=0.,
        help="Decay rate of the scale factor of the entropy regularizer."
    )
    flags.DEFINE_float(
        "entropy_regularizer_scale_factor_min_value",
        default=0.,
        help="Minimum value that can take the scale factor of the entropy regularizer."
    )
    flags.DEFINE_float(
        "marginal_entropy_regularizer_ratio",
        default=0.,
        lower_bound=0.,
        upper_bound=0.5,
        help="Indicates the ratio of the entropy regularizer focusing on enforcing a high marginal state encoder "
             "entropy "
             "(experimental)."
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
    flags.DEFINE_string(
        "logdir",
        default="log",
        help="logs directory"
    )
    flags.DEFINE_bool(
        "display_progressbar",
        default=False,
        help="Display progressbar."
    )
    flags.DEFINE_bool(
        "action_discretizer",
        default=False,
        help="If set, the (continuous) action space of the environment is also discretized."
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
    flags.DEFINE_integer(
        "num_eval_episodes",
        default=30,
        help='Number of episodes to achieve during the latent policy evaluation.'
    )
    flags.DEFINE_bool(
        "full_vae_optimization",
        default=True,
        help='Set whether the ELBO is optimized over the whole VAE or if the optimization is only focused on the'
             'state or action discretizer part of the VAE.'
    )
    flags.DEFINE_bool(
        'relaxed_state_encoding',
        default=True,
        help='Use a relaxed encoding of states to optimize the action discretizer part of the VAE.'
    )
    flags.DEFINE_integer(
        "number_of_discrete_actions",
        default=16,
        help='Number of discrete actions per latent state to learn.'
    )
    flags.DEFINE_string(
        "load_vae",
        default='',
        help='Path of a (trained) VAE model to load (saved via the tf.saved_model function).'
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
        "label_transition_layers",
        default=[256, 256],
        help='Number of units to use for each layer of the label transition network.'
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
        help="Name of the training environment."
    )
    flags.DEFINE_string(
        "env_suite",
        default='suite_gym',
        help='Name of the tf_agents environment suite.'
    )
    flags.DEFINE_string(
        "policy_environment",
        default=None,
        help='Name of the environment used for latent policy evaluation.'
             'Default behavior is to use the training environment.'
    )
    flags.DEFINE_integer(
        "parallel_env",
        default=1,
        help='Number of parallel environments to be used during training.'
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
    flags.DEFINE_integer(
        'seed', help='set seed', default=42
    )
    flags.DEFINE_bool(
        'logs',
        default=True,
        help="Enable logging training metrics to the logs directory."
    )
    flags.DEFINE_bool(
        'checkpoint',
        default=True,
        help='Enable to save/load checkpoints to/from the save directory.'
    )
    flags.DEFINE_float(
        'epsilon_greedy',
        default=0.,
        help='Epsilon value used for training the model via epsilon-greedy with the input policy.'
    )
    flags.DEFINE_float(
        'epsilon_greedy_decay_rate',
        default=5e-6,
        help='Decay rate of the epsilon parameter'
    )
    flags.DEFINE_bool(
        'decompose_training',
        default=False,
        help='Decompose the VAE training in two phases: 1) state space abstraction, 2) action space + policy '
             'abstraction. '
    )
    flags.DEFINE_bool(
        'prioritized_experience_replay',
        default=False,
        help='Use a prioritized experience replay buffer'
    )
    flags.DEFINE_float(
        'priority_exponent',
        default=.6,
        help='Exponent parameter for the priority experience replay'
    )
    flags.DEFINE_float(
        'importance_sampling_exponent',
        default=0.4,
        help='Exponent parameter of the importance sampling weights used with the prioritized experience replay buffer'
    )
    flags.DEFINE_float(
        'importance_sampling_exponent_growth_rate',
        default=1e-5,
        help='Growth rate used for annealing the weighted importance sampling exponent parameter when using a '
             'prioritized '
             'experience replay buffer.'
    )
    flags.DEFINE_bool(
        'buckets_based_priority',
        default=True,
        help='If set, prioritized replay buffers use a bucket-based priority scheme (where each bucket '
             'corresponds '
             'to a discrete latent state). If not, the loss is used '
    )
    flags.DEFINE_integer(
        'collect_steps_per_iteration',
        help='Collect steps per iteration. If the provided value is <= 0, then collect_steps is set to batch_size / 8',
        default=0
    )
    flags.DEFINE_bool(
        'hyperparameter_search',
        help='Perform a hyperparameter search with Optuna. If --action_discretizer is set, uses the flag'
             '--number_of_discrete_actions as the maximum number of discrete actions to consider.',
        default=False
    )
    flags.DEFINE_integer(
        'hyperparameter_search_trials',
        help='Number of trials for the hyperparameter search',
        default=1
    )
    flags.DEFINE_bool(
        'prune_trials',
        help='Whether to allow for pruning trials during hyperparameter search or not',
        default=False,
    )
    flags.DEFINE_integer(
        'evaluation_window_size',
        help="Size of the evaluation window, i.e., the number of evaluation values to be averaged for computing the"
             "final score. These values might either correspond to the best or the last values obtained during"
             "the evaluation or the model.",
        default=5
    )
    flags.DEFINE_string(
        'wall_time',
        help='(optional) walltime, in the format %H:%M:%S',
        default='.')
    flags.DEFINE_float(
        'memory',
        help='(optional) physical memory limit (in gb)',
        default=-1.)
    flags.DEFINE_integer(
        'time_stacked_states',
        help='If > 1, then the specified last observations of the environment are stacked to form the state to be '
             'processed by the VAE',
        default=1)
    flags.DEFINE_bool(
        'state_encoder_pre_processing_network',
        help="Whether to add a pre-processing network before encoding states in the architecture of the VAE",
        default=False,
    )
    flags.DEFINE_multi_integer(
        "state_encoder_pre_processing_layers",
        default=[256, 256],
        help='Number of units to use for each layer of the state encoder pre-processing network.'
    )
    flags.DEFINE_bool(
        'state_decoder_pre_processing_network',
        help="Whether to add a pre-processing network before decoding states in the architecture of the VAE",
        default=False,
    )
    flags.DEFINE_multi_integer(
        "state_decoder_pre_processing_layers",
        default=[256, 256],
        help='Number of units to use for each layer of the state decoder pre-processing network.'
    )
    flags.DEFINE_string(
        "optimizer",
        default='Adam',
        help='Optimizer name (see tf.optimizers).'
    )
    flags.DEFINE_float(
        'learning_rate',
        default=1e-4,
        help='Learning rate for the optimizer.'
    )
    flags.DEFINE_bool(
        'local_losses_evaluation',
        default=False,
        help='Whether to estimate local losses during evaluation or not.'
    )
    flags.DEFINE_integer(
        'local_losses_evaluation_steps',
        default=int(3e4),
        help='Number of steps to perform to estimate the local losses'
    )
    flags.DEFINE_integer(
        'local_losses_replay_buffer_size',
        default=int(1e5),
        help='Size of the replay buffer used to estimate the local losses'
    )
    flags.DEFINE_integer(
        'evaluation_interval',
        default=int(1e4),
        help='Number of training steps to perform between the evaluations.'
    )
    flags.DEFINE_bool(
        'label_transition_function',
        default=True,
        help='Whether to use a label transition distribution for the transition function or not.'
    )
    flags.DEFINE_float(
        'action_entropy_regularizer_scaling',
        default=1.,
        help="Scale factor of the action entropy regularizer."
    )
    flags.DEFINE_float(
        "reward_upper_bound",
        default=None,
        help='maximum values that rewards can have'
    )
    flags.DEFINE_float(
        "reward_lower_bound",
        default=None,
        help='minimum values that rewards can have'
    )
    flags.DEFINE_bool(
        "generate_videos",
        default=False,
        help="whether to generate videos during the latent policy evaluation or not."
    )
    flags.DEFINE_bool(
        "wae",
        default=False,
        help='abstract the environment and distill the input policy via a Wasserstein Autoencoder.'
    ),
    flags.DEFINE_float(
        "global_wasserstein_regularizer_scale_factor",
        default=10.,
        help='default Wasserstein regularizer scale factor used when the dedicated one is not provided.'
    ),
    flags.DEFINE_float(
        "global_gradient_penalty_scale_factor",
        default=10.,
        help='default gradient penalty scale factor used when the dedicated one is not provided.'
    )
    flags.DEFINE_float(
        "steady_state_wasserstein_regularizer_scale_factor",
        default=None,
        # required=False,
        help='Scale factor of the Wasserstein regularizer of the steady state distribution.'
    )
    flags.DEFINE_float(
        "steady_state_gradient_penalty_multiplier",
        default=None,
        # required=False,
        help="Multiplier of the gradient penalty for the steady-state Lipschitz function."
    )
    flags.DEFINE_multi_integer(
        "steady_state_layers",
        default=[256, 256],
        help="Number of units to use for each layer of the network representing the Lipschitz function for the "
             "steady-state Wasserstein regularizer."
    )
    flags.DEFINE_float(
        "local_transition_loss_regularizer_scale_factor",
        default=None,
        # required=False,
        help='Scale factor of the local loss regularizer.'
    )
    flags.DEFINE_float(
        "local_transition_loss_gradient_penalty_multiplier",
        default=None,
        # required=False,
        help="Multiplier of the gradient penalty for the local loss Lipschitz function."
    )
    flags.DEFINE_multi_integer(
        "local_transition_loss_layers",
        default=[256, 256],
        help="Number of units to use for each layer of the network representing the Lipschitz function for the "
             "local loss regularizer."
    )
    flags.DEFINE_string(
        "wasserstein_optimizer",
        default='Adam',
        help='Optimizer name for the Wasserstein regularizers (see tf.optimizers).'
    )
    flags.DEFINE_float(
        'wasserstein_learning_rate',
        default=1e-4,
        help='Learning rate for the optimizer of the Wasserstein regularizers.'
    )
    flags.DEFINE_bool(
        'policy_based_decoding',
        default=False,
        help='Whether to use the latent policy instead of an action encoding/decoding scheme'
             ' to learn to reconstruct the action.'
    )
    flags.DEFINE_bool(
        'enforce_upper_bound',
        default=False,
        help="Whether to set an upper bound on the (2-)Wasserstein distance "
             "or not when using a nondeterministic decoder."
    )
    flags.DEFINE_bool(
        'squared_wasserstein',
        default=False,
        help='Whether to optimize the squared instead of the simple Wasserstein distance or not.'
    )
    flags.DEFINE_integer(
        'n_critic',
        default=5,
        help='Number of critic (Wasserstein networks) updates to perform before updating the auto-encoders components.'
    )
    flags.DEFINE_bool(
        'trainable_prior',
        default=False,
        help='Whether to allow for training the latent steady state distribution or not.',
    )
    flags.DEFINE_enum(
        'state_encoder_type',
        'autoregressive',
        ['autoregressive', 'lstm', 'independent'],
        'State encoder type, defining which technique to use to encode states.'
    )
    flags.DEFINE_bool(
        'deterministic_state_embedding',
        default=True,
        help='Whether to use the mode of the probabilistic encoder to deploy the policy in the original environment or'
             'not. If not, the probabilistic state encoding will be used.'
    )
    flags.DEFINE_float(
        'environment_perturbation',
        lower_bound=0.,
        upper_bound=1.,
        help="Environment perturbation to enforce an ergodic episodic RL process (see Huang et al. 2020).",
        default=.75
    )
    flags.DEFINE_bool(
        'recursive_environment_perturbation',
        help='Whether to apply recursive perturbations to the environment to enforce an ergodic episodic RL process.',
        default=True
    )

    FLAGS = flags.FLAGS

    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, main))
