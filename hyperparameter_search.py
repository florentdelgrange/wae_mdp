import logging
import math
import os
import sys
import time
import random
from typing import Optional

import tensorflow as tf
import numpy as np
import optuna
import importlib

import wasserstein_mdp
from layers.encoders import EncodingType
from policies.saved_policy import SavedTFPolicy
import reinforcement_learning
import reinforcement_learning.environments
from train import get_environment_specs, generate_network_components, generate_wae_name, initialize_summary_writer, \
    generate_vae_name, get_architecture, update_layer_params
import variational_action_discretizer
import variational_mdp


def optimize_hyperparameters(study_name, env_name, optimize_trial, storage=None, n_trials=100):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if storage is None:
        if not os.path.exists('studies'):
            os.makedirs('studies')
        storage = 'sqlite:///studies/{}.db'.format(study_name)

    sqlite_timeout = 300
    storage = optuna.storages.RDBStorage(
        storage,
        engine_kwargs={'connect_args': {'timeout': sqlite_timeout}})
    study = optuna.create_study(
        study_name=env_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize')

    return study.optimize(optimize_trial, n_trials=n_trials)


def search(
        fixed_parameters: dict,
        num_steps: int = 1e6,
        study_name='study',
        n_trials=100,
        wall_time: Optional[str] = None,
):
    start_time = time.time()

    environment_suite_name = fixed_parameters['env_suite']
    environment_name = fixed_parameters['environment']
    environment_suite = None
    try:
        environment_suite = importlib.import_module('tf_agents.environments.' + environment_suite_name)
    except BaseException as err:
        serr = str(err)
        print("An error occurred when loading the module '" + environment_suite_name + "': " + serr)

    def suggest_hyperparameters(trial):
        defaults = {}
        optimizer = 'Adam'
        learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 2e-4, 3e-4])
        adam_beta_1 = trial.suggest_categorical('beta_1', [0., 0.5, 0.9])
        adam_beta_2 = trial.suggest_categorical('beta_2', [0.9, 0.999])
        batch_size = 128
        neurons = trial.suggest_categorical('neurons', [64, 128, 256, 512])
        hidden = trial.suggest_int('hidden', 1, 3)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'smooth_elu', 'tanh'])
        entropy_regularizer_scale_factor = 0.
        entropy_regularizer_decay_rate = 0.
        action_entropy_regularizer_scaling = 0.

        # temperatures
        state_encoder_softclipping = trial.suggest_categorical('softclipping', [True, False])
        temperature_decay = trial.suggest_categorical('temperature_decay', [True, False])
        encoder_temperature_decay_rate = 1e-6
        prior_temperature_decay_rate = 2e-6
        if fixed_parameters['state_encoder_temperature'] < 0:
            state_encoder_temperature = trial.suggest_categorical(
                'state_encoder_temperature',
                [0.1, 1. / 3, .5, 2. / 3, 3. / 5, 0.99])
        else:
            state_encoder_temperature = fixed_parameters['state_encoder_temperature']
        if fixed_parameters['state_prior_temperature'] < 0:
            state_prior_temperature = trial.suggest_categorical(
                'state_prior_temperature',
                [0.1, 1. / 3, .5, 2. / 3, 3. / 4, 0.99])
        else:
            state_prior_temperature = fixed_parameters['state_prior_temperature']

        if fixed_parameters['epsilon_greedy'] > 0.:
            use_epsilon_greedy = trial.suggest_categorical('use_epsilon_greedy', [True, False])
        else:
            use_epsilon_greedy = False

        if use_epsilon_greedy:
            epsilon_greedy = .5
            epsilon_greedy_decay_rate = 1e-5
        else:
            epsilon_greedy = 0.
            epsilon_greedy_decay_rate = 0.

        wasserstein_optimizer = 'Adam'
        wasserstein_learning_rate = trial.suggest_categorical('wasserstein_learning_rate', [1e-4, 2e-4, 3e-4])

        if fixed_parameters['policy_based_decoding']:
            policy_based_decoding = trial.suggest_categorical('policy_based_decoding', [True, False])
        else:
            policy_based_decoding = False
        if policy_based_decoding:
            squared_wasserstein = True
            enforce_upper_bound = trial.suggest_categorical('enforce_upper_bound', [True, False])
        else:
            squared_wasserstein = trial.suggest_categorical('squared_wasserstein', [True, False])
            enforce_upper_bound = False

        local_transition_loss_regularizer_scale_factor = trial.suggest_categorical(
            'local_transition_loss_regularizer_scale_factor', [10., 25., 50., 75., 100.])
        steady_state_wasserstein_regularizer_scale_factor = trial.suggest_categorical(
            'steady_state_wasserstein_regularizer_scale_factor', [10., 25., 50., 75., 100.])
        global_gradient_penalty_scale_factor = trial.suggest_categorical(
            'global_gradient_penalty_scale_factor', [10, 20])
        n_critic = trial.suggest_categorical('n_critic', [5, 10, 15, 20])

        if fixed_parameters['trainable_prior']:
            trainable_prior = trial.suggest_categorical('trainable_prior', [True, False])
        else:
            trainable_prior = False

        if fixed_parameters['freeze_state_encoder_type']:
            state_encoder_type = fixed_parameters['state_encoder_type']
        else:
            state_encoder_type = trial.suggest_categorical(
                'state_encoder_type', ['autoregressive', 'lstm', 'independent', 'deterministic'])

        if state_encoder_type != 'deterministic':
            deterministic_state_embedding = trial.suggest_categorical(
                'deterministic_state_embedding', [True, False])
        else:
            deterministic_state_embedding = True

        if fixed_parameters['time_stacked_states'] > 1:
            time_stacked_states = trial.suggest_int('time_stacked_states', 1, fixed_parameters['time_stacked_states'])
        else:
            time_stacked_states = 1

        specs = get_environment_specs(
            environment_suite=environment_suite,
            environment_name=environment_name,
            discrete_action_space=not fixed_parameters['action_discretizer'],
            time_stacked_states=time_stacked_states)
        if fixed_parameters['latent_size'] <= 0:
            latent_state_size = trial.suggest_int(
                'latent_state_size', specs.label_shape[0] + 3, max(14, specs.label_shape[0] + 8))
        else:
            latent_state_size = fixed_parameters['latent_size']

        prioritized_experience_replay = fixed_parameters['prioritized_experience_replay']

        if prioritized_experience_replay:
            if fixed_parameters['collect_steps_per_iteration'] > 0:
                collect_steps_per_iteration = trial.suggest_int(
                    'prioritized_experience_replay_collect_steps_per_iteration',
                    fixed_parameters['collect_steps_per_iteration'],
                    batch_size // 8)
            else:
                collect_steps_per_iteration = batch_size // 8
            bucket_based_priorities = trial.suggest_categorical('buckets_based_priorities', [True, False])
            priority_exponent = trial.suggest_categorical(
                'priority_exponent', [1. / 3, 2. / 3, 0.99])
            importance_sampling_exponent = 0.4
            importance_sampling_exponent_growth_rate = trial.suggest_categorical('is_growth_rate', [1e-5, 7.5e-5])

            if priority_exponent == 0.:
                prioritized_experience_replay = False
        else:
            if fixed_parameters['collect_steps_per_iteration'] <= 0:
                collect_steps_per_iteration = batch_size // 8
            else:
                collect_steps_per_iteration = trial.suggest_int(
                    'uniform_replay_buffer_collect_steps_per_iteration', 1,
                    fixed_parameters['collect_steps_per_iteration'])
            # default values
            bucket_based_priorities = False
            priority_exponent = 0.
            importance_sampling_exponent = 1.
            importance_sampling_exponent_growth_rate = 1.

        if fixed_parameters['state_encoder_pre_processing_network']:
            state_encoder_pre_processing_network = trial.suggest_categorical(
                'state_encoder_pre_processing_network', [True, False])
        else:
            state_encoder_pre_processing_network = False

        if fixed_parameters['state_decoder_pre_processing_network']:
            state_decoder_pre_processing_network = trial.suggest_categorical(
                'state_decoder_pre_processing_network', [True, False])
        else:
            state_decoder_pre_processing_network = False

        if fixed_parameters['action_discretizer']:
            if fixed_parameters['number_of_discrete_actions'] <= 0:
                number_of_discrete_actions = trial.suggest_int(
                    'number_of_discrete_actions', 2, 5)
            else:
                number_of_discrete_actions = fixed_parameters['number_of_discrete_actions']
            action_temperature_base = 1. / (number_of_discrete_actions - 1)
            if fixed_parameters['action_encoder_temperature'] < 0:
                action_encoder_temperature = trial.suggest_categorical(
                    "action_encoder_temperature", [action_temperature_base, action_temperature_base / 1.5])
            else:
                action_encoder_temperature = fixed_parameters['action_encoder_temperature']
            if fixed_parameters['latent_policy_temperature'] < 0:
                latent_policy_temperature = trial.suggest_categorical(
                    "latent_policy_temperature", [action_temperature_base, action_temperature_base / 1.5])
            else:
                latent_policy_temperature = fixed_parameters['latent_policy_temperature']

        else:
            action_temperature_base = 1. / (specs.action_shape[0] - 1)
            number_of_discrete_actions = fixed_parameters['number_of_discrete_actions']
            action_encoder_temperature = -1.
            if fixed_parameters['latent_policy_temperature'] < 0:
                latent_policy_temperature = trial.suggest_categorical(
                    "latent_policy_temperature", [action_temperature_base, action_temperature_base / 1.5])
            else:
                latent_policy_temperature = fixed_parameters['latent_policy_temperature']

        if fixed_parameters['wae']:
            for attr in ['learning_rate', 'adam_beta_1', 'adam_beta_2', 'batch_size', 'collect_steps_per_iteration',
                         'latent_state_size', 'state_encoder_softclipping', 'temperature_decay',
                         'encoder_temperature_decay_rate', 'prior_temperature_decay_rate',
                         'entropy_regularizer_scale_factor', 'entropy_regularizer_decay_rate',
                         'action_entropy_regularizer_scaling', 'prioritized_experience_replay',
                         'n_critic', 'neurons', 'hidden', 'activation', 'priority_exponent',
                         'importance_sampling_exponent', 'importance_sampling_exponent_growth_rate', 'specs',
                         'bucket_based_priorities', 'epsilon_greedy', 'epsilon_greedy_decay_rate',
                         'time_stacked_states',
                         'state_encoder_pre_processing_network', 'state_decoder_pre_processing_network',
                         'optimizer', 'state_encoder_temperature', 'state_prior_temperature',
                         'action_encoder_temperature', 'latent_policy_temperature',
                         'wasserstein_optimizer', 'wasserstein_learning_rate', 'policy_based_decoding',
                         'global_gradient_penalty_scale_factor', 'local_transition_loss_regularizer_scale_factor',
                         'steady_state_wasserstein_regularizer_scale_factor',
                         'n_critic', 'squared_wasserstein', 'enforce_upper_bound', 'trainable_prior',
                         'state_encoder_type', 'deterministic_state_embedding', 'number_of_discrete_actions'
                         ]:
                defaults[attr] = locals().get(attr, 'default')
        else:
            raise ValueError("Hyperparameter search is currently only enabled for WAEs.")
        return defaults

    def optimize_trial(trial: optuna.Trial):
        hyperparameters = suggest_hyperparameters(trial)

        # set the seed according to the trial number
        seed = trial.number
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        hyperparameters['latent_size'] = hyperparameters['latent_state_size']

        print("Suggested hyperparameters")
        for key in hyperparameters.keys():
            if key != "specs":
                print("{}={}".format(key, hyperparameters[key]))

        hyperparameters['global_network_layers'] = hyperparameters['hidden'] * [hyperparameters['neurons']]
        hyperparameters['wae'] = fixed_parameters['wae']
        update_layer_params(hyperparameters)

        evaluation_window_size = fixed_parameters['evaluation_window_size']
        specs = hyperparameters['specs']
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        _params = {key: value for key, value in fixed_parameters.items()}
        for key, value in hyperparameters.items():
            _params[key] = value

        if fixed_parameters['wae']:
            wasserstein_regularizer_scale_factor = wasserstein_mdp.WassersteinRegularizerScaleFactor(
                local_transition_loss_scaling=hyperparameters['local_transition_loss_regularizer_scale_factor'],
                steady_state_scaling=hyperparameters['steady_state_wasserstein_regularizer_scale_factor'],
                global_gradient_penalty_multiplier=hyperparameters["global_gradient_penalty_scale_factor"], )

            vae_name = generate_wae_name(params=_params, wasserstein_regularizer=wasserstein_regularizer_scale_factor)

            autoencoder_optimizer = tf.optimizers.Adam(  # getattr(tf.optimizers, hyperparameters['optimizer'])(
                learning_rate=hyperparameters['learning_rate'],
                beta_1=hyperparameters['adam_beta_1'],
                beta_2=hyperparameters['adam_beta_2'])

            # getattr(tf.optimizers, hyperparameters['wasserstein_optimizer'])(
            wasserstein_optimizer = tf.optimizers.Adam(
                learning_rate=hyperparameters['wasserstein_learning_rate'],
                beta_1=hyperparameters['adam_beta_1'],
                beta_2=hyperparameters['adam_beta_2'])
            optimizer = [autoencoder_optimizer, wasserstein_optimizer]
            vae_mdp = wasserstein_mdp.WassersteinMarkovDecisionProcess(
                state_shape=specs.state_shape,
                action_shape=specs.action_shape,
                reward_shape=specs.reward_shape,
                label_shape=specs.label_shape,
                discretize_action_space=fixed_parameters['action_discretizer'],
                state_encoder_network=get_architecture(hyperparameters, 'encoder', 'state_encoder'),
                action_encoder_network=get_architecture(hyperparameters, 'encoder', 'action_encoder') \
                    if not hyperparameters['policy_based_decoding'] else None,
                action_decoder_network=get_architecture(hyperparameters, 'decoder', 'action_decoder'),
                transition_network=get_architecture(hyperparameters, 'transition'),
                reward_network=get_architecture(hyperparameters, 'reward'),
                decoder_network=get_architecture(hyperparameters, 'decoder', 'state_decoder'),
                latent_policy_network=get_architecture(hyperparameters, 'discrete_policy'),
                steady_state_lipschitz_network=get_architecture(hyperparameters, 'steady_state'),
                transition_loss_lipschitz_network=get_architecture(
                    hyperparameters, 'local_transition_loss', 'transition_loss'),
                state_encoder_pre_processing_network=(
                    get_architecture(hyperparameters, 'state_encoder_pre_processing')
                    if hyperparameters['state_encoder_pre_processing_network'] else None),
                state_decoder_pre_processing_network=(
                    get_architecture(hyperparameters, 'state_decoder_pre_processing')
                    if hyperparameters['state_decoder_pre_processing_network'] else None),
                latent_state_size=hyperparameters['latent_state_size'],
                number_of_discrete_actions=hyperparameters['number_of_discrete_actions'],
                time_stacked_states=hyperparameters['time_stacked_states'] > 1,
                state_encoder_temperature=hyperparameters['state_encoder_temperature'],
                state_prior_temperature=hyperparameters['state_prior_temperature'],
                action_encoder_temperature=hyperparameters['action_encoder_temperature'],
                latent_policy_temperature=hyperparameters['latent_policy_temperature'],
                wasserstein_regularizer_scale_factor=wasserstein_regularizer_scale_factor,
                encoder_temperature_decay_rate=hyperparameters['encoder_temperature_decay_rate'] \
                    if hyperparameters['temperature_decay'] else 0.,
                prior_temperature_decay_rate=hyperparameters['prior_temperature_decay_rate'] \
                    if hyperparameters['temperature_decay'] else 0.,
                importance_sampling_exponent=hyperparameters['importance_sampling_exponent'],
                importance_sampling_exponent_growth_rate=hyperparameters['importance_sampling_exponent_growth_rate'],
                entropy_regularizer_scale_factor=hyperparameters['entropy_regularizer_scale_factor'],
                entropy_regularizer_decay_rate=hyperparameters['entropy_regularizer_decay_rate'],
                entropy_regularizer_scale_factor_min_value=0.,
                action_entropy_regularizer_scaling=hyperparameters["action_entropy_regularizer_scaling"],
                enforce_upper_bound=hyperparameters['enforce_upper_bound'],
                squared_wasserstein=hyperparameters['squared_wasserstein'],
                n_critic=hyperparameters['n_critic'],
                trainable_prior=hyperparameters['trainable_prior'],
                state_encoder_type={
                    'autoregressive': EncodingType.AUTOREGRESSIVE,
                    'lstm': EncodingType.LSTM,
                    'independent': EncodingType.INDEPENDENT,
                    'deterministic': EncodingType.DETERMINISTIC,
                }[hyperparameters['state_encoder_type']],
                policy_based_decoding=hyperparameters['policy_based_decoding'],
                deterministic_state_embedding=hyperparameters['deterministic_state_embedding'],
                state_encoder_softclipping=hyperparameters['state_encoder_softclipping'])
        else:
            vae_name = generate_vae_name(_params)
            network = generate_network_components(hyperparameters, 'state')
            vae_mdp = variational_mdp.VariationalMarkovDecisionProcess(
                state_shape=specs.state_shape, action_shape=specs.action_shape,
                reward_shape=specs.reward_shape, label_shape=specs.label_shape,
                encoder_network=network.encoder,
                transition_network=network.transition,
                label_transition_network=network.label_transition if hyperparameters[
                    'label_transition_function'] else None,
                reward_network=network.reward, decoder_network=network.decoder,
                state_encoder_pre_processing_network=(network.state_encoder_pre_processing
                                                      if hyperparameters['state_encoder_pre_processing_network']
                                                      else None),
                state_decoder_pre_processing_network=(network.state_decoder_pre_processing
                                                      if hyperparameters['state_decoder_pre_processing_network']
                                                      else None),
                latent_policy_network=(network.discrete_policy if fixed_parameters['latent_policy'] else None),
                latent_state_size=hyperparameters['latent_state_size'],
                mixture_components=fixed_parameters['mixture_components'],
                encoder_temperature_decay_rate=0.,
                prior_temperature_decay_rate=0.,
                entropy_regularizer_scale_factor=fixed_parameters['entropy_regularizer_scale_factor'],
                entropy_regularizer_decay_rate=hyperparameters['entropy_regularizer_decay_rate'],
                entropy_regularizer_scale_factor_min_value=fixed_parameters[
                    'entropy_regularizer_scale_factor_min_value'],
                marginal_entropy_regularizer_ratio=fixed_parameters['marginal_entropy_regularizer_ratio'],
                kl_scale_factor=fixed_parameters['kl_annealing_scale_factor'],
                kl_annealing_growth_rate=hyperparameters['kl_annealing_growth_rate'],
                multivariate_normal_full_covariance=fixed_parameters['full_covariance'],
                full_optimization=True,
                importance_sampling_exponent=hyperparameters['importance_sampling_exponent'],
                importance_sampling_exponent_growth_rate=hyperparameters['importance_sampling_exponent_growth_rate'],
                evaluation_window_size=evaluation_window_size,
                evaluation_criterion=variational_mdp.EvaluationCriterion.MAX,
                time_stacked_states=hyperparameters['time_stacked_states'] > 1, )
            optimizer = getattr(tf.optimizers, hyperparameters['optimizer'])(
                learning_rate=hyperparameters['learning_rate'])

            if fixed_parameters['action_discretizer']:
                network = generate_network_components(hyperparameters, name='variational_action_discretizer')
                vae_mdp = variational_action_discretizer.VariationalActionDiscretizer(
                    vae_mdp=vae_mdp,
                    number_of_discrete_actions=hyperparameters['number_of_discrete_actions'],
                    action_encoder_network=network.encoder,
                    transition_network=network.transition,
                    action_label_transition_network=(network.label_transition
                                                     if hyperparameters['label_transition_function'] else None),
                    reward_network=network.reward, action_decoder_network=network.decoder,
                    latent_policy_network=network.discrete_policy,
                    encoder_temperature_decay_rate=0.,
                    prior_temperature_decay_rate=0.,
                    one_output_per_action=hyperparameters['one_output_per_action'],
                    relaxed_state_encoding=True,
                    full_optimization=True,
                    reconstruction_mixture_components=1,
                    action_entropy_regularizer_scaling=fixed_parameters['action_entropy_regularizer_scaling'])

        environments = vae_mdp.initialize_environments(
            environment_suite=environment_suite,
            env_name=environment_name,
            parallel_environments=fixed_parameters['parallel_env'] > 1,
            num_parallel_environments=fixed_parameters['parallel_env'],
            collect_steps_per_iteration=hyperparameters['collect_steps_per_iteration'],
            environment_seed=trial.number,
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            policy_evaluation_num_episodes=fixed_parameters['num_eval_episodes'],
            policy_evaluation_env_name=fixed_parameters['policy_environment'],
            environment_perturbation=fixed_parameters['environment_perturbation'],
            recursive_environment_perturbation=fixed_parameters['recursive_environment_perturbation'],
            enforce_no_reward_shaping=fixed_parameters['no_reward_shaping'],
            local_losses_evaluation=fixed_parameters['local_losses_evaluation'],
            local_losses_eval_steps=fixed_parameters['local_losses_evaluation_steps'],
            local_losses_eval_replay_buffer_size=fixed_parameters['local_losses_replay_buffer_size'],
            local_losses_reward_scaling=reinforcement_learning.reward_scaling.get(
                environment_name, vae_mdp._dynamic_reward_scaling))

        environment = environments.training
        policy_evaluation_driver = environments.policy_evaluation_driver

        policy = SavedTFPolicy(fixed_parameters['policy_path'], specs.time_step_spec, specs.action_spec)
        epsilon_greedy = tf.Variable(hyperparameters['epsilon_greedy'], trainable=False)
        dataset_components = vae_mdp.initialize_dataset_components(
            env=environment,
            policy=policy,
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            batch_size=hyperparameters['batch_size'],
            manager=None,
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
            priority_exponent=hyperparameters['priority_exponent'],
            bucket_based_priorities=hyperparameters['bucket_based_priorities'],
            discrete_action_space=not fixed_parameters['action_discretizer'],
            collect_steps_per_iteration=hyperparameters['collect_steps_per_iteration'],
            initial_collect_steps=int(1e4),
            replay_buffer_capacity=int(1e6),
            epsilon_greedy=epsilon_greedy)

        training_steps_per_iteration = num_steps // 100
        initial_training_steps = training_steps_per_iteration

        log_name = 'trial_number={:d}'.format(trial.number)
        train_summary_writer = initialize_summary_writer(
            _params,
            environment_name,
            os.path.join(log_name, vae_name),
            dump_params_into_json=False)

        def train_model(training_steps):
            return vae_mdp.train_from_policy(
                policy=policy,
                environment_suite=environment_suite,
                env_name=environment_name,
                labeling_function=reinforcement_learning.labeling_functions[environment_name],
                training_steps=training_steps,
                train_summary_writer=train_summary_writer if fixed_parameters['log'] else None,
                log_name=log_name,
                log_interval=fixed_parameters['log_interval'],
                use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
                bucket_based_priorities=hyperparameters['bucket_based_priorities'],
                global_step=global_step,
                optimizer=optimizer,
                eval_steps=int(1e5),
                annealing_period=fixed_parameters['annealing_period'],
                start_annealing_step=fixed_parameters['start_annealing_step'],
                eval_and_save_model_interval=training_steps_per_iteration,
                save_directory=fixed_parameters['save_dir'],
                policy_evaluation_num_episodes=30,
                environment=environment,
                dataset_components=dataset_components,
                policy_evaluation_driver=policy_evaluation_driver,
                close_at_the_end=False,
                display_progressbar=fixed_parameters['display_progressbar'],
                start_time=start_time,
                wall_time=wall_time,
                memory_limit=fixed_parameters['memory'] if fixed_parameters['memory'] > 0. else None,
                epsilon_greedy=hyperparameters['epsilon_greedy'],
                epsilon_greedy_decay_rate=hyperparameters['epsilon_greedy_decay_rate'])

        def sanity_check(score: float) -> bool:
            if math.isinf(score) or math.isnan(score):
                optuna.TrialPruned()
                return False
            else:
                return True

        try:
            result = train_model(initial_training_steps)
        except ValueError as ve:
            print(ve)
            raise optuna.TrialPruned()

        score = float(result['score'])

        if result['continue']:
            for step in range(initial_training_steps, num_steps, training_steps_per_iteration):

                try:
                    result = train_model(step + training_steps_per_iteration)
                except Exception as e:
                    print("The training has stopped prematurely due to the following error:")
                    print(e)
                    result['continue'] = False

                score = float(result['score'])
                print("Step {} intermediate score: {}".format(step + training_steps_per_iteration, score))
                result['continue'] = result['continue'] and sanity_check(score)

                # Report intermediate objective value.
                if hasattr(vae_mdp, '_last_score') and vae_mdp._last_score is not None:
                    trial.report(vae_mdp._last_score, step=step + training_steps_per_iteration)
                else:
                    trial.report(score, step=step + training_steps_per_iteration)

                # Handle pruning based on the intermediate value.
                if fixed_parameters['prune_trials'] and trial.should_prune():
                    raise optuna.TrialPruned()

                if not result['continue']:
                    break

        dataset_components.close_fn()

        if not sanity_check(score):
            raise ValueError("Study stopped due to Inf or NaN values.")

        return score

    return optimize_hyperparameters(study_name, environment_name, optimize_trial, n_trials=n_trials)
