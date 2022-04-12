from collections import namedtuple
from typing import Optional, Callable, Dict, Any
import time

import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import Float
from tf_agents.utils import common
import tensorflow_probability.python.distributions as tfd

from reinforcement_learning import metrics
from util.io.dataset_generator import map_rl_trajectory_to_vae_input, \
    ergodic_batched_labeling_function, is_reset_state
from reinforcement_learning.environments.latent_environment import LatentEmbeddingTFEnvironmentWrapper
from verification import binary_latent_space
from verification.model import TransitionFrequencyEstimator, TransitionFunctionCopy, RewardFunctionCopy
from verification.value_iteration import value_iteration


def estimate_local_losses_from_samples(
        environment: TFPyEnvironment,
        latent_policy: tf_policy.TFPolicy,
        steps: int,
        latent_state_size: int,
        number_of_discrete_actions: int,
        state_embedding_function: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        labeling_function: Callable[[tf.Tensor], tf.Tensor],
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution] = None,
        estimate_transition_function_from_samples: bool = False,
        assert_transition_distribution: bool = False,
        fill_in_replay_buffer: bool = True,
        replay_buffer_max_frames: int = int(1e5),
        reward_scaling: Optional[float] = 1.,
        atomic_prop_dims: Optional[int] = None,
        probabilistic_state_embedding: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None,
        estimate_value_difference: bool = False,
        gamma: float = 0.99,
        epsilon: float = 1e-6
):
    """
    Estimates reward and probability local losses from samples.

    Args:
        environment: batched TFPyEnvironment
        latent_policy: a policy sets over latent states and producing (one-hot) latent actions
        steps: number of environment steps used to perform the estimation
        latent_state_size: binary size of the latent space
        number_of_discrete_actions: number of discrete actions
        state_embedding_function: mapping from real states and labels to binary states (of type int32)
        action_embedding_function: mapping from real states and discrete actions (given in base 10 with type int32)
                                   to real actions.
                                   Important: latent actions are here assumed to be given in base 10 and not in
                                   one-hot since environments generally process discrete actions represented in
                                   base 10 by convention.
        latent_reward_function: mapping from (binary) latent state, (one-hot) latent actions and (binary) next latent
                                state to rewards. Latent states and actions are assumed to be of type int32 while
                                returned reward type is float32.
        labeling_function: labeling function (mapping from real states to subset of atomic propositions)
        latent_transition_function: mapping from (binary) latent states and (one-hot) actions (given in int32) to
                                    a probability distribution over latent states.
                                    A probability distribution object is assumed to have a prob method (e.g.,
                                    tensorflow probability distribution objects) such that
                                    distribution.prob(label, latent_state_no_label) is the probability of the
                                    latent state given by tf.concat([label, latent_state_no_label], axis=-1)
        estimate_transition_function_from_samples: whether to estimate the latent transition function from samples
                                                   or not. If True, the latent transition function estimated this
                                                   way stores the probability matrix into a sparse tensor.
        assert_transition_distribution: whether to assert the transition function is correctly computed or not
        fill_in_replay_buffer: whether to fill in the replay buffer or not before the loss estimation.
        replay_buffer_max_frames: maximum number of frames to be contained in the replay buffer.
        atomic_prop_dims: number of atomic propositions; in other words the rightmost label shape.
        probabilistic_state_embedding: underlying (conditional) distribution of the state embedding function.
                                       If provided, this distribution will be used for estimating the local
                                       transition loss.
    Returns: a namedtuple containing an estimation of the local reward and probability losses
    """
    time_metrics = dict()
    time_metrics['start'] = time.time()

    if latent_transition_function is None and not estimate_transition_function_from_samples:
        raise ValueError('no latent transition function provided')
    # generate environment wrapper for discrete actions
    latent_environment = LatentEmbeddingTFEnvironmentWrapper(
        tf_env=environment,
        state_embedding_fn=state_embedding_function,
        action_embedding_fn=action_embedding_function,
        labeling_fn=labeling_function,
        latent_state_size=latent_state_size,
        number_of_discrete_actions=number_of_discrete_actions,
        reward_scaling=reward_scaling)
    # set the latent policy over real states
    policy = latent_environment.wrap_latent_policy(latent_policy)
    trajectory_spec = trajectory.from_transition(
        time_step=latent_environment.time_step_spec(),
        action_step=policy.policy_step_spec,
        next_time_step=latent_environment.time_step_spec(),
    )
    # replay_buffer
    replay_buffer = TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=latent_environment.batch_size,
        max_length=replay_buffer_max_frames,
        # to retrieve all the transitions when single_deterministic_pass is True
        dataset_window_shift=1,
        dataset_drop_remainder=True)
    # initialize driver
    observers = [replay_buffer.add_batch]
    if estimate_value_difference:
        observers.append(metrics.AverageDiscountedReturnMetric(
            gamma=gamma,
            batch_size=latent_environment.batch_size))
    driver = DynamicStepDriver(
        env=latent_environment,
        policy=policy,
        num_steps=replay_buffer_max_frames if fill_in_replay_buffer else steps,
        observers=observers)
    driver.run = common.function(driver.run)
    # collect environment steps
    driver.run()

    time_metrics['collect'] = time.time() - time_metrics['start']

    # retrieve dataset from the replay buffer
    generator = lambda trajectory, _: map_rl_trajectory_to_vae_input(
        trajectory=trajectory,
        include_latent_states=True,
        discrete_action=True,
        num_discrete_actions=number_of_discrete_actions,
        labeling_function=ergodic_batched_labeling_function(labeling_function))

    def sample_from_replay_buffer(num_transitions: Optional[int] = None, single_deterministic_pass=False):
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            num_steps=2,
            # whether to gather transitions only once or not
            single_deterministic_pass=single_deterministic_pass,
        ).map(
            map_func=generator,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(
            batch_size=num_transitions if num_transitions else replay_buffer.num_frames(),
            drop_remainder=False)
        dataset_iterator = iter(dataset)

        state, label, latent_state, latent_action, reward, next_state, next_label, next_latent_state = \
            next(dataset_iterator)

        return namedtuple(
            'ErgodicMDPTransitionSample',
            ['state', 'label', 'latent_state', 'latent_action', 'reward', 'next_state', 'next_label',
             'next_latent_state', 'batch_size'])(
            state, label, latent_state, latent_action, reward, next_state, next_label,
            next_latent_state, tf.shape(state)[0])

    time_metrics['local_reward_loss'] = time.time()

    samples = sample_from_replay_buffer(num_transitions=steps)
    (state, label, latent_state, latent_action, reward, next_state, next_label,
     next_latent_state) = (
        samples.state, samples.label, samples.latent_state, samples.latent_action, samples.reward,
        samples.next_state, samples.next_label, samples.next_latent_state)

    if atomic_prop_dims is None:
        atomic_prop_dims = latent_state_size - tf.shape(
            state_embedding_function(state, None)
        )[-1]

    next_latent_state_no_label = next_latent_state[..., atomic_prop_dims:]

    local_reward_loss = estimate_local_reward_loss(
        state, label, latent_action, reward, next_state, next_label,
        latent_reward_function, latent_state, next_latent_state,
        reward_scaling=reward_scaling)

    time_metrics['local_reward_loss'] = time.time() - time_metrics['local_reward_loss']

    if estimate_transition_function_from_samples:

        time_metrics['transition_function_estimation'] = time.time()
        _samples = sample_from_replay_buffer(single_deterministic_pass=True)
        _transition_function_estimation_num_frames = _samples.batch_size.numpy()
        empirical_latent_transition_function = TransitionFrequencyEstimator(
            _samples.latent_state, _samples.latent_action, _samples.next_latent_state,
            backup_transition_function=latent_transition_function,
            assert_distribution=assert_transition_distribution,
            split_label_from_latent_space=True)

        time_metrics['transition_function_estimation'] = time.time() - \
                                                         time_metrics['transition_function_estimation']
    else:
        empirical_latent_transition_function = None
        _transition_function_estimation_num_frames = 0
        time_metrics['transition_function_estimation'] = 0

    time_metrics['local_transition_loss'] = time.time()

    local_transition_loss = estimate_local_transition_loss(
        state, label, latent_action, next_state, next_label,
        latent_transition_function, latent_state, next_latent_state_no_label,
        probabilistic_state_embedding=probabilistic_state_embedding)

    time_metrics['local_transition_loss'] = time.time() - time_metrics['local_transition_loss']

    time_metrics['local_transition_loss_2'] = time.time()
    if empirical_latent_transition_function is not None:
        local_transition_loss_transition_fn_estimation = estimate_local_transition_loss(
            state, label, latent_action, next_state, next_label,
            empirical_latent_transition_function, latent_state, next_latent_state_no_label,
            probabilistic_state_embedding=probabilistic_state_embedding)
    else:
        local_transition_loss_transition_fn_estimation = None

    time_metrics['local_transition_loss_2'] = time.time() - time_metrics['local_transition_loss_2']

    value_diff = dict()
    if estimate_value_difference:
        transition_functions = {"latent_transition_function": latent_transition_function}
        if empirical_latent_transition_function is not None:
            transition_functions["empirical_latent_transition_function"] = empirical_latent_transition_function
        for name, transition_fn in transition_functions.items():
            time_metrics['value_diff_' + name] = time.time()
            values = compute_values(
                latent_state_size=latent_state_size,
                atomic_prop_dims=atomic_prop_dims,
                state=state,
                number_of_discrete_actions=number_of_discrete_actions,
                latent_policy=latent_policy,
                latent_transition_function=transition_fn,
                latent_reward_function=latent_reward_function,
                epsilon=epsilon,
                gamma=gamma,
                stochastic_state_embedding=(
                    lambda _state: probabilistic_state_embedding(
                        _state, ergodic_batched_labeling_function(labeling_function)(_state))
                ) if probabilistic_state_embedding else (
                    lambda _state: tfd.Independent(
                            tfd.Deterministic(loc=state_embedding_function(
                            _state, ergodic_batched_labeling_function(labeling_function)(_state))),
                        reinterpreted_batch_ndims=1)),)
            value_diff["value_diff_" + name] = tf.abs(
                observers[-1].result() - values)
            time_metrics['value_diff_' + name] = time.time() - time_metrics['value_diff_' + name]

    def print_time_metrics():
        print("Time metrics:")

        def replace(_dict: Dict[str, Any], key_dict: Dict[str, str]):
            _new_dict = dict()
            for old_key, value in _dict.items():
                new_key = key_dict.get(old_key, None)
                if new_key is not None:
                    _new_dict[new_key] = value
                else:
                    _new_dict[old_key] = value
            return _new_dict

        for _name, _time in replace(
                time_metrics,
                {
                    'collect':
                        "Fill in the Replay Buffer ({:d} frames)".format(replay_buffer_max_frames),
                    'local_reward_loss':
                        "Estimate the local reward loss function (from {:d} transitions)".format(steps),
                    'local_transition_loss':
                        "Estimate the local transition loss function (from {:d} transitions):".format(steps),
                    'transition_function_estimation':
                        "Time to build the transition function via frequency estimation"
                        "(from {:d} transitions)".format(_transition_function_estimation_num_frames),
                    'local_transition_loss_2':
                        "Time to estimate the local transition loss function via the frequency "
                        "estimated transition function (from {:d} transitions):".format(steps),
                    'value_diff_latent_transition_function': "Value difference",
                    "value_diff_empirical_latent_transition_function":
                        "Value difference (empirical latent transition function)"
                }
        ).items():
            if _name != 'start':
                print("{}: {:.3f}".format(_name, _time))

    replay_buffer.clear()

    return namedtuple(
        'LocalLossesEstimationMetrics',
        ['local_reward_loss', 'local_transition_loss', 'local_transition_loss_transition_function_estimation',
         'print_time_metrics', 'time_metrics', 'value_difference'])(
        local_reward_loss, local_transition_loss, local_transition_loss_transition_fn_estimation,
        print_time_metrics, time_metrics, value_diff)


@tf.function
def estimate_local_reward_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        reward: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        latent_state: Optional[tf.Tensor] = None,
        next_latent_state: Optional[tf.Tensor] = None,
        state_embedding_function: Optional[Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = None,
        reward_scaling: Optional[float] = 1.
):
    if latent_state is None:
        latent_state = state_embedding_function(state, label)
    if next_latent_state is None:
        next_latent_state = state_embedding_function(next_state, next_label)

    return tf.reduce_mean(reward_scaling * tf.abs(
        reward - latent_reward_function(latent_state, latent_action, next_latent_state)))


@tf.function
def estimate_local_transition_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        latent_state: Optional[tf.Tensor] = None,
        next_latent_state_no_label: Optional[tf.Tensor] = None,
        state_embedding_function: Optional[Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = None,
        probabilistic_state_embedding: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None
):
    if latent_state is None:
        latent_state = state_embedding_function(state, label)
    if next_latent_state_no_label is None:
        next_latent_state_no_label = state_embedding_function(next_state, None)

    next_label = tf.cast(next_label, tf.float32)
    next_latent_state_no_label = tf.cast(next_latent_state_no_label, tf.float32)

    if probabilistic_state_embedding:
        latent_state_space = binary_latent_space(tf.shape(latent_state)[-1], dtype=tf.float32)

        @tf.function
        def total_variation(transition):
            _latent_state, _latent_action, _next_state, _next_label = transition
            tile = lambda t: tf.tile(
                tf.expand_dims(t, 0),
                [tf.shape(latent_state_space)[0], 1])

            latent_transition_distribution = latent_transition_function(
                tile(_latent_state), tile(_latent_action))
            embedding_distribution = probabilistic_state_embedding(
                tile(_next_state), tile(_next_label))
            return .5 * tf.reduce_sum(
                tf.abs(
                    embedding_distribution.prob(latent_state_space) -
                    latent_transition_distribution.prob(
                        latent_state_space[..., :tf.shape(next_label)[-1]],
                        latent_state_space[..., tf.shape(next_label)[-1]:],
                        full_latent_state_space=True)),
                axis=0)

        return tf.reduce_mean(
            tf.map_fn(
                fn=total_variation,
                elems=[latent_state, latent_action, next_state, next_label],
                fn_output_signature=tf.float32))

    else:
        latent_transition_distribution = latent_transition_function(latent_state, latent_action)
        return tf.reduce_mean(1. - latent_transition_distribution.prob(next_label, next_latent_state_no_label))

@tf.function
def compute_values(
        latent_state_size: int,
        atomic_prop_dims: int,
        state: tf.Tensor,
        number_of_discrete_actions: int,
        latent_policy: Callable[[tf.Tensor], tfd.OneHotCategorical],
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Float],
        epsilon: Float,
        gamma: Float,
        stochastic_state_embedding: Callable[[tf.Tensor], tfd.Distribution],
        v_init: Optional[Float] = None,
):
    latent_state_space = binary_latent_space(latent_state_size)
    latent_transition_fn = lambda state, action: TransitionFnDecorator(
        next_state_distribution=latent_transition_function(state, action),
        atomic_prop_dims=atomic_prop_dims)
    latent_transition_fn = TransitionFunctionCopy(
        num_states=tf.cast(tf.pow(2, latent_state_size), dtype=tf.int32),
        num_actions=number_of_discrete_actions,
        transition_function=latent_transition_fn)
    latent_reward_function = RewardFunctionCopy(
        num_states=tf.cast(tf.pow(2, latent_state_size), dtype=tf.int32),
        num_actions=number_of_discrete_actions,
        reward_function=latent_reward_function,
        copied_transition_function=latent_transition_fn)

    p_init = stochastic_state_embedding(
        tf.tile(tf.zeros_like(state[:1, ...]), [tf.shape(latent_state_space)[0], 1])
    ).prob(latent_state_space)
    is_reset_state_test_fn = lambda latent_state: is_reset_state(latent_state, atomic_prop_dims)

    values = value_iteration(
        latent_state_size=latent_state_size,
        num_actions=number_of_discrete_actions,
        transition_fn=latent_transition_fn,
        reward_fn=latent_reward_function,
        gamma=gamma,
        policy=PolicyDecorator(latent_policy),
        epsilon=epsilon,
        is_reset_state_test_fn=is_reset_state_test_fn,
        episodic_return=tf.equal(tf.reduce_max(p_init), 1.),
        error_type='absolute',
        v_init=v_init,
        transition_matrix=tf.sparse.to_dense(latent_transition_fn.transitions),
        reward_matrix=tf.sparse.to_dense(latent_reward_function.transitions))

    if tf.equal(tf.reduce_max(p_init), 1.):
        # deterministic reset
        reset_state = stochastic_state_embedding(
            tf.tile(tf.zeros_like(state[:1, ...]), [tf.shape(latent_state_space)[0], 1])
        ).sample()
        reset_state = tf.cast(reset_state, tf.float32)

        latent_action_space = tf.one_hot(
            indices=tf.range(number_of_discrete_actions),
            depth=tf.cast(number_of_discrete_actions, tf.int32),
            dtype=tf.float32)
        
        p_init = tf.reduce_sum(
            tf.transpose(
                PolicyDecorator(latent_policy)(
                    reset_state
                ).probs_parameter()
            ) * tf.map_fn(
                fn=lambda latent_action: latent_transition_fn(
                    reset_state,
                    tf.tile(tf.expand_dims(latent_action, 0), [tf.shape(latent_state_space)[0], 1]),
                ).prob(
                    tf.cast(latent_state_space, tf.float32),
                    full_latent_state_space=True),
                elems=latent_action_space),
            axis=0) * (1. - tf.cast(is_reset_state_test_fn(latent_state_space), tf.float32))

        return tf.reduce_sum(
            p_init * values
        ) / tf.reduce_sum(p_init)
        
    else:
        return tf.reduce_sum(p_init * values, axis=0)



class TransitionFnDecorator:
    """
    Decorates a latent transition function P with a new prob function so that:
    P_new(s' | s, a) = P(l(s'), [s' without label] | s, a)
    where l is the labeling function

    Usage:
    ```python
    decorated_transition_fn = lambda state, action: TransitionFnDecorator(
        next_state_distribution=transition_fn(state, action),
        atomic_prop_dims=atomic_prop_dims)
    ```
    """

    def __init__(self, next_state_distribution, atomic_prop_dims):
        self.next_state_distribution = next_state_distribution
        self.atomic_prop_dims = atomic_prop_dims

    def prob(self, latent_state, *args, **kwargs):
        return self.next_state_distribution.prob(
            latent_state[..., :self.atomic_prop_dims],
            latent_state[..., self.atomic_prop_dims:],
            *args, **kwargs)


class PolicyDecorator:
    """
    Decorates a TFPolicy π to retrieve a mapping from states to the space of distributions over actions.
    """
    def __init__(self, policy: tf_policy.TFPolicy):
        self._policy = policy

    def __call__(self, state, *args, **kwargs):
        return self._policy.distribution(
            ts.transition(observation=state, reward=tf.zeros_like(state)), *args, **kwargs
        ).action
