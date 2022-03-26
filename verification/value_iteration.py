import enum
import time
from typing import Callable, Optional, Dict, Union

import tensorflow as tf
from tensorflow.debugging import Assert
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float, Int, Bool

from util.io.dataset_generator import is_reset_state
from verification import binary_latent_space


class Error(enum.Enum):
    ABSOLUTE = enum.auto()
    RELATIVE = enum.auto()


error = {
    'absolute': Error.ABSOLUTE,
    'relative': Error.RELATIVE,
}


@tf.function
def value_iteration(
        latent_state_size: Int,
        num_actions: Int,
        transition_fn: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        reward_fn: Callable[[tf.Tensor,tf.Tensor, tf.Tensor], tf.Tensor],
        gamma: Float,
        policy: Optional[Callable[[tf.Tensor], tfd.OneHotCategorical]] = None,
        error_type: Union[str, Error] = Error.RELATIVE,
        epsilon: Float = 1e-6,
        is_reset_state_test_fn: Optional[Callable[[tf.Tensor], Bool]] = None,
        episodic_return: bool = True,
        debug: bool = False,
) -> Dict[str, Float]:
    """
    Iteratively compute the value of (i.e., the expected return obtained from running an input policy from) each state up
    to a certain precision, depending on the error between two consecutive iterations.

    Args:
        latent_state_size: size of the state space in binary, i.e., number of bits used to represent the state space
        num_actions: size of the action space
        transition_fn: function mapping each state-action pair to a distribution over (binary encoded) states
        reward_fn: function mapping each transition to a tensor containing the reward
                   obtained by going through this transition.
        gamma: discount factor
        policy: function mapping each (binary encoded) state to a distribution over (one-hot) actions.
                If not provided, the action yielding the best values is chosen at each step.
        error_type: error type (absolute or relative)
        epsilon: error between two consecutive iterations
        is_reset_state_test_fn: function testing whether the input state is a reset (or null) state or not.
                                If provided,
                                - Rewards obtained from transitions issued from states marked as 'True' are undiscounted
                                - Rewards obtained by transitioning from/to a state marked as 'True' are null
        episodic_return: Whether to estimate the finite-horizon episodic return or infinite horizon return.
                         If True, is_reset_state_fn has to be provided. In that case, values obtained by transitioning
                         to a reset state will be ignored.
        debug: whether to display iteration error and time metrics 
    """
    if error_type not in [Error.RELATIVE, Error.ABSOLUTE]:
        error_type = error[error_type]
    num_states = 2 ** latent_state_size
    values = tf.zeros(num_states, dtype=tf.float32)
    delta = tf.constant(float('inf'))
    state_space = binary_latent_space(latent_state_size, dtype=tf.float32)
    action_space = tf.one_hot(indices=tf.range(num_actions), depth=tf.cast(num_actions, tf.int32), dtype=tf.float32)

    def q_s(state: Float, values: tf.Tensor):
        return tf.transpose(
            tf.map_fn(
                fn=lambda action: compute_next_q_value(
                    state=state,
                    action=action,
                    values=values,
                    transition_fn=transition_fn,
                    reward_fn=reward_fn,
                    gamma=gamma,
                    state_space=state_space,
                    is_reset_state_test_fn=is_reset_state_test_fn,
                    episodic_return=episodic_return),
                elems=action_space,
                fn_output_signature=tf.float32))

    log_error = tf.math.log(epsilon) / tf.math.log(10.)
    
    @tf.function
    def update_values(values: tf.Tensor, _):
        q_values = tf.map_fn(
            fn=lambda state: q_s(state, values),
            elems=state_space,
            fn_output_signature=tf.float32)
        next_values = tf.map_fn(
            fn=lambda state: compute_next_value(
                state=state,
                q_values=q_values,
                num_actions=num_actions,
                policy=policy,),
            elems=state_space)

        if error_type is Error.ABSOLUTE:
            delta = tf.reduce_max(tf.abs(next_values - values))
        else:
            delta = tf.reduce_max(
                tf.abs(1. - 
                    tf.where(
                        condition=values == next_values,
                        x=tf.ones_like(values),
                        y=values / next_values)))
        
        if debug:
            tf.print("current error:", delta)
            progress = tf.math.log(delta) * 100. / (tf.math.log(10.) * log_error)
            tf.print("progress (%):", progress)
        
        return next_values, delta
    
    start_time = time.time() if debug else 0
    values, _ = tf.while_loop(
        cond=lambda _, _delta: tf.greater_equal(_delta, epsilon),
        body=update_values,
        loop_vars=[values, delta],)

    if debug:
        print("time to perform value iteration:", time.time() - start_time)

    return values


def compute_next_value(
        state: tf.Tensor,
        q_values: tf.Tensor,
        num_actions: int,
        policy: Optional[Callable[[Int], tfd.OneHotCategorical]] = None,
) -> Float:
    """

    Args:
        state: unbatched (single) binary state; expected shape: [S]
               where S is the number of bits used to represent each individual state
        q_values: tensor containing the Q-values of the current step; expected shape: [2**S, A]
               where 2**S is the size of the state space, and A is the size of the action space
        num_actions: number of actions, i.e., A.
        policy: function mapping each (binary encoded) state to a distribution over (one-hot) actions.
                If not provided, then the values of the best action is chosen.
    Returns: the next value of the input state (shape=()).
    """
    v = q_values[tf.reduce_sum(tf.cast(state, dtype=tf.int32) * 2 ** tf.range(tf.shape(state)[0]), axis=-1), ...]

    if policy is not None:
        return tf.reduce_sum(
            # the latent policy is assumed batched
            policy(tf.expand_dims(state, 0)).probs_parameter()[0, ...] * v,
            axis=0)
    else:
        return tf.math.maximum(v, axis=0)


def compute_next_q_value(
        state: Int,
        action: Int,
        values: tf.Tensor,
        transition_fn: Callable[[Int, Int], tfd.Distribution],
        reward_fn: Callable[[Int, Int, Int], tf.Tensor],
        gamma: Float,
        state_space: Optional[tf.Tensor] = None,
        is_reset_state_test_fn: Optional[Callable[[tf.Tensor], Bool]] = None,
        episodic_return: bool = True,
) -> Float:
    """
    Compute the next-step Q-value of the input state-action pair.

    Args:
        state: unbatched binary state; expected shape: [S]
               where S is the number of bits used to represent each individual state
        action: unbatched one-hot encoded action; expected shape: [A]
                where A is the size of the action space
        values: tensor containing the value of each state; expected shape: [2**S]
        transition_fn: function mapping each state-action pair to a distribution over (binary encoded) states
        reward_fn: function mapping each transition to a tensor containing the reward
                   obtained by going through this transition.
        gamma: discount factor
        state_space: full binary-encoded state space
        is_reset_state_test_fn: function testing whether the input state is a reset (or null) state or not.
                                If provided,
                                - Rewards obtained from transitions issued from states marked as 'True' are undiscounted
                                - Rewards obtained by transitioning from/to a state marked as 'True' are null
        episodic_return: Whether to estimate the finite-horizon episodic return or infinite horizon return.
                         If True, is_reset_state_fn has to be provided. In that case, values obtained by transitioning
                         to a reset state will be ignored.

    Returns: the Q-value of the input state-action pair
    """
    num_states = 2 ** tf.shape(state)[0]
    tile = lambda t: tf.tile(
        tf.expand_dims(t, 0),
        [num_states, 1])
    if state_space is None:
        next_states = binary_latent_space(num_states, dtype=tf.float32)
    else:
        next_states = state_space
    tiled_state = tile(state)
    tiled_action = tile(action)
    reward = reward_fn(tiled_state, tiled_action, next_states)[..., 0]  # unidimensional reward

    if is_reset_state_test_fn is not None:
        # next values issued from a reset state are undiscounted
        gamma = gamma ** (1. - tf.cast(is_reset_state_test_fn(state), tf.float32))
        # rewards obtained by transitioning from/to a reset states are ignored
        reward = tf.where(
            condition=tf.logical_or(is_reset_state_test_fn(state), is_reset_state_test_fn(next_states)),
            x=tf.zeros_like(reward),
            y=reward)
        if episodic_return:
            # next values obtained by transitioning from a non-reset state to a reset state are ignored
            values = tf.where(
                condition=tf.logical_and(
                    tf.logical_not(is_reset_state_test_fn(state)),
                    is_reset_state_test_fn(next_states)),
                x=tf.zeros_like(values),
                y=values)
    
    return tf.reduce_sum(
        transition_fn(
            tiled_state, tiled_action
        ).prob(next_states, full_latent_state_space=True) *
        (reward + gamma * values),)
