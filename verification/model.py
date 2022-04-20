from collections import namedtuple
from typing import Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.util.deprecation import deprecated
from tf_agents.typing.types import Float

from verification import binary_latent_space

tfd = tfp.distributions


class TransitionFunction:
    """
    Class representing an MDP transition function where probabilities are encoded via
    a transition matrix (specifically, via a SparseTensor).

    Args:
        transition_matrix: the SparseTensor representing the probability transition matrix;
                           expected size: [S, A, S], where S is the number of states and A the number of actions of
                           the MDP.
        backup_transition_function: backup transition function (mapping state-action pairs to a probability
                                    distribution) that will be used if a state-action pair entry doesn't exist in the
                                    transition matrix.
        assert_distribution: whether to check if the transition matrix is sound or not (the probabilities of
                             each state-action pair entry should sum up to one).
        split_label_from_latent_space: if set, the 'prob' method obtained via __call__ will require 2 input arguments,
                                       in binary: the label of the next latent state, followed by the representation of
                                       the next latent state without the bits of the next latent state.
                                       Concretely, this means the next latent state can be retrieved via:
                                       next_latent_state = tf.concat([label, next_latent_state_no_label], axis=-1)
    """

    def __init__(
            self,
            transition_matrix: tf.sparse.SparseTensor,
            backup_transition_function: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None,
            assert_distribution: bool = False,
            split_label_from_latent_space: bool = False
    ):
        self.num_states = transition_matrix.dense_shape[0]
        self.latent_state_size = tf.cast(
            tf.math.log(tf.cast(self.num_states, tf.float32)) / tf.math.log(2.),
            tf.int32)
        self.num_actions = transition_matrix.dense_shape[1]
        self.transitions = transition_matrix
        self.split_label_from_latent_space = split_label_from_latent_space
        self.enabled_actions = tf.cast(
            tf.sparse.reduce_sum(self.transitions, axis=-1, output_is_sparse=True),
            dtype=tf.bool)
        if assert_distribution:
            epsilon = 1e-6
            state_action_pairs = tf.sparse.reduce_sum(self.transitions, axis=-1, output_is_sparse=True)
            tf.assert_less(tf.abs(1. - state_action_pairs.values), epsilon)
        self.backup_transition_function = backup_transition_function
        self.latent_state_space = binary_latent_space(self.latent_state_size, dtype=tf.float32)

    def __call__(self, latent_state: tf.Tensor, latent_action: tf.Tensor):
        """
        Gives the transition distribution formed by the input latent state and latent action.
        The returned distribution is a namedtuple with the following method 'prob':
            prob(next_latent_state, full_latent_state=False), which returns either the probability of
                - the next_latent_state, given in binary;
                - the full matrix entry P(.|latent_state, latent_action) if full_latent_state is set, i.e.,
                  the probability of going to all the next latent states, given the input latent state and actions.
        Args:
            latent_state: the latent state, given in binary (expected size: log_2 S)
            latent_action: the latent action, given in one-hot (expected size: A)

        Returns: the distribution P(. | latent_state, latent_action)
        """
        latent_state = tf.cast(latent_state, tf.int32)
        state = tf.reduce_sum(latent_state * 2 ** tf.range(self.latent_state_size), axis=-1)
        action = tf.argmax(latent_action, axis=-1)
        if self.backup_transition_function is None:
            backup_transition_fn = lambda *args: namedtuple('zeros_backup', ['prob'])(lambda *x: 0.)
        else:
            backup_transition_fn = self.backup_transition_function

        @tf.function
        def _get_prob_value(transition):
            latent_state, state, action, next_latent_state_no_label, next_label = transition
            if self.split_label_from_latent_space:
                next_latent_state = tf.cast(tf.concat([next_label, next_latent_state_no_label], axis=-1), tf.int32)
            else:
                next_latent_state = next_latent_state_no_label
            next_state = tf.reduce_sum(
                tf.cast(next_latent_state, tf.float32) * 2. ** tf.range(self.latent_state_size, dtype=tf.float32),
                axis=-1)

            # check if the action has been visited in the given state during the transition sparse tensor construction
            action_is_enabled = tf.squeeze(
                tf.sparse.to_dense(
                    tf.sparse.slice(self.enabled_actions, [state, action], [1, 1])))

            if action_is_enabled:
                probs = tf.squeeze(tf.sparse.slice(self.transitions, [state, action, next_state], [1, 1, 1]).values)
                # if the entry of the slice is empty, then the probability is 0
                return 0. if tf.equal(tf.size(probs), 0) else probs
            # if not, then use a backup transition function to retrieve the probability distribution for [state, action]
            else:
                if self.split_label_from_latent_space:
                    next_latent_state = [tf.expand_dims(next_label, axis=0),
                                         tf.expand_dims(next_latent_state_no_label, axis=0)]
                else:
                    next_latent_state = [tf.expand_dims(next_latent_state, 0)]
                return tf.squeeze(
                    backup_transition_fn(
                        tf.expand_dims(latent_state, axis=0),
                        tf.expand_dims(tf.one_hot(action, depth=tf.cast(self.num_actions, tf.int32)), axis=0)
                    ).prob(*next_latent_state))

        @tf.function
        def _prob(*value, **kwargs):
            if self.split_label_from_latent_space:
                next_label, next_latent_state_no_label = value
                next_latent_state = tf.concat(next_label, next_latent_state_no_label, axis=-1)
            else:
                next_label = next_latent_state_no_label = next_latent_state = value[0]
            full_latent_state_space = kwargs.get('full_latent_state_space', False)
            if full_latent_state_space:
                if self.split_label_from_latent_space:
                    return _probs_row(next_label, next_latent_state_no_label)
                else:
                    return _probs_row(next_latent_state)
            else:
                return tf.map_fn(
                    fn=_get_prob_value,
                    elems=(latent_state, state, action, next_latent_state_no_label, next_label),
                    fn_output_signature=tf.float32)

        @tf.function
        def _probs_row(*value):
            action_is_enabled = tf.squeeze(tf.sparse.to_dense(tf.sparse.slice(
                self.enabled_actions, [state[0, ...], action[0, ...]], [1, 1])))
            if action_is_enabled:
                return tf.squeeze(tf.sparse.to_dense(tf.sparse.slice(
                    self.transitions, [state[0, ...], action[0, ...], 0], [1, 1, self.num_states])))
            else:
                _distr = backup_transition_fn(
                    latent_state,
                    tf.one_hot(action, depth=tf.cast(self.num_actions, tf.int32)))
                try:
                    return _distr.prob(*value, full_latent_state_space=True)
                except TypeError:
                    return _distr.prob(*value)

        return namedtuple('next_state_transition_distribution', ['prob'])(_prob)


class TransitionFunctionCopy(TransitionFunction):
    """
    Copies a transition function to a transition matrix (encoded through a SparseTensor)

    Args:
        transition_function: mapping from state-action pairs to distributions, implementing the prob method
        epsilon: probability tolerance error
    """

    def __init__(
            self,
            num_states: int,
            num_actions: int,
            transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
            epsilon: float = 1e-6,
            split_label_from_latent_space: bool = False,
            atomic_prop_dims: Optional[int] = None,
    ):
        if split_label_from_latent_space and atomic_prop_dims is None:
            raise ValueError("You need provide atomic_prop_dims if split_label_from_latent_space is set.")
        latent_state_size = tf.cast(
            tf.math.log(tf.cast(num_states, tf.float32)) / tf.math.log(2.),
            tf.int32)
        latent_state_space = binary_latent_space(latent_state_size, dtype=tf.float32)
        action_space = tf.one_hot(tf.range(num_actions), depth=tf.cast(num_actions, dtype=tf.int32))
        tiled_action_space = tf.tile(action_space, multiples=[num_states, 1])
        state_action_space = tf.repeat(latent_state_space, repeats=num_actions, axis=0)

        @tf.function
        def _get_sparse_entry(latent_state):
            state = tf.tile(tf.expand_dims(latent_state, 0), [num_states * num_actions, 1])
            action = tiled_action_space
            _distr = transition_function(state, action)
            if split_label_from_latent_space:
                next_latent_states = [
                    state_action_space[..., :atomic_prop_dims],
                    state_action_space[..., atomic_prop_dims:]]
            else:
                next_latent_states = [state_action_space]
            probs = _distr.prob(*next_latent_states)
            # sparsify
            probs = tf.where(
                condition=probs > epsilon,
                x=probs,
                y=tf.zeros_like(probs))
            # normalize
            probs = probs / tf.reduce_sum(probs)
            indices = tf.where(tf.math.not_equal(probs, tf.zeros_like(probs)))
            actions = tf.argmax(tf.gather_nd(action, indices), axis=-1)
            values = tf.gather_nd(probs, indices)
            return tf.sparse.SparseTensor(
                indices=tf.concat([tf.expand_dims(actions, -1), indices], axis=-1),
                values=values,
                dense_shape=[num_actions, num_states])

        transitions = tf.map_fn(
            fn=_get_sparse_entry,
            # elems=(tf.repeat(latent_state_space, repeats=num_actions, axis=0),
            #        tf.tile(action_space, multiples=[num_states, 1])),
            elems=latent_state_space,
            fn_output_signature=tf.SparseTensorSpec(shape=[num_actions, num_states]))

        #  with tf.device('/CPU:0'):
        #      transitions = tf.sparse.reorder(transitions)

        super(TransitionFunctionCopy, self).__init__(
            transition_matrix=transitions,
            split_label_from_latent_space=split_label_from_latent_space)
        self.atomic_prop_dims = atomic_prop_dims


class RewardFunctionCopy:

    def __init__(
            self,
            num_states: int,
            num_actions: int,
            reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Float],
            transition_function: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None,
            split_label_from_latent_space: bool = False,
            epsilon: float = 1e-6,
    ):
        self.latent_state_size = tf.cast(
            tf.math.log(tf.cast(num_states, tf.float32)) / tf.math.log(2.),
            tf.int32)
        latent_state_space = binary_latent_space(self.latent_state_size, dtype=tf.float32)
        action_space = tf.one_hot(tf.range(num_actions), depth=tf.cast(num_actions, dtype=tf.int32))
        tiled_action_space = tf.tile(action_space, multiples=[num_states, 1])
        state_action_space = tf.repeat(latent_state_space, repeats=num_actions, axis=0)

        @tf.function
        def _get_sparse_entry(latent_state):
            state = tf.tile(tf.expand_dims(latent_state, 0), [num_states * num_actions, 1])
            action = tiled_action_space
            rewards = tf.squeeze(reward_function(state, action, state_action_space))
            if transition_function is not None:
                _distr = transition_function(state, action)
                if split_label_from_latent_space:
                    next_latent_states = [
                        state_action_space[..., :atomic_prop_dims],
                        state_action_space[..., atomic_prop_dims:]]
                else:
                    next_latent_states = [state_action_space]
                probs = _distr.prob(*next_latent_states)
                rewards = tf.where(
                    condition=probs > epsilon,
                    x=rewards,
                    y=tf.zeros_like(rewards))
            indices = tf.where(tf.greater(tf.abs(rewards), epsilon))
            values = tf.gather_nd(rewards, indices)
            actions = tf.argmax(tf.gather_nd(action, indices), axis=-1)
            return tf.sparse.SparseTensor(
                indices=tf.concat([tf.expand_dims(actions, -1), indices], axis=-1),
                values=values,
                dense_shape=[num_actions, num_states])

        self.transitions = tf.map_fn(
            fn=_get_sparse_entry,
            elems=latent_state_space,
            fn_output_signature=tf.SparseTensorSpec(shape=[num_actions, num_states]))
        self.num_states = num_states

    @tf.function
    def __call__(self, latent_state: tf.Tensor, latent_action: tf.Tensor, *args, **kwargs):
        """
        Get the full entry for R(latent_state, latent_action, .)
        latent_state, latent_actions are assumed to be batched, but only the first batch element is used to provide
        the entry (the others are ignored).

        Args:
            latent_state: batched binary latent state
            latent_action: batched one-hot encoded action

        Returns: the rewards for all the next states, given latent state and latent action.
        """
        latent_state = tf.cast(latent_state[0, ...], tf.int32)
        state = tf.reduce_sum(latent_state[0, ...] * 2 ** tf.range(self.latent_state_size), axis=-1)
        action = tf.argmax(latent_action[0, ...], axis=-1)

        return tf.squeeze(tf.sparse.to_dense(tf.sparse.slice(
            self.transitions, [state, action, 0], [1, 1, self.num_states])))


class TransitionFrequencyEstimator(TransitionFunction):
    def __init__(
            self,
            latent_states: tf.Tensor,
            latent_actions: tf.Tensor,
            next_latent_states: tf.Tensor,
            backup_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
            assert_distribution: bool = False,
            split_label_from_latent_space: bool = False
    ):
        latent_states = tf.cast(latent_states, dtype=tf.int32)
        next_latent_states = tf.cast(next_latent_states, dtype=tf.int32)
        latent_state_size = tf.shape(latent_states)[1]  # first axis is batch, second is latent state size
        num_states = 2 ** latent_state_size
        num_actions = tf.shape(latent_actions)[1]  # first axis is batch, second is a one-hot vector

        @tf.function
        def compute_transition_counter():
            states = tf.reduce_sum(latent_states * 2 ** tf.range(latent_state_size), axis=-1)
            actions = tf.cast(tf.argmax(latent_actions, axis=-1), dtype=tf.int32)
            next_states = tf.reduce_sum(next_latent_states * 2 ** tf.range(latent_state_size), axis=-1)

            # flat transition indices
            transitions = states * num_actions * num_states + actions * num_states + next_states
            transitions, _, count = tf.unique_with_counts(transitions)
            transitions = tf.stack([transitions // (num_states * num_actions),  # state index
                                    (transitions // num_states) % num_actions,  # action index
                                    transitions % num_states],  # next state index
                                   axis=-1)
            transitions = tf.cast(transitions, dtype=tf.int64)
            transition_counter = tf.sparse.SparseTensor(
                indices=transitions,
                values=tf.cast(count, tf.float32),
                dense_shape=(num_states, num_actions, num_states))
            return tf.sparse.reorder(transition_counter)

        transition_counter = compute_transition_counter()
        probs = tf.Variable(tf.cast(transition_counter.values, dtype=tf.float32), trainable=False)
        i = tf.Variable(0, trainable=False)
        j = tf.Variable(0, trainable=False)

        @tf.function
        def compute_transition_probabilities(
                transition_counter: tf.sparse.SparseTensor,
                probs: tf.Variable,
                i: tf.Variable,
                j: tf.Variable):
            state_action_pair_counter = tf.sparse.reduce_sum(transition_counter, axis=-1, output_is_sparse=True)
            indices = transition_counter.indices[..., :-1]
            while i < tf.shape(probs)[0]:
                if tf.reduce_all(indices[i] == state_action_pair_counter.indices[j], axis=-1):
                    probs[i].assign(tf.cast(transition_counter.values[i], dtype=tf.float32) /
                                    tf.cast(state_action_pair_counter.values[j], dtype=tf.float32))
                    i.assign_add(1)
                else:
                    j.assign_add(1)  # works only if indices are ordered

            transition_tensor = tf.sparse.SparseTensor(
                indices=transition_counter.indices,
                values=probs,
                dense_shape=(num_states, num_actions, num_states))

            return tf.sparse.reorder(transition_tensor)

        super(TransitionFrequencyEstimator, self).__init__(
            transition_matrix=compute_transition_probabilities(transition_counter, probs, i, j),
            backup_transition_function=backup_transition_function,
            assert_distribution=assert_distribution,
            split_label_from_latent_space=split_label_from_latent_space)

    def merge(self, other: TransitionFunction, epsilon: Float = 1e-12):
        a_idx = self.transitions.indices
        b_idx = other.transitions.indices
        to_retain = tf.Variable(
            tf.cast(tf.ones_like(other.transitions.values), tf.bool),
            trainable=False)
        tf.print(b_idx, tf.shape(b_idx))
        import time
        time.sleep(10)

        @tf.function
        def _to_retain(i, j, to_retain):
            i = tf.minimum(i, tf.shape(a_idx)[0] - 1)
            j = tf.minimum(j, tf.shape(b_idx)[0] - 1)
            tf.print("i", i, "--", "j", j)
            s, a, s_prime = a_idx[i, 0], a_idx[i, 1], a_idx[i, 2]
            s_, a_, s_prime_ = b_idx[j, 0], b_idx[j, 1], b_idx[j, 2]
            tf.print("a_idx:", a_idx[i, ...])
            tf.print("b_idx:", b_idx[j, ...])
            if tf.geq(
                s * self.num_actions * self.num_states + a * self.num_states + s_prime,
                _s * self.num_actions * self.num_states + _a * self.num_states + _s_prime
            ):
                if tf.reduce_all(tf.equal(a_idx[i, ...], b_idx[j, ...])):
                    to_retain[j].assign(False)
                    i += 1
                j += 1
            else:
                i += 1
            tf.print("retain?", to_retain[j])
            return i, j
        
        with tf.device('/CPU:0'):
            i, j = tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)
            tf.while_loop(
                cond=lambda i, j: tf.math.logical_and(
                    tf.less(i, tf.shape(a_idx)[0]),
                    tf.less(j, tf.shape(b_idx)[0])),
                body=lambda i, j: _to_retain(i, j, to_retain),
                loop_vars=[i, j],
                swap_memory=True,)

            self.transitions = tf.sparse.add(
                self.transitions,
                tf.sparse.retain(other.transitions, to_retain),
                threshold=epsilon,)

class TransitionFnDecorator:
    """
    Decorates a latent transition function P with a new prob function so that:
    P_new(s' | s, a) = P(l(s'), [s' without label] | s, a)
    with l being the labeling function

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
