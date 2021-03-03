import h5py
import os
import glob
import datetime
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
import tf_agents.trajectories.time_step as ts
import time


def gather_rl_observations(
        iterator,
        labeling_function,
        dataset_path: Optional['str'] = None,
        dataset_name='rl_exploration',
        scalar_rewards=True):
    """
    Writes the observations gathered through the training of an RL policy into an hdf5 dataset.
    Important: the next() call of the iterator function must yield a bash containing 3-steps of tf_agents Trajectories.
    The labeling function is defined over Trajectories observations.
    """
    data = iterator.next()[0]  # a tf_agents dataset typically returns a tuple (trajectories, information)
    states = data.observation[:, :2, :].numpy()
    actions = data.action[:, :2, :].numpy()
    rewards = data.reward[:, :2].numpy() if scalar_rewards else data.reward[:, :2, :].numpy()
    if scalar_rewards:
        rewards.reshape(list(rewards.shape) + [1])
    next_states = data.observation[:, 1:, :].numpy()
    next_labels = labeling_function(next_states)
    if next_labels.shape == states.shape[:-1]:
        next_labels.reshape(list(next_labels.shape) + [1])
    state_type = data.step_type[:, :2].numpy()
    next_state_type = data.next_step_type[:, :2].numpy()

    # remove transitions where the incident state is terminal and next state is initial
    # note: such transitions correspond to those where the reset() function has been called
    filtering = state_type[:, 0] != ts.StepType.LAST
    filtering &= state_type[:, 1] != ts.StepType.LAST
    filtering &= next_state_type[:, 0] != ts.StepType.FIRST
    filtering &= next_state_type[:, 1] != ts.StepType.FIRST

    data = {'state': states[filtering], 'action': actions[filtering], 'reward': rewards[filtering],
            'next_state': next_states[filtering], 'next_state_label': next_labels[filtering],
            'state_type': state_type[filtering], 'next_state_type': next_state_type[filtering]}

    if dataset_path is not None:
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with h5py.File(os.path.join(dataset_path, dataset_name + current_time + '.hdf5'), 'w') as h5f:
            h5f['state'] = data['state']
            h5f['action'] = data['action']
            h5f['reward'] = data['reward']
            h5f['next_state'] = data['next_state']
            h5f['next_state_label'] = data['next_state_label']
            h5f['state_type'] = data['state_type']
            h5f['next_state_type'] = data['next_state_type']

    return data


@tf.function
def map_rl_trajectory_to_vae_input(trajectory, labeling_function):
    """
    Maps a tf-agent trajectory of 2 time steps to a transition tuple of the form
    <state, state label, action, reward, next state, next state label>
    """

    state = trajectory.observation[0, ...]
    labels = tf.cast(labeling_function(trajectory.observation), tf.float32)
    if tf.rank(labels) == 1:
        labels = tf.expand_dims(labels, axis=-1)
    label = labels[0, ...]
    action = trajectory.action[0, ...]
    reward = trajectory.reward[0, ...]
    if tf.rank(reward) == 1:
        reward = tf.expand_dims(reward, axis=-1)
    next_state = trajectory.observation[1, ...]
    next_label = labels[1, ...]

    return state, label, action, reward, next_state, next_label


class DictionaryDatasetGenerator:
    """
    Generates a dataset from a dictionary (or hdf5 file) containing the data to be processed.
    """

    def __init__(self, initial_dummy_state=None, initial_dummy_action=None, data: Optional = None):
        self.initial_dummy_state = initial_dummy_state
        self.initial_dummy_action = initial_dummy_action
        self._data = data

    def process_data(self, data: Optional = None):
        if data is None:
            data = self._data
        for (state, action, reward, next_state, label, state_type, next_state_type) in \
                zip(data['state'], data['action'], data['reward'], data['next_state'],
                    data['next_state_label'], data['state_type'], data['next_state_type']):

            if state.shape[:-1] == reward.shape:  # singleton shape
                reward = reward.reshape(list(reward.shape) + [1])
            if state.shape[:-1] == label.shape:
                label = label.reshape(list(label.shape) + [1])

            if state_type[0] == ts.StepType.FIRST:  # initial state handling
                initial_state = self.initial_dummy_state if self.initial_dummy_state is not None \
                    else np.zeros(shape=state.shape[1:])
                initial_action = self.initial_dummy_action if self.initial_dummy_action is not None \
                    else np.zeros(shape=action.shape[1:])
                yield (np.stack((initial_state, state[0])),
                       np.stack((initial_action, action[0])),
                       np.stack((np.zeros(shape=reward.shape[1:]), reward[0])),
                       state,
                       np.stack((label[0], label[0])))

            yield state, action, reward, next_state, label

    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            yield from self.process_data(data=hf)


class RawDatasetGenerator:
    """
    Generates a dataset from raw data contained in a replay buffer.
    """

    def __init__(
            self, replay_buffer, labeling_function,
            initial_dummy_state=None, initial_dummy_action=None, deadlock_action=None,
            scalar_rewards=True, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ):
        self.initial_dummy_state = initial_dummy_state
        self.initial_dummy_action = initial_dummy_action
        self.deadlock_action = deadlock_action
        self.labeling_function = labeling_function
        self.raw_dataset = replay_buffer.as_dataset(
            num_parallel_calls=num_parallel_calls,
            num_steps=3).prefetch(tf.data.experimental.AUTOTUNE)
        self.scalar_rewards = scalar_rewards

    def __call__(self):
        for raw_data, _ in self.raw_dataset:
            yield from map_rl_trajectory_to_vae_input(
                trajectory=raw_data,
                labeling_function=self.labeling_function,
                scalar_rewards=self.scalar_rewards,
                initial_state=self.initial_dummy_state,
                initial_action=self.initial_dummy_action,
                deadlock_action=self.deadlock_action
            )


def get_tensor_shape(data):
    reward_shape = list(data['reward'].shape[1:]) + \
                   ([1] if (tf.TensorShape(data['state'].shape[:-1]) == data['reward'].shape) else [])
    label_shape = list(data['next_state_label'].shape[1:]) + \
                  ([1] if (tf.TensorShape(data['state'].shape[:-1]) == data['next_state_label'].shape) else [])
    return (tf.TensorShape(data['state'].shape[1:]),
            tf.TensorShape(data['action'].shape[1:]),
            tf.TensorShape(reward_shape),
            tf.TensorShape(data['next_state'].shape[1:]),
            tf.TensorShape(label_shape))


def create_dataset(cycle_length=4,
                   block_length=32,
                   num_parallel_calls=tf.data.experimental.AUTOTUNE,
                   hdf5_files_path='dataset/reinforcement_learning',
                   regex='*.hdf5'):
    file_list: List[str] = glob.glob(os.path.join(hdf5_files_path, regex), recursive=True)
    random.shuffle(file_list)
    cycle_length = min(len(file_list), cycle_length)

    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    def tensor_shape(file):
        with h5py.File(file, 'r') as hf:
            return get_tensor_shape(hf)

    dataset = dataset.interleave(
        lambda filename: tf.data.Dataset.from_generator(
            DictionaryDatasetGenerator(),
            (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),  # , tf.int8, tf.int8),
            tensor_shape(file_list[0]),  # all files are assumed to have same Tensor Shape
            args=(filename,)
        ),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=num_parallel_calls
    )
    return dataset


def get_num_samples(hdf5_files_path: str = 'dataset/reinforcement_learning', regex: str = '*.hdf5',
                    drop_remainder: bool = True, batch_size: int = 32):
    file_list: List[str] = glob.glob(os.path.join(hdf5_files_path, regex), recursive=True)
    size = 0
    for h5f_name in file_list:
        with h5py.File(h5f_name, 'r') as h5f:
            # we assume that all h5f datasets have the same length (= size of axis 0)
            h5f_size = h5f['state'].shape[0]
            size += (h5f_size // batch_size) * batch_size if drop_remainder else h5f_size
    return size


def merge_rl_observations_dataset(
        hdf5_files_path='dataset/reinforcement_learning',
        dataset_name='rl_exploration.hdf5',
        regex='*.hdf5',
        shuffle: bool = False):
    file_list: List[str] = glob.glob(os.path.join(hdf5_files_path, regex), recursive=True)

    print("File list:")
    print(file_list)

    length: int = 0
    h5f_indices: Dict[str, Tuple[int, int]] = {}
    shape: Dict[str, Tuple[int, ...]] = {}
    start: float = time.time()

    for h5f_name in file_list:
        with h5py.File(h5f_name, 'r') as h5f:
            # we assume that all h5f datasets have the same length (= size of axis 0)
            h5f_length = h5f['state'].shape[0]
            h5f_indices[h5f_name] = (length, length + h5f_length)
            length += h5f_length

    random_indices = np.arange(length) if shuffle else np.empty(0)
    np.random.shuffle(random_indices)

    with h5py.File(os.path.join(hdf5_files_path, dataset_name), 'w') as merged_h5f:
        for i, h5f_name in enumerate(file_list):
            with h5py.File(h5f_name, 'r') as h5f:
                first, last = h5f_indices[h5f_name]
                indices = np.sort(random_indices[first: last]) if shuffle else np.empty(0)
                for key in h5f:
                    if i == 0:  # dataset file initialization
                        shape[key] = (length,) + h5f[key].shape[1:]
                        merged_h5f.create_dataset(key, shape[key], dtype=h5f[key].dtype)
                    if shuffle:
                        merged_h5f[key][indices] = h5f[key]
                    else:
                        merged_h5f[key][first: last] = h5f[key]

        print("Dataset files merged into {}. Time: {:.3f} sec, size: {:.3f} GB).".format(
            dataset_name, time.time() - start,
                          os.path.getsize(os.path.join(hdf5_files_path, dataset_name)) / 2.0 ** 30))
