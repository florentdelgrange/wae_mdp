import tensorflow as tf
import math

labeling_functions = {
    'HumanoidBulletEnv-v0': lambda observation: tf.stack([
            # falling down -- observation[0] is the head position, 0.8 is the initial position
            observation[..., 0] + 0.8 <= 0.78,
            # has stuck joints
            # tf.expand_dims(tf.math.count_nonzero(tf.abs(observation[..., 8: 42][0::2]) > 0.99) > 0, axis=-1),
            # feet contact
            tf.cast(observation[..., -2], tf.bool), tf.cast(observation[..., -1], tf.bool),
            # move forward
            observation[..., 3] > 0.,
        ], axis=-1),
    'BipedalWalker-v2':
        lambda observation: tf.stack([
            tf.math.abs(observation[..., 0]) > math.pi / 3.,  # hull angle too high/low (unsafe flag)
            tf.cast(observation[..., 8], tf.bool),  # contact of the left leg with the ground
            tf.cast(observation[..., 13], tf.bool)  # contact of the right leg with the ground
        ], axis=-1),
    'Pendulum-v0':  # safe labels
        lambda observation: tf.stack([
            # easy: |theta| < 90
            tf.logical_and(observation[..., 0] > 0., tf.abs(observation[..., 1]) < tf.math.sin(math.pi / 2)),
            # soft: |theta| < 30
            tf.logical_and(observation[..., 0] > 0., tf.abs(observation[..., 1]) < tf.math.sin(math.pi / 6)),
            # hard: |theta| < 20
            tf.logical_and(observation[..., 0] > 0., tf.abs(observation[..., 1]) < tf.math.sin(math.pi / 9)),
        ], axis=-1),
    'CartPole-v0':  # safe labels
        lambda observation: tf.stack([
            tf.abs(observation[..., 0]) < 1.5,  # cart position is less than 1.5
            tf.abs(observation[..., 2]) < 0.15,  # pole angle is inferior than 9 degrees
        ], axis=-1),
    'LunarLander-v2':
        lambda observation: tf.stack([
            tf.abs(observation[..., 0]) <= 0.15,  # land along the lunar pad x-position
            tf.abs(observation[..., 0]) >= 0.8,  # close to the edge of the frame
            # close to the lunar pad
            tf.math.logical_and(tf.abs(observation[..., 1]) <= 0.3, tf.abs(observation[..., 0]) <= 0.3),
            tf.abs(observation[..., 1]) <= 0.02,  # land along the lunar pad y-position
            observation[..., 2] == 0.,  # horizontal speed is 0
            observation[..., 3] == 0.,   # vertical speed is 0
            observation[..., 3] <= -0.5,  # fast vertical (landing) speed
            tf.abs(observation[..., 4]) <= math.pi / 3,  # lander angle is safe
            tf.abs(observation[..., 4]) <= math.pi / 6,  # weak lander angle
            observation[..., 5] == 0.,  # angular velocity is 0
            tf.cast(observation[..., 6], dtype=tf.bool),  # left leg ground contact
            tf.cast(observation[..., 7], dtype=tf.bool)   # right leg ground contact
        ], axis=-1),
}

labeling_functions['LunarLanderContinuous-v2'] = labeling_functions['LunarLander-v2']
