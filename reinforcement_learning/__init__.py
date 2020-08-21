import tensorflow as tf
import math

labeling_functions = {
    'HumanoidBulletEnv-v0':
        lambda states: states[..., 0] + 0.8 <= 0.78,  # falling down
    # np.count_nonzero(np.abs(observation[:, :, 8: 42][0::2]) > 0.99) > 0  # has stuck joints
    'BipedalWalker-v2':
        lambda states: tf.math.abs(states[..., 0]) > math.pi / 3.  # hull angle too high/low
}
