import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tf_agents.policies import tf_policy
import tf_agents.trajectories.time_step as ts

from tf_agents.trajectories import policy_step
from tf_agents.typing import types


class OneHotTFPolicyWrapper(tf_policy.TFPolicy):
    """
    Categorical policy wrapper; changes Categorical to OneHotCategorical in tf.float32
    """

    def __init__(self, categorical_policy: tf_policy.TFPolicy,
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec):
        super().__init__(time_step_spec, action_spec)
        self._policy = categorical_policy

    def _distribution(
            self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec
    ) -> policy_step.PolicyStep:
        _step = self._policy.distribution(time_step, policy_state)
        logits = _step.action.logits_parameter()
        return policy_step.PolicyStep(tfd.OneHotCategorical(logits=logits, dtype=tf.float32), _step.state, _step.info)
