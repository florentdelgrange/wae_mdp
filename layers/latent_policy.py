import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from layers.base_models import DiscreteDistributionModel


class LatentPolicyNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input,
            latent_policy_network: tfk.Input,
            number_of_discrete_actions: int,
            relaxed_exp_one_hot_action_encoding: bool = True,
            epsilon: Float = 1e-12,
    ):
        _latent_policy_network = latent_policy_network(latent_state)
        _latent_policy_network = tfkl.Dense(
            units=number_of_discrete_actions,
            activation=None,
            name='latent_policy_exp_one_hot_logits'
        )(_latent_policy_network)
        super(LatentPolicyNetwork, self).__init__(
            inputs=latent_state,
            outputs=_latent_policy_network,
            name='latent_policy_network')
        self.relaxed_exp_one_hot_action_encoding = relaxed_exp_one_hot_action_encoding
        self.epsilon = tf.Variable(epsilon, trainable=False)

    def relaxed_distribution(
            self,
            latent_state: Float,
            temperature: Float
    ) -> tfd.Distribution:
        if self.relaxed_exp_one_hot_action_encoding:
            return tfd.TransformedDistribution(
                distribution=tfd.ExpRelaxedOneHotCategorical(
                    temperature=temperature,
                    logits=self(latent_state),
                    allow_nan_stats=False),
                bijector=tfb.Exp())
        else:
            return tfd.RelaxedOneHotCategorical(
                logits=self(latent_state),
                temperature=temperature,
                allow_nan_stats=False)

    def discrete_distribution(self, latent_state: Float) -> tfd.Distribution:
        logits = self(latent_state)
        if self.relaxed_exp_one_hot_action_encoding:
            relaxed_distribution = tfd.ExpRelaxedOneHotCategorical(
                temperature=1e-5,
                logits=self(latent_state),
                allow_nan_stats=False)
            log_probs = tf.math.log(relaxed_distribution.probs_parameter() + self.epsilon)
            return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False, dtype=self.dtype)
        else:
            return tfd.OneHotCategorical(logits=logits, dtype=self.dtype)

    def get_config(self):
        config = super(LatentPolicyNetwork, self).get_config()
        config.update({"relaxed_exp_one_hot_action_encoding": self.relaxed_exp_one_hot_action_encoding})
        return config
