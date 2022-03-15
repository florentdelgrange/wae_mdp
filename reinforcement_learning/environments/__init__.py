from typing import Optional

from tf_agents.typing.types import Sequence, PyEnvWrapper
from gym.envs.registration import register
from tf_agents.environments.wrappers import HistoryWrapper

register(
    id='LunarLanderNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuousNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderRandomInitNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRandomInitNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuousRandomInitNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousRandomInitNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderRandomInit-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRandomInit',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderContinuousRandomInit-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousRandomInit',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderContinuousRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderRandomInitRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRandomInitRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderContinuousRandomInitRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:'
                'LunarLanderContinuousRandomInitRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='PendulumRandomInit-v1',
    entry_point='reinforcement_learning.environments.pendulum:PendulumRandomInit',
    max_episode_steps=150,
)

register(
    id='AcrobotRandomInit-v1',
    entry_point='reinforcement_learning.environments.acrobot:AcrobotEnvRandomInit',
    reward_threshold=-100.0,
    max_episode_steps=500,
)


class EnvironmentLoader:
    def __init__(
            self,
            environment_suite,
            seed=None,
            time_stacked_states=1,
    ):
        self.n = 0
        self.environment_suite = environment_suite
        self.seed = seed
        self.time_stacked_states = time_stacked_states

    def load(self, env_name: str, env_wrappers: Optional[Sequence[PyEnvWrapper]] = ()):
        if self.time_stacked_states:
            env_wrappers = list(env_wrappers) + \
                           [lambda env: HistoryWrapper(env=environment, history_length=self.time_stacked_states)]
        environment = self.environment_suite.load(env_name, env_wrappers)
        if self.seed is not None:
            try:
                environment.seed(self.seed + self.n)
                self.n += 1
            except NotImplementedError:
                print("Environment {} has no seed support.".format(env_name))
        return environment
