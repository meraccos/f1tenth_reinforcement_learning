from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from wrappers import RewardWrapper, FrenetObsWrapper
from wrappers import ReducedObsWrapper, NormalizeActionWrapper

import numpy as np
import gym

NUM_BEAMS = 1080
DTYPE = np.float64

def create_env(maps, seed=5):
    env = gym.make(
        "f110_gym:f110-v0",
        num_agents=1,
        maps=maps,
        seed=seed,
        num_beams=NUM_BEAMS,
    )

    env = FrenetObsWrapper(env)
    env = RewardWrapper(env)
    env = ReducedObsWrapper(env)
    env = NormalizeActionWrapper(env)

    env = Monitor(env, info_keywords=("is_success",))
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    # env = VecFrameStack(env, 3)
    return env


def linear_schedule(initial_learning_rate: float):
    def schedule(progress_remaining: float):
        return initial_learning_rate * progress_remaining

    return schedule
