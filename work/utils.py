from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from wrappers import RewardWrapper, FrenetObsWrapper

from gym.wrappers import FilterObservation
from gym.wrappers import TimeLimit
from gym.wrappers import RescaleAction

import numpy as np
import gym

NUM_BEAMS = 2055
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
    
    env = FilterObservation(env, filter_keys=["scans", "linear_vel"])
    env = TimeLimit(env, max_episode_steps=10)
    env = RescaleAction(env, min_action = np.array([-1.0, 0.0]), 
                             max_action = np.array([1.0, 1.0]))

    env = Monitor(env, info_keywords=("is_success",), filename='./metrics/data')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    # env = VecFrameStack(env, 5)
    return env


def linear_schedule(initial_learning_rate: float):
    def schedule(progress_remaining: float):
        return initial_learning_rate * progress_remaining

    return schedule
