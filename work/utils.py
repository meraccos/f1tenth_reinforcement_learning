import gym
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from wrappers import RewardWrapper, FrenetObsWrapper
from wrappers import ReducedObsWrapper, NormalizeActionWrapper

NUM_BEAMS = 1080
DTYPE = np.float64

def create_env(maps=[0], seed=5):
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

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    # env = VecFrameStack(env, 3)
    return env


def linear_schedule(initial_learning_rate: float):
    def schedule(progress_remaining: float):
        return initial_learning_rate * progress_remaining

    return schedule


def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=";")
    return data


def get_closest_point_index(x, y, kdtree):
    _, indices = kdtree.query(np.array([[x, y]]), k=1)
    closest_point_index = indices[0, 0]
    return closest_point_index


def convert_to_frenet(x, y, vel_magnitude, pose_theta, map_data, kdtree):
    closest_point_index = get_closest_point_index(x, y, kdtree)
    closest_point = map_data[closest_point_index]
    s_m, x_m, y_m, psi_rad = closest_point[0:4]

    dx = x - x_m
    dy = y - y_m

    vx = vel_magnitude * np.cos(pose_theta)
    vy = vel_magnitude * np.sin(pose_theta)

    s = -dx * np.sin(psi_rad) + dy * np.cos(psi_rad) + s_m
    d = dx * np.cos(psi_rad) + dy * np.sin(psi_rad)

    vs = -vx * np.sin(psi_rad) + vy * np.cos(psi_rad)
    vd = vx * np.cos(psi_rad) + vy * np.sin(psi_rad)

    return s, d, vs, vd
