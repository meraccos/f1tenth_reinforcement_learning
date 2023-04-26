import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from f110_gym.envs.base_classes import Integrator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from gym import spaces
from reward import NewReward
from copy import copy

from sklearn.neighbors import KDTree

NUM_BEAMS = 1080
DTYPE = np.float32


def create_env(maps=[0], seed=5):
    env = gym.make(
        "f110_gym:f110-v0",
        num_agents=1,
        maps=maps,
        seed=seed,
        num_beams=NUM_BEAMS,
        integrator=Integrator.RK4,
    )

    env = FrenetObsWrapper(env)
    env = NewReward(env)
    env = ReducedObs(env)
    env = NormalizeActionWrapper(env)

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    # num_stack = 3
    # env = VecFrameStack(env, 3)
    return env


# Steering -1 and +1, vel 0 and +1
class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_space = gym.spaces.Box(np.array([-1.0, 0.0]), np.array([1.0, 1.0]), dtype=np.float32)

    def denormalize_action(self, norm_action):
        act_steer = self.low[0] + (norm_action[0] + 1.0) * 0.5 * (self.high[0] - self.low[0])
        act_vel = self.low[1] + norm_action[1] * (self.high[1] - self.low[1])

        act = np.array([act_steer, act_vel])
        return act

    def step(self, action):
        denormalized_action = self.denormalize_action(action)
        next_state, reward, done, info = self.env.step(denormalized_action)
        return next_state, reward, done, info


class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        self.map_data = copy(env.map_csv_data)
        self.kdtree = KDTree(self.map_data[:, 1:3])

        self.observation_space = spaces.Dict(
            {
                "ego_idx": spaces.Box(0, self.num_agents - 1, (1,), np.int32),
                "scans": spaces.Box(0, 100, (NUM_BEAMS,), DTYPE),
                "poses_x": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
                "poses_y": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
                "poses_theta": spaces.Box(
                    -2 * np.pi, 2 * np.pi, (self.num_agents,), DTYPE
                ),
                "linear_vels_x": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "linear_vels_y": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "ang_vels_z": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "collisions": spaces.Box(0, 1, (self.num_agents,), DTYPE),
                "lap_times": spaces.Box(0, 1e6, (self.num_agents,), DTYPE),
                "lap_counts": spaces.Box(0, 999, (self.num_agents,), np.int32),
                "poses_s": spaces.Box(-1000, 1000, (1,), DTYPE),
                "poses_d": spaces.Box(-1000, 1000, (1,), DTYPE),
                "linear_vels_s": spaces.Box(-10, 10, (1,), DTYPE),
                "linear_vels_d": spaces.Box(-10, 10, (1,), DTYPE),
            }
        )

    def observation(self, obs):
        new_obs = copy(obs)
        poses_x = new_obs["poses_x"][0]
        poses_y = new_obs["poses_y"][0]
        vel_magnitude = new_obs["linear_vels_x"]
        poses_theta = new_obs["poses_theta"][0]

        frenet_coords = convert_to_frenet(
            poses_x, poses_y, vel_magnitude, poses_theta, self.map_data, self.kdtree
        )

        self.poses_s = np.array(frenet_coords[0]).reshape((1, -1))

        new_obs["poses_s"] = np.array(frenet_coords[0]).reshape((1, -1))
        new_obs["poses_d"] = np.array(frenet_coords[1])
        new_obs["linear_vels_s"] = np.array(frenet_coords[2]).reshape((1, -1))
        new_obs["linear_vels_d"] = np.array(frenet_coords[3])

        return new_obs


class ReducedObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(ReducedObs, self).__init__(env)

        self.observation_space = spaces.Dict(
            {
                "scans": spaces.Box(0, 1.2, (NUM_BEAMS,), DTYPE),
                "linear_vel": spaces.Box(0, 1, (self.num_agents,), DTYPE),
                "ang_vels_z": spaces.Box(-1.2, 1.2, (self.num_agents,), DTYPE),
            }
        )

    def observation(self, obs):
        # Scale the Observation
        obs["linear_vels_x"] = obs["linear_vels_x"] / 3.2
        obs["linear_vels_y"] = obs["linear_vels_y"] / 3.2
        obs["ang_vels_z"] /= 3.0

        obs["scans"] = np.sqrt(obs["scans"] + 1.0) / 3.0 - 0.40

        # Introduce velocity magnitude as observation
        obs["linear_vel"] = (obs["linear_vels_x"]**2 + obs["linear_vels_x"]**2) ** 0.5

        del obs["poses_x"]
        del obs["poses_y"]

        del obs["ego_idx"]
        del obs["collisions"]
        del obs["lap_times"]
        del obs["lap_counts"]
        del obs["poses_theta"]

        del obs["poses_s"]
        del obs["poses_d"]
        del obs["linear_vels_s"]
        del obs["linear_vels_d"]

        return obs


class TensorboardCallback(BaseCallback):
    def __init__(self, save_interval, save_path, verbose=1):
        super().__init__(verbose)

        self.save_interval = save_interval
        self.save_path = save_path

        self.lap_count_log = np.zeros(100, int)
        self.collision_log = np.ones(100, int)
        self.sfraction_log = np.zeros(100, float)

        self.episode_index = 0

        self.success_rate = 0.0
        self.collision_rate = 1.0
        self.sfraction_rate = 0.0

    def _on_step(self) -> bool:
        infos = copy(self.locals.get("infos", [{}])[0])

        if infos['checkpoint_done']:
            # Calculate the max fraction of the track covered
            s_frac = copy(infos['poses_s'] / infos['max_s'])

            if infos['lap_time'] <= 20.0 and s_frac > 0.5:
                s_frac -= 1.0
            else:
                s_frac += infos['lap_count']

            self.lap_count_log[self.episode_index] = infos["lap_count"]
            self.collision_log[self.episode_index] = infos['collision']
            self.sfraction_log[self.episode_index] = s_frac

            self.episode_index = (self.episode_index + 1) % 100

            self.success_rate = np.mean(self.lap_count_log >= 1)
            self.collision_rate = np.mean(self.collision_log)
            self.sfraction_rate = np.mean(self.sfraction_log)

        # Save the model
        if self.num_timesteps % self.save_interval == 0:
            self.model.save(f"{self.save_path}_{int(self.num_timesteps / 1000)}k")

        # Log the track fraction
        self.logger.record("rollout/track_fraction", float(self.sfraction_rate))
        self.logger.record("rollout/success_rate", float(self.success_rate))
        self.logger.record("rollout/collision_rate", float(self.collision_rate))

        return True


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
