import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from f110_gym.envs.base_classes import Integrator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
from reward import NewReward
from copy import copy

from sklearn.neighbors import KDTree

NUM_BEAMS = 1080


def create_env(maps=[0]):
    env = gym.make(
        "f110_gym:f110-v0",
        num_agents=1,
        maps=maps,
        num_beams=NUM_BEAMS,
        integrator=Integrator.RK4,
    )

    env = FrenetObsWrapper(env)
    env = NewReward(env)
    env = ReducedObs(env)
    env = NormalizeActionWrapper(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env


class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=env.action_space.shape, dtype=np.float32
        )

    def denormalize_action(self, normalized_action):
        return self.low + (normalized_action + 1.0) * 0.5 * (self.high - self.low)

    def step(self, action):
        denormalized_action = self.denormalize_action(action)
        next_state, reward, done, info = self.env.step(denormalized_action)
        return next_state, reward, done, info


class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        self.map_data = env.map_csv_data
        self.kdtree = KDTree(self.map_data[:, 1:3])

        self.observation_space = spaces.Dict(
            {
                "ego_idx": spaces.Box(0, self.num_agents - 1, (1,), np.int32),
                "scans": spaces.Box(0, 100, (NUM_BEAMS,), np.float32),
                "poses_x": spaces.Box(-1000, 1000, (self.num_agents,), np.float32),
                "poses_y": spaces.Box(-1000, 1000, (self.num_agents,), np.float32),
                "poses_theta": spaces.Box(-2 * np.pi, 2 * np.pi, (self.num_agents,), np.float32),
                "linear_vels_x": spaces.Box(-10, 10, (self.num_agents,), np.float32),
                "linear_vels_y": spaces.Box(-10, 10, (self.num_agents,), np.float32),
                "ang_vels_z": spaces.Box(-10, 10, (self.num_agents,), np.float32),
                "collisions": spaces.Box(0, 1, (self.num_agents,), np.float32),
                "lap_times": spaces.Box(0, 1e6, (self.num_agents,), np.float32),
                "lap_counts": spaces.Box(0, 9999, (self.num_agents,), np.int32),
                "poses_s": spaces.Box(-1000, 1000, (1,), np.float32),
                "poses_d": spaces.Box(-1000, 1000, (1,), np.float32),
                "linear_vels_s": spaces.Box(-10, 10, (1,), np.float32),
                "linear_vels_d": spaces.Box(-10, 10, (1,), np.float32),
            }
        )

    def observation(self, obs):
        poses_x = obs["poses_x"][0]
        poses_y = obs["poses_y"][0]
        vel_magnitude = obs["linear_vels_x"]
        poses_theta = obs["poses_theta"][0]

        frenet_coords = convert_to_frenet(
            poses_x, poses_y, vel_magnitude, poses_theta, self.map_data, self.kdtree
        )

        self.poses_s = np.array(frenet_coords[0]).reshape((1, -1))

        obs["poses_s"] = np.array(frenet_coords[0]).reshape((1, -1))
        obs["poses_d"] = np.array(frenet_coords[1])
        obs["linear_vels_s"] = np.array(frenet_coords[2]).reshape((1, -1))
        obs["linear_vels_d"] = np.array(frenet_coords[3])

        return obs


class ReducedObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(ReducedObs, self).__init__(env)

        self.observation_space = spaces.Dict(
            {
                "scans": spaces.Box(0, 100, (NUM_BEAMS,), np.float32),
                "linear_vels_x": spaces.Box(-10, 10, (self.num_agents,), np.float32),
                "linear_vels_y": spaces.Box(-10, 10, (self.num_agents,), np.float32),
                "ang_vels_z": spaces.Box(-10, 10, (self.num_agents,), np.float32),
            }
        )

    def observation(self, obs):
        del obs["poses_x"]
        del obs["poses_y"]
        # del obs["linear_vels_x"]
        # del obs["linear_vels_y"]

        del obs["ego_idx"]
        del obs["collisions"]
        del obs["lap_times"]
        del obs["lap_counts"]
        # del obs["ang_vels_z"]
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
        self.prev_s = 0.0
        self.prev_lap_times = 0.0
        self.max_s_frac = 0.0
        self.max_s = 100.0

        self.lap_countss = np.zeros(100, int)
        self.episode_index = 0
        self.success_rate = 0.0

    def _on_step(self) -> bool:
        vec_env = self.locals.get("env")
        env = copy(vec_env.get_attr("env")[0])

        infos = copy(self.locals.get("infos", [{}])[0])
        checkpoint_done = infos.get("checkpoint_done", False)

        if checkpoint_done:
            # Calculate the fraction
            self.max_s_frac = copy(self.prev_s / self.max_s)
            
            if self.prev_lap_times <=20.0 and self.max_s_frac > 0.5:
                self.max_s_frac -= 1.0
            else:
                self.max_s_frac += float(self.prev_lap_counts)

            self.lap_countss[self.episode_index] = infos.get("lap_count", 0)
            self.episode_index = (self.episode_index + 1) % 100
            self.success_rate = np.mean(self.lap_countss >= 1)

        self.prev_s = copy(env.poses_s)
        self.prev_lap_times = copy(env.lap_times)
        self.prev_lap_counts = copy(env.lap_counts)
        self.max_s = copy(env.map_max_s)

        # Save the model
        if self.num_timesteps % self.save_interval == 0:
            self.model.save(f"{self.save_path}_{self.num_timesteps}")

        # Log the track fraction
        self.logger.record("rollout/track_fraction", float(self.max_s_frac))
        self.logger.record("rollout/success_rate", float(self.success_rate))

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
