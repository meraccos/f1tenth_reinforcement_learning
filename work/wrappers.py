import gym
import numpy as np

from gym import spaces
from copy import copy

from sklearn.neighbors import KDTree

NUM_BEAMS = 1080
DTYPE = np.float64


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, obs):
        new_obs = copy(obs)
        vs = new_obs["linear_vels_s"][self.ego_idx]
        vd = new_obs["linear_vels_d"][self.ego_idx]
        w = new_obs["ang_vels_z"]
        d = new_obs["poses_d"]

        reward = 0.01

        if (vs**2 + vd**2) ** 0.5 <= 0.25:
            reward -= 2.0

        # Encourage the agent to move in the vs direction
        reward += 1.0 * vs
        reward -= 0.01 * abs(vd)

        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 2000.0

        # Minimize d (encourage the agent to stay near the center of the track)
        reward -= 0.05 * abs(d)

        # Angular velocity penalty (discourage sharp turns)
        reward -= 0.05 * abs(w)

        # Penalize the agent for getting too close to walls or obstacles
        min_distance = abs(np.min(new_obs["scans"]))
        distance_threshold = 0.5
        if min_distance < distance_threshold:
            reward -= 0.01 * (distance_threshold - min_distance)

        return reward

    def step(self, action):
        obs, _, done, info = copy(self.env.step(action))
        info['poses_s'] = obs['poses_s']
        info['collision'] = 1 - self.env.collisions[0]
        info['is_success'] = bool(info['lap_count'][0] >= 1)
        new_reward = copy(self.reward(obs))
        return obs, new_reward.item(), done, info
    
    
# Steering -1 and +1, vel 0 and +1
class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_space = gym.spaces.Box(np.array([-1.0, 0.0]), 
                                           np.array([1.0, 1.0]), 
                                           dtype=np.float64)

    def denormalize_action(self, norm_action):
        act_steer = self.low[0] + (norm_action[0] + 
                                   1.0) * 0.5 * (self.high[0] - self.low[0])
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

        self.map_data = copy(env.map_data.to_numpy())
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


class ReducedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ReducedObsWrapper, self).__init__(env)

        self.observation_space = spaces.Dict(
            {
                "scans": spaces.Box(0, 1.2, (NUM_BEAMS,), DTYPE),
                "linear_vel": spaces.Box(0, 1, (self.num_agents,), DTYPE),
                "ang_vels_z": spaces.Box(-1.2, 1.2, (self.num_agents,), DTYPE),
            }
        )

    def observation(self, obs):
        # Clip the lidar data to 10m, add noise
        clipped_indices = np.where(obs["scans"] >= 10)
        noise = np.random.uniform(-0.5, 0, clipped_indices[0].shape)
        
        obs["scans"] = np.clip(obs["scans"], None, 10)
        obs["scans"][clipped_indices] += noise
        obs["scans"] /= 10.0
        
        obs["ang_vels_z"] /= 3.0

        obs["linear_vel"] = np.sqrt(obs["linear_vels_x"]**2 + 
                                    obs["linear_vels_y"]**2) / 3.2

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
        
        del obs["linear_vels_x"]
        del obs["linear_vels_y"]

        return obs
    
    
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