import gym
from gym import spaces

from copy import copy
import numpy as np

from sklearn.neighbors import KDTree

NUM_BEAMS = 2155
DTYPE = np.float32


class LidarRandomizer(gym.ObservationWrapper):
    def __init__(self, env, epsilon=0.05, zone_p=0.1, extreme_p=0.05):
        super().__init__(env)
        self.epsilon = epsilon
        self.zone_p = zone_p
        self.extreme_p = extreme_p

    def observation(self, obs):
        lidar_data = obs["scans"]
        
        # Try normal vs uniform noise
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=lidar_data.shape)
        lidar_data += noise

        # Randomly choose areas to increase/decrease.
        if np.random.random() < self.zone_p:
            # Define size of the area (20% of the readings).
            size = int(len(lidar_data) * 0.2)
            start = np.random.randint(0, len(lidar_data) - size)
            end = start + size
            # Randomly choose whether to increase or decrease, and by how much.
            change = np.random.uniform(-0.1, 0.1)
            lidar_data[start:end] += change

        # Randomly set some readings to very high or very low.
        if np.random.random() < self.extreme_p:
            index = np.random.randint(len(lidar_data))
            lidar_data[index] = np.random.choice([0, 1])

        # Make sure the output is still between 0 and 1.
        lidar_data = np.clip(lidar_data, 0, 1)
        
        
        obs["scans"] = lidar_data

        return obs
    

class ActionRandomizer(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon
    def action(self, action):
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=action.shape)
        action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return action


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, obs):
        vs = obs["linear_vels_s"][self.ego_idx]
        vd = obs["linear_vels_d"][self.ego_idx]
        w = obs["ang_vels_z"]
        d = obs["poses_d"]

        reward = 0.01

        if abs(obs["linear_vels_x"]) <= 0.25:
            reward -= 2.0

        # Encourage the agent to move in the vs direction
        reward += 1.0 * vs
        reward -= 0.01 * abs(vd)

        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 1000.0

        # Minimize d (encourage the agent to stay near the center of the track)
        reward -= 0.05 * abs(d)

        # Angular velocity penalty (discourage sharp turns)
        reward -= 0.05 * abs(w)

        # Penalize the agent for getting too close to walls or obstacles
        min_distance = abs(np.min(obs["scans"]))
        distance_threshold = 0.5
        if min_distance < distance_threshold:
            reward -= 0.01 * (distance_threshold - min_distance)

        return reward

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        info['poses_s'] = obs['poses_s']
        info['collision'] = 1 - self.env.collisions[0]
        info['is_success'] = bool(info['lap_count'][0] >= 1)
        new_reward = self.reward(obs)
        return obs, new_reward.item(), done, info


class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        self.map_data = env.map_data.to_numpy()
        self.kdtree = KDTree(self.map_data[:, 1:3])

        self.observation_space = spaces.Dict(
            {
                "ego_idx": spaces.Box(0, self.num_agents - 1, (1,), np.int32),
                "scans": spaces.Box(0, 1, (NUM_BEAMS,), DTYPE),
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
                "linear_vel": spaces.Box(0, 1, (self.num_agents,), DTYPE),
            }
        )

    def observation(self, obs):
        new_obs = copy(obs)

        frenet_coords = convert_to_frenet(new_obs["poses_x"][0],
                                          new_obs["poses_y"][0],
                                          new_obs["linear_vels_x"], 
                                          new_obs["poses_theta"][0], 
                                          self.map_data, 
                                          self.kdtree
        )
        
        new_obs["poses_s"] = np.array(frenet_coords[0]).reshape((1, -1))
        new_obs["poses_d"] = np.array(frenet_coords[1])
        new_obs["linear_vels_s"] = np.array(frenet_coords[2]).reshape((1, -1))
        new_obs["linear_vels_d"] = np.array(frenet_coords[3])
        
        # Scale the scans and add linear_vel
        clipped_indices = np.where(new_obs["scans"] >= 10)
        noise = np.random.uniform(-0.5, 0, clipped_indices[0].shape)
        
        new_obs["scans"] = np.clip(new_obs["scans"], None, 10)
        new_obs["scans"][clipped_indices] += noise
        new_obs["scans"] /= 10.0

        new_obs["linear_vel"] = new_obs["linear_vels_x"] / 3.2

        return new_obs
    
    
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

    s = -dx * np.sin(psi_rad) + dy * np.cos(psi_rad) + s_m
    d = dx * np.cos(psi_rad) + dy * np.sin(psi_rad)

    vs = vel_magnitude * np.sin(pose_theta - psi_rad)
    vd = vel_magnitude * np.cos(pose_theta - psi_rad)

    return s, d, vs, vd