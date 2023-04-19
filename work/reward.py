import numpy as np
from copy import copy
import gym


class NewReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # self.first = 0.0
        # self.second = 0.0
        # self.third = 0.0
        # self.known_lap_count = 0
        # self.current_action = None

    def reward(self, obs):
        new_obs = copy(obs)
        vs = new_obs["linear_vels_s"][self.ego_idx]
        vd = new_obs["linear_vels_d"][self.ego_idx]
        w = new_obs["ang_vels_z"]
        d = new_obs["poses_d"]

        reward = 0

        # Penalize the agent for being slow
        if np.sqrt(vs**2 + vd**2) < 0.25:
            reward -= 1.0

        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 10000.0
        else:
            reward += 1.0

        # Encourage the agent to move in the vs direction
        reward += 2.0 * vs
        reward -= 0.5 * abs(vd)
        
        # Minimize d
        if abs(d) < 0.5:
            reward += 0.1
        else:
            reward -= 0.1
            
        
        # Angular velocity penalty
        reward -= -min(0.5, abs(w))
        
        # Penalize the agent for getting too close to walls or obstacles
        min_distance = np.min(new_obs["scans"])
        distance_threshold = 0.5  # Adjust this value based on your desired safety distance
        if min_distance < distance_threshold:
            reward -= (distance_threshold - min_distance) * 10  # Adjust the penalty weight (10) as needed

        # lap_count = obs['lap_counts'][self.ego_idx]
        # lap_time  = obs['lap_times'][self.ego_idx]

        # if lap_count != self.known_lap_count and lap_time >= 50.0:
        #     self.known_lap_count = lap_count
        #     reward += max(500 - 2.5 * self.first, 500)
        #     print(lap_time)
        # print('reward ', reward)

        return reward

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.current_action = action
        new_reward = self.reward(obs)
        return obs, new_reward.item(), done, info
