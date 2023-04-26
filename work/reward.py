import numpy as np
from copy import copy
import gym


class NewReward(gym.Wrapper):
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
        info['collision'] = self.env.collisions[0]
        new_reward = copy(self.reward(obs))
        return obs, new_reward.item(), done, info
