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

        reward = 0

        # Penalize the agent for being slow
        if np.sqrt(vs**2 + vd**2) < 0.25:
            reward -= 2.0

        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 1000.0
            # reward -= 1.0
            self.collided = True
        else:
            reward += 1.0
            self.collided = False

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
        distance_threshold = 0.5
        if min_distance < distance_threshold:
            reward -= (distance_threshold - min_distance) * 10

        return reward

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.current_action = action
        new_reward = self.reward(obs)
        return obs, new_reward.item(), done, info
