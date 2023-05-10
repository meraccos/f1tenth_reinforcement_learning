from stable_baselines3 import PPO
from utils import create_env
import torch
import random

maps = list(range(1, 450))

# random.seed(72)

env = create_env(maps=maps, seed=5)
env.training = False

model = "/Users/meraj/workspace/f1tenth_gym/work/models/exp_250k"

model = PPO.load(path=model)

obs = env.reset()
done = False

# # obs = torch.tensor(obs).unsqueeze(0) 
# actions, values, log_probs, _ = model.policy.forward(obs)
# probs = torch.exp(log_probs)
# action_prob = probs[0][actions[0].item()]
# print(f"Selected action: {actions[0].item()}, probability: {action_prob.item()}")

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human_fast")
