from stable_baselines3 import PPO
from utils import create_env, lidar_to_image
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

maps = list(range(1, 450))

# random.seed(72)

env = create_env(maps=maps, seed=5)
env.training = False

model = "/Users/meraj/workspace/f1tenth_gym/work/models/dr_delayed_100k"

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
    
    
    img = lidar_to_image(obs[0][:-1])
    
    # Convert the image to a numpy array for visualization
    image_array = np.array(img)

    # Plot the image
    plt.imshow(image_array, cmap='gray')
    plt.show()
    # env.render(mode="human_fast")



