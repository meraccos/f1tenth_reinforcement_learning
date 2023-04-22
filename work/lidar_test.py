import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from utils import create_env

maps = list(range(1, 250))

env = create_env(maps=maps)
env.training = False

model = "/Users/meraj/workspace/f1tenth_gym/work/models/obs_normalized2_500000"

model = PPO.load(path=model, env=env)

obs = env.reset()
done = False

# Set up the LiDAR data plot
plt.ion()
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Get the LiDAR data from obs
    # lidar_data = np.log10(obs["scans"]+1)
    # lidar_data = np.sqrt(obs["scans"])
    lidar_data = obs["scans"] = np.sqrt(obs["scans"]+1.0) / 1.5 

    # Convert LiDAR data to polar coordinates
    num_angles = lidar_data.size
    full_lidar_angle = np.pi * 270 / 180  # degrees
    angles = np.linspace(full_lidar_angle / 2, -full_lidar_angle / 2, num_angles)

    # Update the LiDAR data plot
    ax.clear()
    ax.plot(angles, lidar_data.flatten(), marker="o", markersize=2, linestyle="None")
    ax.set_title("Real-time LiDAR data")
    ax.set_ylim(0, np.max(lidar_data))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.draw()
    plt.pause(0.00001)

    env.render(mode="human_fast")

# Close the LiDAR data plot
plt.ioff()
plt.close(fig)
