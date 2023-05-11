import threading
import time

import matplotlib.pyplot as plt

import keyboard
import numpy as np
from utils import create_env

# instantiating the environment
maps = list(range(1, 200))
maps = [5]
env = create_env(maps=maps, flatten=False, domain_randomize=True)

obs = env.reset()

done = False

# Initialize action array
action = np.array([[0.0, 0.0]])

# Initialize control variables
steering_angle = 0.0
velocity = 0.0
delta = 0.05

# Set up the LiDAR data plot
plt.ion()
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

def update_action():
    global action, steering_angle, velocity, delta
    while True:
        # Increase steering angle (right)
        if keyboard.is_pressed("a"):
            steering_angle += delta
            steering_angle = min(1.0, steering_angle)
            action[0, 0] = steering_angle
            print("Action: ", action)

        # Decrease steering angle (left)
        if keyboard.is_pressed("d"):
            steering_angle -= delta
            steering_angle = max(-1.0, steering_angle)
            action[0, 0] = steering_angle
            print("Action: ", action)

        # Increase velocity
        if keyboard.is_pressed("w"):
            velocity += delta
            velocity = min(1.0, velocity)
            action[0, 1] = velocity
            print("Action: ", action)

        # Decrease velocity
        if keyboard.is_pressed("s"):
            velocity -= delta
            velocity = max(-1.0, velocity)
            action[0, 1] = velocity
            print("Action: ", action)

        time.sleep(0.1)


keyboard_thread = threading.Thread(target=update_action)
keyboard_thread.start()

while not done:
    obs, reward, done, info = env.step(action)
    # env.render()
    
    # Get the LiDAR data from obs
    # print(obs)
    # print(obs["scans"])
    lidar_data = obs["scans"]
    print(action[0])
    # lidar_data = np.log10(obs["scans"]+1)
    # lidar_data = np.sqrt(obs["scans"])
    # lidar_data = obs["scans"] = np.sqrt(obs["scans"]+1.0) / 1.5 

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
